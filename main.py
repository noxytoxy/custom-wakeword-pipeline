import os
import time
import glob
import logging
import argparse
import urllib.request
import numpy as np
import soundfile as sf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchaudio.transforms as T
from audiomentations import Compose, PitchShift, TimeStretch, AddGaussianNoise

try:
    from audiomentations import AddBackgroundNoise
    HAS_BG_NOISE = True
except ImportError:
    HAS_BG_NOISE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TARGET_SAMPLES = 32000
SAMPLE_RATE = 16000


class WakeWordModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 96, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# --- Noise generation helpers (#3, #4) ---

def _pink_noise(n_samples: int) -> np.ndarray:
    white = np.random.randn(n_samples).astype(np.float32)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1
    fft = fft / np.sqrt(freqs)
    return np.fft.irfft(fft, n=n_samples).astype(np.float32)


def _brown_noise(n_samples: int) -> np.ndarray:
    white = np.random.randn(n_samples).astype(np.float32)
    brown = np.cumsum(white)
    return (brown - np.mean(brown)).astype(np.float32)


def _lowpass(signal: np.ndarray, cutoff: int, sr: int) -> np.ndarray:
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1 / sr)
    fft[freqs > cutoff] = 0
    return np.fft.irfft(fft, n=len(signal))


def generate_noise_files(output_dir: str = "dataset/noise"):
    os.makedirs(output_dir, exist_ok=True)
    sr = SAMPLE_RATE
    n = 3 * sr

    noises = {
        "white": lambda: np.random.randn(n).astype(np.float32),
        "pink": lambda: _pink_noise(n),
        "brown": lambda: _brown_noise(n),
        "hum": lambda: (np.sin(2 * np.pi * 60 * np.arange(n, dtype=np.float32) / sr) * 0.3
                        + np.random.randn(n).astype(np.float32) * 0.1),
        "room": lambda: _lowpass(np.random.randn(n).astype(np.float32), 800, sr) * 0.5,
        "fan": lambda: _lowpass(np.random.randn(n).astype(np.float32), 300, sr) * 0.5,
    }

    generated = 0
    for name, gen in noises.items():
        path = os.path.join(output_dir, f"{name}.wav")
        if os.path.exists(path):
            continue
        audio = gen()
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.25
        sf.write(path, audio, sr, subtype='PCM_16')
        generated += 1

    if generated:
        logger.info(f"Noise library: {generated} files in {output_dir}/")


def _build_augment_pipeline(noise_dir: str = "dataset/noise"):
    transforms = [
        PitchShift(min_semitones=-4, max_semitones=4, p=0.9),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.9),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    ]
    if HAS_BG_NOISE and os.path.isdir(noise_dir):
        wavs = [f for f in os.listdir(noise_dir) if f.endswith('.wav')]
        if wavs:
            transforms.insert(0, AddBackgroundNoise(sounds_path=noise_dir, min_snr_db=10, max_snr_db=30, p=0.7))
    return Compose(transforms)


def generate_dataset(target_word: str, lang: str, adversarial_words: str, adversarial_phrases: str = ""):
    logger.info(f"Initializing TTS for word: '{target_word}' (Language: {lang})")

    pos_dir = "dataset/positive"
    neg_dir = "dataset/negative"
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    generate_noise_files()
    augment = _build_augment_pipeline()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    speaker_model = f'v4_{lang}' if lang == 'ru' else 'v3_en'

    model_tts, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                  model='silero_tts', language=lang,
                                  speaker=speaker_model)
    model_tts.to(device)
    resampler = T.Resample(orig_freq=48000, new_freq=SAMPLE_RATE).to(device)

    speakers = ["aidar", "baya", "kseniya", "xenia", "eugene"] if lang == 'ru' else ["en_0", "en_1", "en_2", "en_3", "en_4"]

    # 1. Positives
    logger.info("Generating POSITIVE samples...")
    pos_count = 0
    for speaker in speakers:
        try:
            audio = model_tts.apply_tts(text=target_word, speaker=speaker, sample_rate=48000).to(device)
            audio_16k = resampler(audio).cpu().numpy()
            for _ in range(500):
                y_aug = augment(samples=audio_16k, sample_rate=SAMPLE_RATE)
                sf.write(os.path.join(pos_dir, f"pos_{pos_count}.wav"), y_aug, SAMPLE_RATE, subtype='PCM_16')
                pos_count += 1
        except Exception as e:
            logger.error(f"TTS generation error for speaker {speaker}: {e}")

    logger.info(f"Generated {pos_count} positive samples.")

    # 2. Word-level adversarial negatives
    adv_words_list = [w.strip() for w in adversarial_words.split(',') if w.strip()]
    if adv_words_list:
        logger.info(f"Generating NEGATIVE samples for words: {adv_words_list}...")
        neg_count = 0
        for word in adv_words_list:
            for speaker in speakers[:2]:
                try:
                    audio = model_tts.apply_tts(text=word, speaker=speaker, sample_rate=48000).to(device)
                    audio_16k = resampler(audio).cpu().numpy()
                    for _ in range(30):
                        y_aug = augment(samples=audio_16k, sample_rate=SAMPLE_RATE)
                        sf.write(os.path.join(neg_dir, f"neg_{neg_count}.wav"), y_aug, SAMPLE_RATE, subtype='PCM_16')
                        neg_count += 1
                except Exception as e:
                    logger.error(f"Error generating negative '{word}': {e}")
        logger.info(f"Generated {neg_count} negative word samples.")

    # 3. Phrase-level adversarials
    phrase_list = [p.strip() for p in adversarial_phrases.split(',') if p.strip()]
    if phrase_list:
        logger.info(f"Generating PHRASE NEGATIVES: {phrase_list}...")
        phrase_neg_count = 0
        for phrase in phrase_list:
            for speaker in speakers[:2]:
                try:
                    audio = model_tts.apply_tts(text=phrase, speaker=speaker, sample_rate=48000).to(device)
                    audio_16k = resampler(audio).cpu().numpy()
                    for _ in range(20):
                        y_aug = augment(samples=audio_16k, sample_rate=SAMPLE_RATE)
                        sf.write(os.path.join(neg_dir, f"neg_phrase_{phrase_neg_count}.wav"), y_aug, SAMPLE_RATE,
                                 subtype='PCM_16')
                        phrase_neg_count += 1
                except Exception as e:
                    logger.error(f"Error generating phrase negative '{phrase}': {e}")
        logger.info(f"Generated {phrase_neg_count} phrase-level negative samples.")


def download_dependencies():
    bg_file = "background_features.npy"
    if not os.path.exists(bg_file):
        logger.info("Background features not found. Downloading (~130MB)...")
        url = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"
        try:
            urllib.request.urlretrieve(url, bg_file)
            logger.info("Background features downloaded.")
        except Exception as e:
            logger.error(f"Failed to download background features: {e}")
            raise e

    logger.info("Verifying openWakeWord base models...")
    import openwakeword
    openwakeword.utils.download_models()


def train_model(model_name: str, epochs: int = 50):
    download_dependencies()
    from openwakeword.utils import AudioFeatures

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting training on device: {device}")

    extractor = AudioFeatures(device=str(device))

    def load_and_pad(file_path):
        y, _ = sf.read(file_path)
        if y.dtype != np.int16:
            y = (y * 32767).astype(np.int16)
        if len(y) > TARGET_SAMPLES:
            y = y[:TARGET_SAMPLES]
        elif len(y) < TARGET_SAMPLES:
            pad_length = TARGET_SAMPLES - len(y)
            pad_after = np.random.randint(0, min(3200, pad_length + 1))
            pad_before = pad_length - pad_after
            noise_before = np.random.normal(0, 80, pad_before).astype(np.int16)
            noise_after = np.random.normal(0, 80, pad_after).astype(np.int16)
            y = np.concatenate([noise_before, y, noise_after])
        return y

    def get_features(folder):
        files = glob.glob(os.path.join(folder, "*.wav"))
        logger.info(f"Extracting features from {len(files)} files in {folder}...")
        if not files:
            return np.empty((0, 16, 96))
        clips = np.array([load_and_pad(f) for f in files], dtype=np.int16)
        features = extractor.embed_clips(clips, batch_size=128)
        return np.stack(features) if isinstance(features, list) else features

    X_pos_raw = get_features("dataset/positive")
    y_pos_raw = np.ones(len(X_pos_raw), dtype=np.float32)
    X_neg = get_features("dataset/negative")
    y_neg = np.zeros(len(X_neg), dtype=np.float32)

    logger.info("Loading background noise features...")
    X_bg_raw = np.load("background_features.npy")

    bg_samples = min(len(X_pos_raw) * 6, 15000)
    starts = np.random.choice(X_bg_raw.shape[0] - 16, bg_samples, replace=False)
    bg_slices = np.stack([X_bg_raw[s: s + 16] for s in starts])

    n_bg_neg = min(len(bg_slices) // 2, len(X_pos_raw) * 3)
    X_bg_neg = bg_slices[:n_bg_neg]
    y_bg_neg = np.zeros(n_bg_neg, dtype=np.float32)

    # (#3) Feature-level MixUp: positives + background
    n_mix = min(len(bg_slices) - n_bg_neg, len(X_pos_raw) * 3)
    if n_mix > 0 and len(X_pos_raw) > 0:
        X_bg_mix = bg_slices[n_bg_neg:n_bg_neg + n_mix]
        alphas = np.random.uniform(0.6, 0.95, n_mix)
        pos_indices = np.random.randint(0, len(X_pos_raw), n_mix)
        X_mixed = np.zeros((n_mix, 16, 96), dtype=np.float32)
        for i in range(n_mix):
            X_mixed[i] = alphas[i] * X_pos_raw[pos_indices[i]] + (1 - alphas[i]) * X_bg_mix[i]
        y_mixed = np.ones(n_mix, dtype=np.float32)
        X_pos = np.concatenate([X_pos_raw, X_mixed], axis=0)
        y_pos = np.concatenate([y_pos_raw, y_mixed], axis=0)
        logger.info(f"Created {n_mix} mixed positive samples.")
    else:
        X_pos, y_pos = X_pos_raw, y_pos_raw

    total_negs = len(X_neg) + len(X_bg_neg)
    if len(X_pos) > 0:
        oversample_factor = max(1, total_negs // len(X_pos))
        X_pos = np.repeat(X_pos, oversample_factor, axis=0)
        y_pos = np.repeat(y_pos, oversample_factor, axis=0)

    X = np.concatenate([X_pos, X_neg, X_bg_neg], axis=0)
    y = np.concatenate([y_pos, y_neg, y_bg_neg], axis=0)

    shuffle_idx = np.random.permutation(len(X))
    X, y = X[shuffle_idx], y[shuffle_idx]

    n_val = max(1, len(X) // 10)
    X_val, y_val = X[-n_val:], y[-n_val:]
    X_train, y_train = X[:-n_val], y[:-n_val]

    logger.info(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Mixed: {n_mix} | Total: {X.shape[0]}")

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    model = WakeWordModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    logger.info(f"Beginning training loop ({epochs} max epochs)...")
    best_val_loss = float('inf')
    patience = 7
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # (#10) SpecAugment on embeddings
            if np.random.random() < 0.5:
                batch_np = batch_x.cpu().numpy()
                t_mask = np.random.randint(1, 4)
                t_start = np.random.randint(0, max(1, 16 - t_mask))
                batch_np[:, t_start:t_start + t_mask, :] = 0
                f_mask = np.random.randint(1, 8)
                f_start = np.random.randint(0, max(1, 96 - f_mask))
                batch_np[:, :, f_start:f_start + f_mask] = 0
                batch_x = torch.tensor(batch_np, device=device, dtype=torch.float32)

            # (#5) Feature-level jitter (Gaussian noise on embeddings)
            if np.random.random() < 0.3:
                batch_x = batch_x + torch.randn_like(batch_x) * 0.01

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            correct += ((outputs > 0.5).float() == batch_y).sum().item()
            total += batch_y.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
                val_correct += ((outputs > 0.5).float() == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {total_loss/len(train_loader):.4f} "
            f"Train Acc: {100*correct/total:.2f}% | "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {100*val_correct/val_total:.2f}% | "
            f"LR: {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        logger.info(f"Restored best model (val_loss: {best_val_loss:.4f})")

    # Post-training evaluation and threshold tuning
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            all_scores.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(batch_y.numpy().flatten())

    scores = np.array(all_scores)
    labels = np.array(all_labels)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    if n_pos > 0 and n_neg > 0:
        order = np.argsort(scores)
        rank_sum = (np.argsort(order)[labels == 1]).sum() + 1
        auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        auc = max(auc, 1 - auc)

        thresholds = np.linspace(0, 1, 1000)
        best_f1 = 0
        best_thresh = 0.5
        labels_int = labels.astype(int)
        for thresh in thresholds:
            preds = (scores >= thresh).astype(int)
            tp = (preds & labels_int).sum()
            fp = (preds & (1 - labels_int)).sum()
            fn = ((1 - preds) & labels_int).sum()
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        logger.info(f"ROC-AUC: {auc:.4f} | Optimal threshold: {best_thresh:.4f} (F1: {best_f1:.4f})")
    else:
        best_thresh = 0.5

    model.eval()
    dummy_input = torch.randn(1, 16, 96, device=device)
    out_name = f"{model_name}.onnx"
    torch.onnx.export(
        model, dummy_input, out_name,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    logger.info(f"Model exported to {out_name}")
    logger.info(f"Inference: python main.py test --model {out_name} --threshold {best_thresh:.4f}")


def test_inference(model_path: str, threshold: float = 0.5):
    import pyaudio
    from openwakeword.model import Model

    audio = pyaudio.PyAudio()
    mic_stream = audio.open(
        format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
        input=True, frames_per_buffer=1280
    )

    logger.info(f"Loading inference model: {model_path}")
    owwModel = Model(wakeword_models=[model_path], inference_framework="onnx")
    internal_name = list(owwModel.models.keys())[0]

    print("\n" + "=" * 50)
    print(f"MICROPHONE IS LIVE. Threshold: {threshold:.3f}")
    print("Press Ctrl+C to stop.")
    print("=" * 50 + "\n")

    last_detection = 0
    cooldown = 2.0

    try:
        while True:
            data = mic_stream.read(1280, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            bars = "|" * min(int(rms / 100), 30)

            start_time = time.time()
            prediction = owwModel.predict(audio_data)
            latency = (time.time() - start_time) * 1000
            score = prediction[internal_name]

            print(f"\rAudio: {int(rms):4d} {bars:<30} | Confidence: {score*100:5.2f}%", end="")

            if score > threshold and (time.time() - last_detection) > cooldown:
                print(f"\n\nWAKE WORD DETECTED! Confidence: {score*100:.1f}% | Latency: {latency:.1f}ms\n")
                last_detection = time.time()

    except KeyboardInterrupt:
        print("\nStopping inference...")
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        audio.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom Wake Word Generation and Training Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_gen = subparsers.add_parser("generate", help="Generate synthetic audio dataset")
    parser_gen.add_argument("--word", type=str, required=True, help="The wake word to synthesize")
    parser_gen.add_argument("--lang", type=str, choices=["en", "ru"], default="en", help="Language for TTS")
    parser_gen.add_argument("--adversarial", type=str,
                            default="Messy,Jessica,Bessie,Restless,Test see,Yes please",
                            help="Comma-separated phonetic hard-negative words")
    parser_gen.add_argument("--adversarial-phrases", type=str, default="",
                            help="Comma-separated phrase-level adversarials (e.g. 'Guess I see you,Best seat')")

    parser_train = subparsers.add_parser("train", help="Train the wake word model")
    parser_train.add_argument("--name", type=str, required=True, help="Output model name")
    parser_train.add_argument("--epochs", type=int, default=50, help="Max training epochs")

    parser_test = subparsers.add_parser("test", help="Test the model via microphone")
    parser_test.add_argument("--model", type=str, required=True, help="Path to the .onnx model")
    parser_test.add_argument("--threshold", type=float, default=0.5,
                             help="Detection threshold output by train")

    args = parser.parse_args()

    if args.command == "generate":
        generate_dataset(args.word, args.lang, args.adversarial, args.adversarial_phrases)
    elif args.command == "train":
        train_model(args.name, args.epochs)
    elif args.command == "test":
        test_inference(args.model, args.threshold)
