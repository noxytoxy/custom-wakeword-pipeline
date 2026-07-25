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

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TARGET_SAMPLES = 32000
SAMPLE_RATE = 16000

# --- Neural Network Definition ---
class WakeWordModel(nn.Module):
    """Custom compact neural network for wake word detection."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 96, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

# --- CLI Commands implementation ---
def generate_dataset(target_word: str, lang: str, adversarial_words: str):
    """Generates synthetic positive and negative datasets using PyTorch Silero TTS."""
    logger.info(f"Initializing TTS for word: '{target_word}' (Language: {lang})")
    
    pos_dir = "dataset/positive"
    neg_dir = "dataset/negative"
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    speaker_model = f'v4_{lang}' if lang == 'ru' else 'v3_en'
    
    model_tts, _ = torch.hub.load(repo_or_dir='snakers4/silero-models', 
                                  model='silero_tts', language=lang, 
                                  speaker=speaker_model)
    model_tts.to(device)
    resampler = T.Resample(orig_freq=48000, new_freq=SAMPLE_RATE).to(device)
    
    augment = Compose([
        PitchShift(min_semitones=-4, max_semitones=4, p=0.9),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.9),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    ])

    speakers = ["aidar", "baya", "kseniya", "xenia", "eugene"] if lang == 'ru' else ["en_0", "en_1", "en_2", "en_3", "en_4"]
    
    # 1. Generate Positives
    logger.info("Generating POSITIVE samples...")
    pos_count = 0
    for speaker in speakers:
        try:
            audio = model_tts.apply_tts(text=target_word, speaker=speaker, sample_rate=48000).to(device)
            audio_16k = resampler(audio).cpu().numpy()
            
            for _ in range(200):
                y_aug = augment(samples=audio_16k, sample_rate=SAMPLE_RATE)
                sf.write(os.path.join(pos_dir, f"pos_{pos_count}.wav"), y_aug, SAMPLE_RATE, subtype='PCM_16')
                pos_count += 1
        except Exception as e:
            logger.error(f"TTS generation error for speaker {speaker}: {e}")
            
    logger.info(f"Successfully generated {pos_count} positive samples.")

    # 2. Generate Hard Negatives
    adv_words_list = [w.strip() for w in adversarial_words.split(',') if w.strip()]
    if not adv_words_list:
        logger.warning("No adversarial words provided. Skipping negative generation.")
        return

    logger.info(f"Generating NEGATIVE samples for words: {adv_words_list}...")
    neg_count = 0
    for word in adv_words_list:
        for speaker in speakers[:2]: # Use fewer speakers for negatives to save time
            try:
                audio = model_tts.apply_tts(text=word, speaker=speaker, sample_rate=48000).to(device)
                audio_16k = resampler(audio).cpu().numpy()
                for _ in range(30):
                    y_aug = augment(samples=audio_16k, sample_rate=SAMPLE_RATE)
                    sf.write(os.path.join(neg_dir, f"neg_{neg_count}.wav"), y_aug, SAMPLE_RATE, subtype='PCM_16')
                    neg_count += 1
            except Exception as e:
                logger.error(f"Error generating negative '{word}': {e}")
                
    logger.info(f"Successfully generated {neg_count} negative samples.")


def download_dependencies():
    """Ensures background noise and openWakeWord models are present."""
    bg_file = "background_features.npy"
    if not os.path.exists(bg_file):
        logger.info("Background features not found. Downloading (~130MB)... This may take a moment.")
        url = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"
        try:
            urllib.request.urlretrieve(url, bg_file)
            logger.info("Background features downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download background features: {e}")
            raise e

    logger.info("Verifying openWakeWord base models...")
    import openwakeword
    openwakeword.utils.download_models()


def train_model(model_name: str):
    """Trains the wake word model using generated features."""
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
            y = y[-TARGET_SAMPLES:]
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

    # Prepare datasets
    X_pos = get_features("dataset/positive")
    y_pos = np.ones(len(X_pos), dtype=np.float32)

    X_neg = get_features("dataset/negative")
    y_neg = np.zeros(len(X_neg), dtype=np.float32)
    
    logger.info("Loading background noise features...")
    X_bg_raw = np.load("background_features.npy")
    # Take 12000 random samples from background noise
    starts = np.random.choice(X_bg_raw.shape[0] - 16, 12000, replace=False)
    X_bg = np.stack([X_bg_raw[s : s + 16] for s in starts])
    y_bg = np.zeros(len(X_bg), dtype=np.float32)

    X = np.concatenate([X_pos, X_neg, X_bg], axis=0)
    y = np.concatenate([y_pos, y_neg, y_bg], axis=0)
    
    logger.info(f"Total training samples: {X.shape[0]}")
    
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1))
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = WakeWordModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    logger.info("Beginning training loop...")
    epochs = 22
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += ((outputs > 0.5).float() == batch_y).sum().item()
            total += batch_y.size(0)
            
        logger.info(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(dataloader):.4f} - Acc: {100 * correct / total:.2f}%")

    model.eval()
    dummy_input = torch.randn(1, 16, 96, device=device)
    out_name = f"{model_name}.onnx"
    torch.onnx.export(model, dummy_input, out_name, input_names=["input"], output_names=["output"], dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
    logger.info(f"Model successfully exported to {out_name}")


def test_inference(model_path: str):
    """Tests the compiled ONNX model using live microphone input."""
    import pyaudio
    from openwakeword.model import Model
    
    audio = pyaudio.PyAudio()
    mic_stream = audio.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=1280)
    
    logger.info(f"Loading inference model: {model_path}")
    owwModel = Model(wakeword_models=[model_path], inference_framework="onnx")
    internal_name = list(owwModel.models.keys())[0]
    
    print("\n" + "="*50)
    print("🎙️ MICROPHONE IS LIVE. Speak your wake word!")
    print("Press Ctrl+C to stop.")
    print("="*50 + "\n")
    
    last_detection = 0
    cooldown = 2.0
    
    try:
        while True:
            data = mic_stream.read(1280, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # CLI visualizer
            rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
            bars = "|" * min(int(rms / 100), 30)
            
            start_time = time.time()
            prediction = owwModel.predict(audio_data)
            latency = (time.time() - start_time) * 1000
            score = prediction[internal_name]
            
            print(f"\r🔊 Audio: {int(rms):4d} {bars:<30} | Confidence: {score*100:5.2f}%", end="")
            
            if score > 0.5 and (time.time() - last_detection) > cooldown:
                print(f"\n\n🔥 WAKE WORD DETECTED! Confidence: {score*100:.1f}% | Latency: {latency:.1f}ms\n")
                last_detection = time.time()
                
    except KeyboardInterrupt:
        print("\nStopping inference...")
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        audio.terminate()

# --- Main Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom Wake Word Generation and Training Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Generate Dataset command
    parser_gen = subparsers.add_parser("generate", help="Generate synthetic audio dataset")
    parser_gen.add_argument("--word", type=str, required=True, help="The wake word to synthesize (e.g., 'Jessie')")
    parser_gen.add_argument("--lang", type=str, choices=["en", "ru"], default="en", help="Language for TTS (en or ru)")
    parser_gen.add_argument("--adversarial", type=str, default="Messy,Jessica,Bessie", 
                            help="Comma-separated list of hard negative words")

    # Train command
    parser_train = subparsers.add_parser("train", help="Train the wake word model")
    parser_train.add_argument("--name", type=str, required=True, help="Output model name (e.g., 'jessie')")

    # Test command
    parser_test = subparsers.add_parser("test", help="Test the model via microphone")
    parser_test.add_argument("--model", type=str, required=True, help="Path to the .onnx model (e.g., 'jessie.onnx')")

    args = parser.parse_args()

    if args.command == "generate":
        generate_dataset(args.word, args.lang, args.adversarial)
    elif args.command == "train":
        train_model(args.name)
    elif args.command == "test":
        test_inference(args.model)