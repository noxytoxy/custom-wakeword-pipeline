# Custom Wake Word Training Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-GPU_Accelerated-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://github.com/noxytoxy/custom-wakeword-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/noxytoxy/custom-wakeword-pipeline/actions/workflows/tests.yml)

An end-to-end, locally runnable pipeline for generating datasets, training, and testing custom wake word models for voice assistants (e.g., "Jarvis", "Jessie", "Computer").

Wraps [openWakeWord](https://github.com/dscripka/openWakeWord) engine and [Silero TTS](https://github.com/snakers4/silero-models) into a single CLI tool. No real voice recordings needed — pipeline synthesizes everything and trains an accurate model in minutes.

## Features

- **Zero-Data Start:** Synthesizes positive and adversarial negative datasets via Silero TTS (EN/RU).
- **Audio Augmentation:** Pitch shift, time stretch, Gaussian noise, **background noise mixing** (procedural white/pink/brown noise + 60Hz hum, room tone, fan).
- **Phrase-Level Adversarials:** Supports full-phrase negatives to reduce false positives on conversational speech.
- **Incremental Dataset:** Same parameters → skip regeneration (MD5 cache).
- **Robust Training:** AdamW, ReduceLROnPlateau, gradient clipping, SpecAugment on embeddings, feature-level jitter, early stopping with validation split.
- **Post-Training Metrics:** ROC-AUC, optimal threshold via F1-maximization.
- **ONNX Export:** Auto-export for deployment on Raspberry Pi, Home Assistant, edge devices.
- **Real-time Inference:** Built-in microphone test with configurable detection threshold.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Framework | PyTorch, torchaudio |
| Feature Extraction | openWakeWord (Google Speech Embeddings via ONNX) |
| TTS Synthesis | Silero TTS (via torch.hub) |
| Audio Augmentation | audiomentations (PitchShift, TimeStretch, AddGaussianNoise, AddBackgroundNoise) |
| Audio I/O | soundfile, numpy |
| Live Capture | pyaudio |
| Model Export | ONNX |
| Inference Engine | openWakeWord Model |

### Model Architecture

```
Input: (Batch, 16, 96) — 2 seconds of Google Speech Embeddings
  → Flatten (1536)
  → Linear(1536 → 128) + BatchNorm1d + ReLU + Dropout(0.2)
  → Linear(128 → 1) + Sigmoid
```

Training: AdamW (weight decay 1e-4), ReduceLROnPlateau scheduler, BCELoss, early stopping (patience 7).

## System Requirements

Generation and training load thousands of audio files into memory. The trained model is lightweight (runs on Raspberry Pi).

### Minimum (CPU-only)
- **CPU:** Any modern multi-core (Intel i3 / AMD Ryzen 3 / Apple M1)
- **RAM:** 8 GB *(close heavy apps during feature extraction)*
- **GPU:** None (~10-15 min training)

### Recommended (GPU-accelerated)
- **CPU:** Intel i5 / AMD Ryzen 5+
- **RAM:** 16–32 GB
- **GPU:** NVIDIA 4GB+ VRAM (RTX 3050+, CUDA) or Apple Silicon (MPS)

## Prerequisites & Installation

[Miniconda](https://docs.conda.io/en/latest/miniconda.html) recommended for PyTorch GPU setup.

### Windows / Linux (NVIDIA GPU)

```bash
git clone https://github.com/noxytoxy/custom-wakeword-pipeline.git
cd custom-wakeword-pipeline

conda create -n wakeword python=3.10 -y
conda activate wakeword

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

*(Without NVIDIA GPU: `pip install torch torchaudio`)*

### macOS (Apple Silicon)

```bash
git clone https://github.com/noxytoxy/custom-wakeword-pipeline.git
cd custom-wakeword-pipeline
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio
pip install -e .
```

## Quick Start Guide

All 3 commands in one CLI. Run from project root.

### Step 1: Generate Dataset

Generates positive samples (wake word) + hard negative samples (phonetically similar words/phrases).

```bash
python main.py generate --word "Jessie" --lang "en" --adversarial "Messy,Jessica,Bessie,Restless,Test see,Yes please"

# With phrase-level adversarials (reduces false positives in conversation):
python main.py generate --word "Jessie" --adversarial "Messy,Jessica,Bessie" --adversarial-phrases "Guess I see you,Best seat in house"

# Russian:
python main.py generate --word "Джесси" --lang "ru" --adversarial "Десять,Месси,Сессия"
```

*Note: First run downloads Silero TTS models (~100MB). Also generates noise library in `dataset/noise/`.*

| Argument | Default | Description |
|----------|---------|-------------|
| `--word` | required | Wake word to synthesize |
| `--lang` | `en` | Language (`en` or `ru`) |
| `--adversarial` | `Messy,Jessica,Bessie,...` | Comma-separated phonetic hard-negative words |
| `--adversarial-phrases` | `` | Comma-separated full-phrase negatives |

### Step 2: Train

Automatically downloads background features (~130MB) and openWakeWord base models. Trains with validation, early stopping, and exports ONNX.

```bash
python main.py train --name "jessie"
# Custom epochs:
python main.py train --name "jessie" --epochs 30
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--name` | required | Output model name (produces `<name>.onnx`) |
| `--epochs` | `50` | Max training epochs (early stopping may end sooner) |

Output example:
```
ROC-AUC: 0.9973 | Optimal threshold: 0.4870 (F1: 0.9871)
Model exported to jessie.onnx
Inference: python main.py test --model jessie.onnx --threshold 0.4870
```

### Step 3: Test Live Inference

Real-time microphone test. Use the `--threshold` value printed after training for optimal detection.

```bash
python main.py test --model "jessie.onnx"
# With tuned threshold:
python main.py test --model "jessie.onnx" --threshold 0.4870
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | Path to `.onnx` model |
| `--threshold` | `0.5` | Detection threshold (use value from train output) |

## Docker

For CPU-only execution without installing Python or dependencies:

```bash
docker build -t wakeword .
docker run --rm wakeword generate --word "Jessie"
docker run --rm wakeword train --name "jessie"
docker run --rm -v /dev/snd:/dev/snd --privileged wakeword test --model jessie.onnx
```

Models and dataset are written inside the container by default. To persist them on host, mount a volume:

```bash
docker run --rm -v %cd%:/app/data wakeword train --name "jessie"
```

*Note: GPU acceleration requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Microphone (`test`) needs `--privileged` and host audio device passthrough.*

## Advanced Usage / Troubleshooting

- **False Positives:** Expand `--adversarial` list with more phonetically similar words; use `--adversarial-phrases` for conversation-like negatives.
- **Low Sensitivity:** Place 10–20 real `.wav` recordings in `dataset/positive/` and retrain. Synthetic + real data yields best results.
- **Audio Device:** Test uses default system microphone. Check OS privacy settings if no input detected.
- **Detection Threshold:** After training, the optimal threshold is printed. Use it with `test --threshold` for best precision-recall balance.

## Development

For contributors who want to run tests and linting:

```bash
pip install -e ".[dev]"
pre-commit install  # runs ruff on every commit
pytest              # run tests
```
## Acknowledgements

- Feature extraction architecture from [openWakeWord](https://github.com/dscripka/openWakeWord) by David Scripka (Apache License 2.0).
- Text-to-Speech by [Silero TTS](https://github.com/snakers4/silero-models).

## License

MIT License. Copyright (c) 2026 Maxim Budyakov. See [LICENSE](LICENSE).
