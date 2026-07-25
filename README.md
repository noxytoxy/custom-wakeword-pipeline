# Custom Wake Word Training Pipeline 🎙️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-GPU_Accelerated-ee4c2c.svg)](https://pytorch.org/)

An end-to-end, locally runnable pipeline for generating datasets, training, and testing custom wake word models for voice assistants (e.g., "Jarvis", "Jessie", "Computer"). 

This project wraps the highly efficient [openWakeWord](https://github.com/dscripka/openWakeWord) engine and [Silero TTS](https://github.com/snakers4/silero-models) into a single, easy-to-use CLI tool. You don't need gigabytes of real human voice recordings — the pipeline synthesizes everything you need and trains a highly accurate model in minutes.

## ✨ Features
- **Zero-Data Start:** Automatically synthesizes positive and adversarial negative datasets using PyTorch-based Silero TTS (supports English and Russian).
- **Audio Augmentation:** Applies pitch shifting, time stretching, and Gaussian noise to ensure model robustness in real-world environments.
- **Lightning Fast Training:** Uses GPU-accelerated PyTorch to train a compact neural network in just a few minutes.
- **Real-time Inference:** Built-in microphone testing script with zero-latency detection.
- **ONNX Export:** Automatically exports the trained model to `.onnx` for seamless deployment on Raspberry Pi, Home Assistant, or edge devices.

---

## 💻 System Requirements

This pipeline involves generating thousands of audio files and loading them into memory for feature extraction. While the final trained model is extremely lightweight (can run on a Raspberry Pi), the **training pipeline** requires decent hardware.

### Minimum Requirements (CPU-only workflow)
- **CPU:** Any modern multi-core processor (Intel i3 / AMD Ryzen 3 / Apple M1)
- **RAM:** 8 GB *(Make sure to close heavy apps like web browsers during the feature extraction phase)*
- **GPU:** None required (Training will run on CPU, taking ~10-15 minutes)

### Recommended Requirements (GPU-accelerated workflow)
- **CPU:** Intel i5 / AMD Ryzen 5 or better
- **RAM:** 16 GB - 32 GB *(Ideal for loading large NumPy arrays without system swapping)*
- **GPU:** NVIDIA GPU with 4GB+ VRAM (e.g., RTX 3050, GTX 1650, or better) for CUDA acceleration. Apple Silicon (M1/M2/M3) via MPS is also fully supported.

## ⚙️ Prerequisites & Installation

While you can use a standard Python `venv`, **[Miniconda](https://docs.conda.io/en/latest/miniconda.html) is highly recommended**. It makes installing PyTorch with GPU (CUDA) support much easier and prevents system-level dependency conflicts.

### Windows / Linux (NVIDIA GPU Recommended)

1. Clone the repository and navigate to the directory:
   ```bash
   git clone https://github.com/noxytoxy/custom-wakeword-pipeline.git
   cd custom-wakeword-pipeline
   ```

2. Create and activate a Conda environment:
   ```bash
   conda create -n wakeword python=3.10 -y
   conda activate wakeword
   ```

3. Install PyTorch with CUDA support (for NVIDIA GPUs):
   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   *(If you don't have an NVIDIA GPU, just run: `pip install torch torchaudio`)*

4. Install the pipeline dependencies:
   ```bash
   pip install -e .
   ```

### macOS (Apple Silicon M1/M2/M3)

PyTorch on macOS supports hardware acceleration via MPS.
```bash
git clone https://github.com/noxytoxy/custom-wakeword-pipeline.git
cd custom-wakeword-pipeline

# Create environment (using Conda or standard venv)
python3 -m venv venv
source venv/bin/activate

# Install PyTorch for macOS and dependencies
pip install torch torchaudio
pip install -e .
```

---

## 🚀 Quick Start Guide

The entire pipeline is wrapped in a unified CLI tool: `main.py`. Follow these 3 steps to get your ONNX model.

### Step 1: Generate Dataset
Generate a synthetic dataset for your custom wake word. The script will generate hundreds of augmented positive examples, as well as "Hard Negatives" (words that sound similar, to prevent false positives).

**Example (English):**
```bash
python main.py generate --word "Jessie" --lang "en" --adversarial "Messy,Jessica,Bessie,Jesse"
```

**Example (Russian):**
```bash
python main.py generate --word "Джесси" --lang "ru" --adversarial "Десять,Месси,Сессия"
```
*Note: The first run will download the Silero TTS models (~100MB).*

### Step 2: Train the Model
Train the neural network using the generated features. 
*The script will automatically download the required background noise dataset (~130MB) and openWakeWord base models if they are missing.*

```bash
python main.py train --name "jessie"
```
Once the training finishes (usually 1-3 minutes on a modern GPU), a file named `jessie.onnx` will be saved in your project root.

### Step 3: Test Live Inference
Test your new model in real-time using your microphone to see its accuracy and latency.

```bash
python main.py test --model "jessie.onnx"
```
Speak into your microphone. You will see a live volume meter and the neural network's confidence score. If the wake word is detected, it will trigger a 🔥 notification!

---

## 🛠️ Advanced Usage / Troubleshooting

- **False Positives (Waking up randomly):** Generate more negative examples by passing a longer list of words to `--adversarial`, then retrain.
- **Low Sensitivity (Ignoring you):** You can manually record 10-20 `.wav` files of your own voice saying the wake word, place them in `dataset/positive/`, and retrain the model. The mix of synthetic and real data yields the best results.
- **Audio Device Issues during Test:** The `test` command uses your default system microphone. Ensure your microphone is active and not blocked by OS privacy settings.

## ⚖️ Acknowledgements & Credits
- This project utilizes the core feature extraction architecture from [openWakeWord](https://github.com/dscripka/openWakeWord) by David Scripka (Apache License 2.0).
- Text-to-Speech generation is powered by [Silero TTS](https://github.com/snakers4/silero-models).

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. Copyright (c) 2026 Maxim Budyakov.
