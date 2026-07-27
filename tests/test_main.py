import numpy as np
import soundfile as sf


def test_pink_noise():
    from main import _pink_noise

    n = _pink_noise(1000)
    assert n.shape == (1000,)
    assert n.dtype == np.float32
    assert np.all(np.isfinite(n))
    assert abs(np.mean(n)) < 0.1


def test_brown_noise():
    from main import _brown_noise

    n = _brown_noise(1000)
    assert n.shape == (1000,)
    assert n.dtype == np.float32
    assert np.all(np.isfinite(n))


def test_lowpass():
    from main import _lowpass

    sr = 16000
    signal = np.random.randn(1000).astype(np.float32)
    filtered = _lowpass(signal, 100, sr)
    assert filtered.shape == signal.shape
    assert filtered.dtype == np.float32
    assert np.all(np.isfinite(filtered))


def test_generate_noise_files(tmp_path):
    from main import generate_noise_files

    generate_noise_files(str(tmp_path))
    files = list(tmp_path.glob("*.wav"))
    assert len(files) == 6
    for f in files:
        y, sr = sf.read(str(f))
        assert sr == 16000


def test_model_forward():
    import torch

    from main import WakeWordModel

    model = WakeWordModel()
    x = torch.randn(4, 16, 96)
    y = model(x)
    assert y.shape == (4, 1)
    assert bool(torch.all(y >= 0))
    assert bool(torch.all(y <= 1))


def test_model_parameter_count():
    from main import WakeWordModel

    model = WakeWordModel()
    n = sum(p.numel() for p in model.parameters())
    assert n > 10000
    assert n < 500000


def test_config_loaded():
    from main import CFG_AUDIO, CFG_TRAIN, CONFIG

    assert "audio" in CONFIG
    assert CFG_AUDIO["sample_rate"] == 16000
    assert CFG_TRAIN["batch_size"] > 0


def test_noise_files_idempotent(tmp_path):
    from main import generate_noise_files

    generate_noise_files(str(tmp_path))
    generate_noise_files(str(tmp_path))
    files = list(tmp_path.glob("*.wav"))
    assert len(files) == 6
