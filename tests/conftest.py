"""Shared test fixtures — synthetic WAV files for testing."""

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory):
    """Create a temp directory with synthetic WAV fixtures."""
    d = tmp_path_factory.mktemp("fixtures")

    sr = 44100
    duration = 1.0
    n_samples = int(sr * duration)
    rng = np.random.default_rng(42)

    # float32 stereo
    data_f32 = rng.uniform(-0.5, 0.5, (n_samples, 2)).astype(np.float32)
    sf.write(d / "stereo_float32.wav", data_f32, sr, subtype="FLOAT")

    # int16 stereo
    data_i16 = (rng.uniform(-0.5, 0.5, (n_samples, 2)) * 32767).astype(np.int16)
    sf.write(d / "stereo_int16.wav", data_i16, sr, subtype="PCM_16")

    # mono float32
    data_mono = rng.uniform(-0.5, 0.5, n_samples).astype(np.float32)
    sf.write(d / "mono_float32.wav", data_mono, sr, subtype="FLOAT")

    # silent stereo
    data_silent = np.zeros((n_samples, 2), dtype=np.float32)
    sf.write(d / "silent_stereo.wav", data_silent, sr, subtype="FLOAT")

    return d


@pytest.fixture
def tmp_output(tmp_path):
    """Temp directory for test output."""
    return tmp_path
