"""Tests for audio.py — I/O, format conversion, stem summing."""

import numpy as np
import pytest
import soundfile as sf

from mss_aggregate.audio import (
    ensure_float32,
    ensure_stereo,
    read_wav,
    sum_stems,
    write_wav_atomic,
)


class TestReadWav:
    def test_reads_float32_stereo(self, fixtures_dir):
        data, sr = read_wav(fixtures_dir / "stereo_float32.wav")
        assert sr == 44100
        assert data.shape == (44100, 2)
        assert data.dtype == np.float32

    def test_reads_int16_as_float32(self, fixtures_dir):
        data, sr = read_wav(fixtures_dir / "stereo_int16.wav")
        assert sr == 44100
        assert data.dtype == np.float32
        assert data.shape[1] == 2

    def test_reads_mono_as_2d(self, fixtures_dir):
        data, sr = read_wav(fixtures_dir / "mono_float32.wav")
        assert data.ndim == 2
        assert data.shape[1] == 1

    def test_nonexistent_raises(self, tmp_path):
        with pytest.raises(Exception):
            read_wav(tmp_path / "nonexistent.wav")


class TestWriteWavAtomic:
    def test_roundtrip(self, tmp_output):
        data = np.random.default_rng(0).uniform(-0.5, 0.5, (44100, 2)).astype(np.float32)
        out = tmp_output / "test.wav"
        write_wav_atomic(out, data, 44100)
        assert out.exists()
        assert not out.with_suffix(".tmp").exists()
        loaded, sr = sf.read(str(out), dtype="float32", always_2d=True)
        np.testing.assert_allclose(loaded, data, atol=1e-6)

    def test_creates_parent_dirs(self, tmp_output):
        out = tmp_output / "a" / "b" / "test.wav"
        data = np.zeros((100, 2), dtype=np.float32)
        write_wav_atomic(out, data, 44100)
        assert out.exists()

    def test_int16_promoted_to_float32(self, tmp_output):
        data = np.array([[1000, -1000]], dtype=np.int16)
        out = tmp_output / "promoted.wav"
        write_wav_atomic(out, data, 44100)
        info = sf.info(str(out))
        assert info.subtype == "FLOAT"


class TestEnsureStereo:
    def test_mono_1d(self):
        mono = np.ones(100, dtype=np.float32)
        stereo = ensure_stereo(mono)
        assert stereo.shape == (100, 2)
        np.testing.assert_array_equal(stereo[:, 0], stereo[:, 1])

    def test_mono_2d(self):
        mono = np.ones((100, 1), dtype=np.float32)
        stereo = ensure_stereo(mono)
        assert stereo.shape == (100, 2)

    def test_stereo_passthrough(self):
        s = np.ones((100, 2), dtype=np.float32)
        result = ensure_stereo(s)
        assert result is s

    def test_multichannel_raises(self):
        with pytest.raises(ValueError):
            ensure_stereo(np.ones((100, 4), dtype=np.float32))


class TestEnsureFloat32:
    def test_float32_passthrough(self):
        d = np.ones(10, dtype=np.float32)
        assert ensure_float32(d) is d

    def test_float64_to_float32(self):
        d = np.ones(10, dtype=np.float64)
        result = ensure_float32(d)
        assert result.dtype == np.float32

    def test_int16_scaled(self):
        d = np.array([32767, -32768], dtype=np.int16)
        result = ensure_float32(d)
        assert result.dtype == np.float32
        assert abs(result[0] - 1.0) < 0.001
        assert abs(result[1] - (-1.0)) < 0.001

    def test_int32_scaled(self):
        d = np.array([2147483647], dtype=np.int32)
        result = ensure_float32(d)
        assert result.dtype == np.float32
        assert abs(result[0] - 1.0) < 0.001


class TestSumStems:
    def test_single_stem(self):
        s = np.ones((100, 2), dtype=np.float32)
        result = sum_stems([s])
        np.testing.assert_array_equal(result, s)
        assert result is not s  # copy

    def test_two_stems_same_length(self):
        a = np.ones((100, 2), dtype=np.float32) * 0.5
        b = np.ones((100, 2), dtype=np.float32) * 0.3
        result = sum_stems([a, b])
        np.testing.assert_allclose(result, 0.8, atol=1e-6)

    def test_different_lengths_zero_padded(self):
        a = np.ones((100, 2), dtype=np.float32)
        b = np.ones((50, 2), dtype=np.float32)
        result = sum_stems([a, b])
        assert result.shape == (100, 2)
        np.testing.assert_allclose(result[:50], 2.0, atol=1e-6)
        np.testing.assert_allclose(result[50:], 1.0, atol=1e-6)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            sum_stems([])
