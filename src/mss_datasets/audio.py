"""Audio I/O utilities — read, write, format conversion, stem summing."""

import os
from pathlib import Path

import numpy as np
import soundfile as sf


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read WAV file, return (samples, sample_rate).

    Output shape: (n_samples, n_channels). Mono files are returned as (n_samples, 1).
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data, sr


def write_wav_atomic(path: str | Path, data: np.ndarray, sr: int) -> None:
    """Write WAV atomically: write to .tmp then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / (path.name + ".tmp")
    data = ensure_float32(data)
    sf.write(str(tmp_path), data, sr, subtype="FLOAT", format="WAV")
    os.replace(str(tmp_path), str(path))


def ensure_stereo(data: np.ndarray) -> np.ndarray:
    """Convert mono to dual-mono, passthrough stereo."""
    if data.ndim == 1:
        return np.stack([data, data], axis=-1)
    if data.shape[-1] == 1:
        return np.concatenate([data, data], axis=-1)
    if data.shape[-1] == 2:
        return data
    raise ValueError(f"Unexpected channel count: {data.shape[-1]}")


def ensure_float32(data: np.ndarray) -> np.ndarray:
    """Promote int16/int32 to float32. Passthrough float32."""
    if data.dtype == np.float32:
        return data
    if data.dtype == np.float64:
        return data.astype(np.float32)
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    if data.dtype == np.int32:
        return data.astype(np.float32) / 2147483648.0
    raise ValueError(f"Unsupported dtype: {data.dtype}")


def sum_stems(stems: list[np.ndarray]) -> np.ndarray:
    """Sum multiple stem arrays. Zero-pad shorter arrays to match longest."""
    if not stems:
        raise ValueError("No stems to sum")
    if len(stems) == 1:
        return stems[0].copy()

    max_len = max(s.shape[0] for s in stems)
    n_channels = stems[0].shape[-1] if stems[0].ndim > 1 else 1

    result = np.zeros((max_len, n_channels), dtype=np.float32)
    for s in stems:
        s = ensure_float32(s)
        if s.ndim == 1:
            s = s[:, np.newaxis]
        result[: s.shape[0]] += s
    return result
