"""Audio helper functions: resampling, normalization, and level metering.

Kept separate from the model/UI code so it can be unit tested without
loading TensorFlow or starting a Gradio server.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

YAMNET_SAMPLE_RATE = 16000


def to_mono_float32(data: np.ndarray) -> np.ndarray:
    """Convert an arbitrary-shape/dtype audio array to mono float32 in [-1, 1]."""
    if data.ndim > 1:
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)

    return data


def resample_to_16k(waveform: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample a mono float32 waveform to the 16 kHz YAMNet expects."""
    if orig_sr == YAMNET_SAMPLE_RATE:
        return waveform
    if waveform.size == 0:
        return waveform
    return resample_poly(waveform, YAMNET_SAMPLE_RATE, orig_sr).astype(np.float32)


def compute_dbfs(waveform: np.ndarray, floor_db: float = -80.0) -> float:
    """Root-mean-square level of a waveform in dBFS (0 dB = full scale)."""
    if waveform.size == 0:
        return floor_db
    rms = float(np.sqrt(np.mean(np.square(waveform))))
    if rms <= 0:
        return floor_db
    return max(20.0 * np.log10(rms), floor_db)


def prepare_chunk(sr: int, data: np.ndarray) -> tuple[np.ndarray, float]:
    """Full pipeline: raw mic chunk -> (16kHz mono float32 waveform, dBFS level)."""
    mono = to_mono_float32(np.asarray(data))
    level_db = compute_dbfs(mono)
    waveform = resample_to_16k(mono, sr)
    return waveform, level_db
