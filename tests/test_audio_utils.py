"""Smoke tests run by CI: fast checks that don't need TensorFlow/microphone.

These are intentionally lightweight (no model download) so they run quickly
on every push as part of the CI "verify" job.
"""
import numpy as np

from src.audio_utils import compute_dbfs, prepare_chunk, resample_to_16k, to_mono_float32


def test_to_mono_float32_from_int16():
    stereo = np.array([[100, -100], [200, -200]], dtype=np.int16)
    mono = to_mono_float32(stereo)
    assert mono.dtype == np.float32
    assert mono.shape == (2,)
    assert np.all(np.abs(mono) <= 1.0)


def test_resample_to_16k_changes_length():
    waveform = np.zeros(44100, dtype=np.float32)
    resampled = resample_to_16k(waveform, 44100)
    assert resampled.shape[0] == 16000


def test_resample_to_16k_noop_when_already_target_rate():
    waveform = np.ones(16000, dtype=np.float32)
    resampled = resample_to_16k(waveform, 16000)
    np.testing.assert_array_equal(resampled, waveform)


def test_compute_dbfs_silence_hits_floor():
    silence = np.zeros(1000, dtype=np.float32)
    assert compute_dbfs(silence, floor_db=-80.0) == -80.0


def test_compute_dbfs_full_scale_is_near_zero():
    full_scale = np.ones(1000, dtype=np.float32)
    assert compute_dbfs(full_scale) == 0.0


def test_prepare_chunk_end_to_end():
    sr = 44100
    data = (np.random.rand(sr) * 2 - 1).astype(np.float32)
    waveform, level_db = prepare_chunk(sr, data)
    assert waveform.shape[0] == 16000
    assert -80.0 <= level_db <= 0.0
