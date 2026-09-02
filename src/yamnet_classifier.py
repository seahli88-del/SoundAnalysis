"""Thin wrapper around the YAMNet TF-Hub model for environmental sound classification.

YAMNet (Google, trained on AudioSet) maps a 16 kHz mono waveform to scores
over 521 sound event classes (e.g. "Speech", "Dog", "Music", "Glass"). It is
~4 MB and runs comfortably on CPU, which is why it's used here instead of a
heavier model — Hugging Face's free Spaces tier has no GPU.
"""
from __future__ import annotations

import csv
from functools import lru_cache

import numpy as np

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"


@lru_cache(maxsize=1)
def _load():
    # Imported lazily so importing this module (e.g. for unit tests that only
    # touch audio_utils) doesn't force a slow TensorFlow import.
    import tensorflow_hub as hub

    model = hub.load(YAMNET_HANDLE)
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    with open(class_map_path, newline="", encoding="utf-8") as f:
        class_names = [row["display_name"] for row in csv.DictReader(f)]
    return model, class_names


def classify(waveform: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
    """Run YAMNet on a 16kHz mono float32 waveform, return top-k (label, score)."""
    model, class_names = _load()
    if waveform.size == 0:
        return []

    scores, _embeddings, _spectrogram = model(waveform)
    mean_scores = scores.numpy().mean(axis=0)
    top_indices = np.argsort(mean_scores)[::-1][:top_k]
    return [(class_names[i], float(mean_scores[i])) for i in top_indices]
