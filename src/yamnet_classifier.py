"""Thin wrapper around the YAMNet model for environmental sound classification.

YAMNet (Google, trained on AudioSet) maps a 16 kHz mono waveform to scores
over 521 sound event classes (e.g. "Speech", "Dog", "Music", "Glass"). It is
~4 MB and runs comfortably on CPU, which is why it's used here instead of a
heavier model — Hugging Face's free Spaces tier has no GPU.

Weights are fetched via `kagglehub` (TensorFlow Hub's tfhub.dev catalog moved
to Kaggle Models in Nov 2023; `hub.load()`'ing the old tfhub.dev URL directly
now 403s because the underlying `tensorflow_hub` HTTP resolver hasn't been
updated to go through Kaggle's CDN). Downloading via `kagglehub` first and
then `hub.load()`'ing the resulting local path works because it uses the
library's plain PathResolver instead.
"""
from __future__ import annotations

import csv
from functools import lru_cache

import numpy as np

YAMNET_KAGGLE_HANDLE = "google/yamnet/tensorFlow2/yamnet"


@lru_cache(maxsize=1)
def _load():
    # Imported lazily so importing this module (e.g. for unit tests that only
    # touch audio_utils) doesn't force a slow TensorFlow import.
    import kagglehub
    import tensorflow_hub as hub

    model_dir = kagglehub.model_download(YAMNET_KAGGLE_HANDLE)
    model = hub.load(model_dir)
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
