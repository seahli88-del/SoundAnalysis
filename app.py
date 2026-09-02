"""Gradio app: live microphone sound classification with YAMNet.

Captures short audio chunks from the browser's microphone, classifies them
with YAMNet (521 AudioSet classes), and presents the result as:
  - a live confidence bar chart of the top predicted sounds,
  - a current input level (dBFS) readout with a spike/alert banner,
  - a scrollable event log with timestamps,
  - CSV export of the session log,
  - live session statistics.

Run locally:  python app.py
"""
from __future__ import annotations

import os
import tempfile
import time
from collections import Counter

import gradio as gr
import pandas as pd

from src.audio_utils import prepare_chunk
from src.yamnet_classifier import classify

LOG_COLUMNS = ["time", "label", "confidence", "level_db", "alert"]

HOW_IT_WORKS = """
**Model:** [YAMNet](https://tfhub.dev/google/yamnet/1) — a small (~4 MB) MobileNet-based
audio event classifier from Google, trained on the AudioSet corpus (521 everyday sound
classes: speech, music, animals, alarms, mechanical noise, etc.). It runs entirely on
CPU, which is required since Hugging Face's free Spaces tier has no GPU.

**Pipeline:** the browser streams short microphone chunks to the server → each chunk is
converted to mono, resampled to 16 kHz, and its dBFS level is measured → YAMNet scores
the chunk → the top-5 classes and confidences are shown, and an event is logged when the
level crosses the alert threshold or the predicted sound changes.

**Limitations:**
- Trained on general AudioSet clips, not domain-specific audio — uncommon or overlapping
  sounds may be misclassified or produce low confidence.
- dBFS is a relative digital level, not a calibrated sound-pressure (dB SPL) reading — it
  depends on your microphone's gain.
- Short chunks trade classification accuracy for responsiveness; very brief sounds may be
  missed or need a few frames to be recognized.
- No sound is stored or streamed anywhere except transiently for classification.
"""


def _new_state() -> dict:
    return {"log": [], "last_label": None}


def _format_banner(is_alert: bool, level_db: float) -> str:
    if is_alert:
        return f"### 🚨 Loud sound detected — {level_db:.1f} dBFS"
    return f"Level normal — {level_db:.1f} dBFS"


def _format_stats(log: list[dict]) -> str:
    if not log:
        return "_No events logged yet._"
    total = len(log)
    alerts = sum(1 for e in log if e["alert"])
    common = Counter(e["label"] for e in log).most_common(1)[0][0]
    return f"**Session stats:** {total} events logged · {alerts} alerts · most common sound: **{common}**"


def _render(state: dict, label_scores: dict | None, level_db: float | None, is_alert: bool):
    df = pd.DataFrame(list(reversed(state["log"])), columns=LOG_COLUMNS) if state["log"] else pd.DataFrame(columns=LOG_COLUMNS)
    banner = _format_banner(is_alert, level_db) if level_db is not None else "Waiting for microphone input…"
    return label_scores or {}, level_db, banner, df, _format_stats(state["log"])


def process_chunk(new_chunk, state, alert_threshold_db, min_confidence):
    """Gradio streaming callback: runs on every new microphone chunk."""
    state = state or _new_state()

    if new_chunk is None:
        label_scores, level_db, banner, df, stats = _render(state, None, None, False)
        return label_scores, level_db, banner, df, stats, state

    sample_rate, raw_data = new_chunk
    waveform, level_db = prepare_chunk(sample_rate, raw_data)
    predictions = classify(waveform, top_k=5)

    if not predictions:
        label_scores, level_db_out, banner, df, stats = _render(state, None, level_db, False)
        return label_scores, level_db_out, banner, df, stats, state

    top_label, top_confidence = predictions[0]
    is_alert = level_db >= alert_threshold_db
    should_log = top_confidence >= min_confidence and (is_alert or top_label != state["last_label"])

    if should_log:
        state["log"].append(
            {
                "time": time.strftime("%H:%M:%S"),
                "label": top_label,
                "confidence": round(top_confidence, 3),
                "level_db": round(level_db, 1),
                "alert": "⚠️" if is_alert else "",
            }
        )
        state["last_label"] = top_label

    label_scores = {label: confidence for label, confidence in predictions}
    label_scores_out, level_db_out, banner, df, stats = _render(state, label_scores, level_db, is_alert)
    return label_scores_out, level_db_out, banner, df, stats, state


def export_csv(state):
    """Write the session's event log to a temp CSV file for download."""
    if not state or not state["log"]:
        return None
    df = pd.DataFrame(state["log"], columns=LOG_COLUMNS)
    fd, path = tempfile.mkstemp(suffix=".csv", prefix="sound_events_")
    os.close(fd)
    df.to_csv(path, index=False)
    return path


with gr.Blocks(title="Live Sound Classifier") as demo:
    gr.Markdown(
        "# 🎙️ Live Sound Classifier\n"
        "Allow microphone access, then make some noise — speech, music, claps, "
        "typing — and watch the model identify it in real time."
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(sources=["microphone"], streaming=True, label="Microphone input")
            alert_threshold = gr.Slider(-60, 0, value=-25, step=1, label="Alert threshold (dBFS)")
            min_confidence = gr.Slider(0.0, 1.0, value=0.15, step=0.01, label="Min. confidence to log an event")
        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=5, label="Top predicted sounds")
            level_output = gr.Number(label="Current level (dBFS)")
            alert_banner = gr.Markdown("Waiting for microphone input…")

    log_table = gr.Dataframe(headers=LOG_COLUMNS, label="Event log (most recent first)", interactive=False)

    with gr.Row():
        stats_output = gr.Markdown("_No events logged yet._")
        export_btn = gr.Button("⬇️ Export event log as CSV")
    export_file = gr.File(label="Download session log")

    with gr.Accordion("How it works / limitations", open=False):
        gr.Markdown(HOW_IT_WORKS)

    state = gr.State(_new_state())

    audio.stream(
        process_chunk,
        inputs=[audio, state, alert_threshold, min_confidence],
        outputs=[label_output, level_output, alert_banner, log_table, stats_output, state],
    )
    export_btn.click(export_csv, inputs=[state], outputs=[export_file])

demo.queue()

if __name__ == "__main__":
    demo.launch()
