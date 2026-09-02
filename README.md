---
title: Live Sound Classifier
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.26.0"
app_file: app.py
pinned: false
---

# Live Sound Classifier

A browser app that listens to your **microphone**, classifies the sound in real time with
**YAMNet** (Google's AudioSet-trained sound event model), and shows the result as a live
confidence chart, an input-level alert banner, and a timestamped event log.

- **Live demo:** <https://huggingface.co/spaces/seahli/SoundAnalysis>
- **Source:** <https://github.com/seahli88-del/SoundAnalysis>

## What it does

1. Captures short microphone chunks in the browser via Gradio's streaming audio input.
2. Resamples each chunk to 16 kHz mono and measures its level (dBFS).
3. Runs [YAMNet](https://tfhub.dev/google/yamnet/1) to classify the chunk against 521
   AudioSet sound classes (speech, music, animals, mechanical noise, alarms, …).
4. Displays the top-5 predicted classes with confidence scores, the current input level,
   and raises an alert banner when the level crosses a user-adjustable threshold.
5. Logs notable events (level spikes or a change in the dominant sound) with timestamps,
   and lets you export the session log as a CSV.

See the **"How it works / limitations"** section inside the app for model details and
known limitations.

## Data / model used

- **Model:** YAMNet (`google/yamnet/1` on TensorFlow Hub) — pretrained, no fine-tuning.
  Loaded lazily on first use so the app starts quickly and CI doesn't need to download it.
- **Data:** no audio is stored — chunks are classified in memory and discarded. Only the
  derived event log (timestamp, predicted label, confidence, level) is kept for the
  current browser session, and only if you click **Export**.

## Run locally

```bash
git clone https://github.com/seahli88-del/SoundAnalysis.git
cd SoundAnalysis
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
python app.py
```

Then open the local URL Gradio prints (usually `http://127.0.0.1:7860`) and allow
microphone access when prompted.

## CI/CD

`.github/workflows/deploy-huggingface.yml` runs on every push to `main`:

1. **verify** — installs dependencies on a clean runner, checks required files exist,
   compiles all Python files, imports `app.py`, and runs the unit tests in `tests/`.
2. **deploy** — if `verify` passes, force-pushes the repo to the Hugging Face Space using
   an `HF_TOKEN` repository secret (Settings → Secrets and variables → Actions).

## Development notes

This project was built iteratively with AI assistance (GitHub Copilot); see the commit
history for the step-by-step process, and the assignment write-up for the hardest problem
encountered and how it was resolved.
