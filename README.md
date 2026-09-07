---
title: Live Sound Classifier
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: static
app_file: index.html
pinned: false
---

# Live Sound Classifier

A browser-only app that listens to your **microphone**, extracts frequency features locally
with the Web Audio API, and shows a live sound profile, confidence bars, input-level alerts,
and a timestamped event log. Because the Space uses the static SDK, it does not need a
Python server or CPU Basic runtime.

- **Live demo:** <https://huggingface.co/spaces/seahli/SoundAnalysis1>
- **Source:** <https://github.com/seahli88-del/SoundAnalysis>

## What it doess

1. Captures microphone audio entirely in the browser with `getUserMedia`.
2. Uses the Web Audio API to measure amplitude, dominant frequency, spectral centroid,
  and low/mid/high frequency energy.
3. Maps those features to interpretable sound profiles such as speech-like, music-like,
  bass-heavy, bright/noisy, or quiet.
4. Displays profile confidence scores, the current input level,
   and raises an alert banner when the level crosses a user-adjustable threshold.
5. Logs notable events (level spikes or a change in the dominant sound) with timestamps,
   and lets you export the session log as a CSV.

See the **"How it works / limitations"** section inside the app for model details and
known limitations.

## Data / model used

- **Analysis:** browser-side DSP heuristics; there is no server-side model or inference.
- **Data:** no audio is stored — chunks are classified in memory and discarded. Only the
  derived event log (timestamp, predicted label, confidence, level) is kept for the
  current browser session, and only if you click **Export**.

## Run locally

```bash
git clone https://github.com/seahli88-del/SoundAnalysis1.git
cd SoundAnalysis
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
python -m http.server 8000
```

For a local UI preview, run `python -m http.server 8000` in the project folder, then open
`http://127.0.0.1:8000`. Allow microphone access when prompted. A static Space must be
served over HTTPS for microphone permissions; the Hugging Face URL provides that.

## CI/CD

`.github/workflows/deploy-huggingface.yml` runs on every push to `main`:

1. **verify** — checks the static entry files exist and runs the Python utility tests.
2. **deploy** — if `verify` passes, force-pushes the repo to the Hugging Face Space using
   an `HF_TOKEN` repository secret (Settings → Secrets and variables → Actions).

## Development notes

This project was built iteratively with AI assistance (GitHub Copilot); see the commit
history for the step-by-step process, and the assignment write-up for the hardest problem
encountered and how it was resolved.
