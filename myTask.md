# myTask — AI Solution Engineering Assignment

**Goal:** Learn the end-to-end workflow of building, versioning, and publishing a real AI application:
**GitHub (source control) → GitHub Actions (CI verify + auto-deploy) → Hugging Face Spaces (public hosting) → Python + a model → live analysis of a real input device.**

- **Level of effort:** ~1 week (part-time alongside normal work)
- **Deadline:** submit **before 8 September**
- **Reference implementation:** this repository, live at
  <https://huggingface.co/spaces/YianXingJian/HighIntensityMachine>
  (a browser app that captures **microphone sound**, extracts frequency features, visualizes them, and raises spike alerts — study its `README.md`, `.github/workflows/deploy-huggingface.yml`, and `app.html`)

---

## 1. What you will build

An application that:

1. Takes a **live input source** of your choice (pick ONE):

   | Track | Input | Suggested model / method |
   |---|---|---|
   | A — Vision | Webcam | **YOLOv8** detecting the 80 COCO object classes (person, cup, phone, …) |
   | B — Audio | Microphone | Audio classification (YAMNet / Audio Spectrogram Transformer) or speech-to-text (**Whisper**), or DSP feature extraction like this project |
   | C — Text/Keyboard | Typed text | Sentiment / toxicity / language detection (Hugging Face `transformers` pipeline) |
   | D — Mouse/Pointer | Mouse movement stream | Anomaly detection on movement features (speed, jitter, path entropy) — e.g. bot-vs-human classifier |
   | E — Network | A network port / packet capture or log stream | Traffic anomaly detection (scikit-learn IsolationForest / autoencoder) |
   | F — Your own idea | Any real input device | Any model — clear it with the assignment owner first |

2. Runs the input through a **model or analysis pipeline in Python**
   (Tracks A–C have ready-made models; D–E train/fit something simple yourself).

3. **Presents the result live** in a web UI — detections drawn on the video, labels with confidence scores, charts, alert banners… make it understandable to a non-engineer in 10 seconds.

4. Is **hosted on Hugging Face Spaces** and **auto-deploys from GitHub on every push to `main`**.

> Recommended stack for Tracks A–E: a **Gradio** app (`sdk: gradio` in the Space README) — it gives you webcam/microphone/text input widgets for free and runs your Python code server-side. This reference project used `sdk: static` (all-JavaScript) instead, which is also acceptable for Track B if you prefer signal processing in the browser — but at least the *tooling* around it (CI checks) must still run Python.

---

## 2. Hard requirements (the checklist we will review)

### Source control (GitHub)
- [ ] Public GitHub repository, default branch `main`
- [ ] Meaningful commit history — **at least 10 commits over multiple days**, each with a descriptive message (one giant "final commit" fails this)
- [ ] A `README.md` that includes: what the app does, the live Space URL, how to run locally, and what data/model it uses
- [ ] `.gitignore` (no `__pycache__`, virtualenvs, model weight blobs, or secrets in the repo)
- [ ] No tokens/credentials anywhere in the code or history

### CI/CD (GitHub Actions)
- [ ] Workflow in `.github/workflows/` that triggers on push to `main`
- [ ] A **verify job** that must pass before deployment: required files exist, `pip install -r requirements.txt` succeeds on a clean runner, all imports work, Python files compile (see this repo's workflow — it caught a missing system library on its first real run)
- [ ] A **deploy job** that pushes to your Space using an `HF_TOKEN` repository secret (never a hardcoded token)
- [ ] At least one screenshot or link showing a **failed run that you then fixed** — we want to see the pipeline actually protecting you

### Hugging Face Space
- [ ] Space under your own HF account, configured via README front matter (`sdk`, `app_file`, etc.)
- [ ] `requirements.txt` with **pinned or bounded versions** of every Python dependency
- [ ] The app loads and works for a stranger with zero instructions beyond what's on screen
- [ ] Handles the "no permission / no device" case gracefully (e.g. camera denied → clear message, not a blank page)

### The application itself
- [ ] Live input → model → visible result, updating continuously or per-capture
- [ ] Shows **model confidence/score**, not just a label
- [ ] At least one "engineering" feature beyond the bare demo, for example: an event log of detections with timestamps, a CSV/JSON export, an alert threshold with a visible warning, a settings control (sensitivity, class filter), or session statistics
- [ ] A short **"How it works"** section in the UI or README explaining the model used and its limitations

---

## 3. Suggested 1-week plan

| Day | Milestone |
|---|---|
| 1 | Create GitHub repo + empty HF Space; get "hello world" auto-deploying end-to-end (this plumbing is half the assignment — do it FIRST, not last) |
| 2 | Input working: webcam/mic/text stream visible in the app |
| 3–4 | Model integrated: real detections/classifications appearing with scores |
| 5 | Presentation layer: overlays/charts, event log, alert threshold |
| 6 | Hardening: CI verify checks, README, error handling, version pinning |
| 7 | Buffer + submission |

**Submit:** reply to the assignment email with (1) GitHub repo URL, (2) live Space URL, (3) 3–5 sentences on the hardest problem you hit and how you solved it.

---

## 4. Grading rubric (100 pts)

| Area | Points | What earns them |
|---|---|---|
| Working live demo on Spaces | 30 | Loads, takes real input, shows model output with scores |
| CI/CD pipeline | 25 | Verify job with real checks + auto-deploy via secret; evidence it caught at least one problem |
| Git craftsmanship | 15 | History tells the story; clean repo; good README |
| Engineering feature | 15 | Log/export/alerts/settings beyond the bare demo |
| UX & explanation | 10 | Readable by a non-engineer; "How it works" present |
| Write-up of hardest problem | 5 | Specific and honest |

Bonus (+10 max): persistence of events across sessions, a second input source, model comparison, or measured latency/FPS stats.

---

## 5. Getting-started pointers

- **YOLOv8 (Track A):** `pip install ultralytics`, then `YOLO("yolov8n.pt")` — the nano model is CPU-friendly, which matters because free Spaces have **no GPU**. Gradio's `gr.Image(sources=["webcam"])` or `gr.Interface` streaming gives you frames.
- **Gradio Space skeleton:** README front matter `sdk: gradio` + `app_file: app.py`; HF builds the Python environment from `requirements.txt` automatically.
- **Auto-deploy:** copy the pattern from this repo's `.github/workflows/deploy-huggingface.yml` — `git push --force https://<user>:${HF_TOKEN}@huggingface.co/spaces/<user>/<space> HEAD:main` after the verify job passes. Create the token at *HF Settings → Access Tokens* with **write** scope and store it as a GitHub Actions secret named `HF_TOKEN`.
- **Free-tier realities:** CPU-only, 16 GB disk, cold starts. Pick small models (`yolov8n`, `whisper-tiny`, MiniLM-class transformers).
- **Study the reference:** this project's history shows the exact bugs you'll likely meet — missing system libraries caught by CI, page-layout issues on different screens, over-sensitive alert thresholds needing a warm-up period and a sensitivity control. Reading its commit log is part of the learning.

## 6. Rules

- You may use AI assistants (Claude, Copilot, ChatGPT) — that's part of modern engineering — but **you must be able to explain every line** in the review session, and the commit history must show your iterative process.
- Don't fork this repo; start clean. Borrowing the workflow YAML pattern is fine and encouraged.
- Any input device beyond the table is welcome — confirm scope first so the LOE stays ~1 week.

Questions → contact the assignment owner. Good luck, and have fun with it — the point is to leave this week knowing you can take *any* model from idea to a public URL with professional plumbing.
