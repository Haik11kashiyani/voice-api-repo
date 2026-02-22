---
title: My Voice Api
emoji: 🗣️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Voice API

A REST API that clones your voice using **Coqui XTTS v2** and generates speech from any text.  
Runs on **Hugging Face Spaces** with Docker.

## Features

- **Best-of-N selection** — generates multiple candidates per request, picks the cleanest one
- **Primer word** — absorbs XTTS v2’s warm-up artifact so speech starts clean from syllable one
- **Short-text padding** — very short inputs (e.g. “Hello”) get trailing filler for stability
- **Automatic cleanup** — silence trimming, edge fades, and temp file removal

---

## Endpoints

| Method | Path        | Description                        |
| ------ | ----------- | ---------------------------------- |
| GET    | `/`         | Welcome message & usage info       |
| GET    | `/health`   | Health check (model + voice sample)|
| GET    | `/docs`     | Interactive Swagger UI             |
| POST   | `/generate` | Generate a WAV audio file          |

### POST `/generate`

**Request body** (JSON):
```json
{
  "text": "Hello, this is a test of the voice API."
}
```

**Response**: `audio/wav` file download.

---

## Quick Test

```bash
# Default Space URL
python test_api.py

# Or point at a custom URL
python test_api.py https://your-space.hf.space
```

---

## Local Development

```bash
pip install -r requirements.txt
# Make sure voice_sample.wav is in the project root
uvicorn app:app --reload --port 7860
```

Then open `http://localhost:7860/docs` in your browser.

---

## Files

| File               | Purpose                                       |
| ------------------ | --------------------------------------------- |
| `app.py`           | FastAPI server with TTS generation endpoint    |
| `test_api.py`      | Smoke-test script (health check + generation)  |
| `trim_audio.py`    | Utility to trim a WAV to N seconds             |
| `requirements.txt` | Python dependencies                            |
| `Dockerfile`       | Container setup for Hugging Face Spaces        |

---

## Trimming Your Voice Sample

The XTTS model works best with a clean **6–10 second** clip.

```bash
python trim_audio.py voice_sample.wav voice_sample_trimmed.wav 10
```

---

## Environment Variables

All optional — sensible defaults are built in.

| Variable | Default | Description |
|---|---|---|
| `VOICE_SAMPLE_PATH` | `voice_sample.wav` | Path to a single reference WAV |
| `VOICE_SAMPLES_DIR` | `voice_samples` | Directory of reference WAVs |
| `TTS_CANDIDATES` | `5` | Number of candidates per request |
| `TTS_PRIMER` | `So,` | Throwaway word prepended to absorb warm-up |
| `TTS_PRIMER_CUT_MS` | `1400` | ms to hard-cut from the start |
| `TTS_START_CUT_MS` | `180` | Additional residual artifact cut |
| `TTS_TEMPERATURE` | `0.35` | XTTS generation temperature |
| `TTS_SPEED` | `1.0` | Speech speed multiplier |
| `TTS_SHORT_TEXT_THRESH` | `15` | Char count below which trailing padding is added |
