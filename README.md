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

The XTTS model works best with a clean **6-10 second** clip.

```bash
python trim_audio.py voice_sample.wav voice_sample_trimmed.wav 10
```
