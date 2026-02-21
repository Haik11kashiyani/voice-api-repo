import os
import glob
import uuid
import wave
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from TTS.api import TTS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
os.environ["COQUI_TOS_AGREED"] = "1"

VOICE_SAMPLE = os.getenv("VOICE_SAMPLE_PATH", "voice_sample.wav")
VOICE_SAMPLES_DIR = os.getenv("VOICE_SAMPLES_DIR", "voice_samples")
TMP_DIR = os.getenv("TMP_DIR", "/tmp/voice_api")
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "2000"))
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# XTTS v2 generation tuning
TEMPERATURE = float(os.getenv("TTS_TEMPERATURE", "0.15"))        # lower = closer to reference voice
TOP_K = int(os.getenv("TTS_TOP_K", "30"))                        # narrower sampling
TOP_P = float(os.getenv("TTS_TOP_P", "0.75"))                    # nucleus sampling threshold
REPETITION_PENALTY = float(os.getenv("TTS_REP_PENALTY", "5.0"))  # avoid repetition glitches
SPEED = float(os.getenv("TTS_SPEED", "1.0"))                     # 1.0 = normal speed

# Post-processing: trim initial jitter
TRIM_START_SEC = float(os.getenv("TRIM_START_SEC", "0.35"))      # seconds to cut from the beginning

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("voice-api")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_voice_samples() -> list[str]:
    """Return a list of voice sample WAV paths for speaker embedding.

    XTTS v2 produces a better speaker embedding when given multiple reference
    clips.  We look for:
      1. All .wav files in VOICE_SAMPLES_DIR  (if the folder exists)
      2. Fall back to the single VOICE_SAMPLE file
    """
    samples: list[str] = []

    if os.path.isdir(VOICE_SAMPLES_DIR):
        samples = sorted(glob.glob(os.path.join(VOICE_SAMPLES_DIR, "*.wav")))

    if not samples and os.path.isfile(VOICE_SAMPLE):
        samples = [VOICE_SAMPLE]

    return samples


def _trim_start(wav_path: str, trim_sec: float) -> None:
    """Trim the first *trim_sec* seconds from a WAV file **in-place**.

    This removes the initial jitter / artefacts that XTTS sometimes produces.
    """
    if trim_sec <= 0:
        return

    with wave.open(wav_path, "rb") as rf:
        params = rf.getparams()
        n_channels = params.nchannels
        sampwidth = params.sampwidth
        framerate = params.framerate
        n_frames = params.nframes
        all_data = rf.readframes(n_frames)

    frames_to_skip = int(framerate * trim_sec)
    if frames_to_skip >= n_frames:
        return  # nothing left — keep original

    bytes_per_frame = n_channels * sampwidth
    trimmed_data = all_data[frames_to_skip * bytes_per_frame:]

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(trimmed_data)


# ---------------------------------------------------------------------------
# App lifespan — load model once, create temp dir
# ---------------------------------------------------------------------------
tts_model: TTS | None = None
voice_samples: list[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_model, voice_samples
    os.makedirs(TMP_DIR, exist_ok=True)

    voice_samples = _collect_voice_samples()
    if voice_samples:
        logger.info("Using %d voice sample(s): %s", len(voice_samples), voice_samples)
    else:
        logger.warning("No voice samples found — /generate will fail until at least one .wav is provided.")

    logger.info("Loading TTS model '%s' …", MODEL_NAME)
    tts_model = TTS(MODEL_NAME).to("cpu")
    logger.info("Model loaded and ready.")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="My Voice API",
    description="Generate speech audio that sounds like your cloned voice.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class TextRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="The text to convert to speech.",
        json_schema_extra={"example": "Hello, this is a test of the voice API."},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {
        "message": "Welcome to My Voice API! The server is running.",
        "endpoints": {
            "POST /generate": "Send JSON {'text': '...'} to generate audio.",
            "GET  /health": "Health-check endpoint.",
        },
    }


@app.get("/health")
def health_check():
    """Quick health-check so monitoring / Docker HEALTHCHECK can ping the service."""
    model_ready = tts_model is not None
    sample_count = len(voice_samples)
    status = "ok" if (model_ready and sample_count > 0) else "degraded"
    return {
        "status": status,
        "model_loaded": model_ready,
        "voice_samples": sample_count,
    }


@app.post("/generate")
def generate_audio(request: TextRequest):
    """Generate a WAV file spoken in the cloned voice."""
    # --- pre-flight checks ------------------------------------------------
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model is still loading. Please retry in a moment.")

    if not voice_samples:
        raise HTTPException(status_code=500, detail="No voice sample files found on the server.")

    # --- generate audio ---------------------------------------------------
    # Unique filename per request avoids race conditions under concurrency
    filename = f"{uuid.uuid4().hex}.wav"
    output_path = os.path.join(TMP_DIR, filename)

    logger.info("Generating audio for %d chars of text …", len(request.text))
    logger.info(
        "Params: temperature=%.2f  top_k=%d  top_p=%.2f  rep_penalty=%.1f  speed=%.1f",
        TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY, SPEED,
    )

    try:
        tts_model.tts_to_file(
            text=request.text,
            speaker_wav=voice_samples,
            language="en",
            file_path=output_path,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            speed=SPEED,
        )
    except Exception as exc:
        logger.exception("TTS generation failed.")
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {exc}") from exc

    if not os.path.isfile(output_path):
        raise HTTPException(status_code=500, detail="Audio file was not created.")

    # --- post-processing: trim jittery start ------------------------------
    try:
        _trim_start(output_path, TRIM_START_SEC)
        logger.info("Trimmed first %.2fs from output.", TRIM_START_SEC)
    except Exception:
        logger.warning("Post-processing trim failed — returning untrimmed audio.", exc_info=True)

    logger.info("Audio saved → %s", output_path)
    return FileResponse(output_path, media_type="audio/wav", filename="output.wav")
