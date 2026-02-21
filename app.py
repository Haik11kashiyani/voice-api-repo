import os
import uuid
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
TMP_DIR = os.getenv("TMP_DIR", "/tmp/voice_api")
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "2000"))
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("voice-api")

# ---------------------------------------------------------------------------
# App lifespan — load model once, create temp dir
# ---------------------------------------------------------------------------
tts_model: TTS | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_model
    os.makedirs(TMP_DIR, exist_ok=True)

    if not os.path.isfile(VOICE_SAMPLE):
        logger.warning("Voice sample '%s' not found — /generate will fail until it exists.", VOICE_SAMPLE)

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
    sample_exists = os.path.isfile(VOICE_SAMPLE)
    status = "ok" if (model_ready and sample_exists) else "degraded"
    return {
        "status": status,
        "model_loaded": model_ready,
        "voice_sample_found": sample_exists,
    }


@app.post("/generate")
def generate_audio(request: TextRequest):
    """Generate a WAV file spoken in the cloned voice."""
    # --- pre-flight checks ------------------------------------------------
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model is still loading. Please retry in a moment.")

    if not os.path.isfile(VOICE_SAMPLE):
        raise HTTPException(status_code=500, detail=f"Voice sample file '{VOICE_SAMPLE}' is missing on the server.")

    # --- generate audio ---------------------------------------------------
    # Unique filename per request avoids race conditions under concurrency
    filename = f"{uuid.uuid4().hex}.wav"
    output_path = os.path.join(TMP_DIR, filename)

    logger.info("Generating audio for %d chars of text …", len(request.text))

    try:
        tts_model.tts_to_file(
            text=request.text,
            speaker_wav=VOICE_SAMPLE,
            language="en",
            file_path=output_path,
        )
    except Exception as exc:
        logger.exception("TTS generation failed.")
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {exc}") from exc

    if not os.path.isfile(output_path):
        raise HTTPException(status_code=500, detail="Audio file was not created.")

    logger.info("Audio saved → %s", output_path)
    return FileResponse(output_path, media_type="audio/wav", filename="output.wav")
