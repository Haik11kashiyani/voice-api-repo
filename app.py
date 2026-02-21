import os
import glob
import uuid
import wave
import logging
from contextlib import asynccontextmanager

import numpy as np
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

# Best-of-N: generate N candidates, return the one closest to your voice
NUM_CANDIDATES = int(os.getenv("TTS_NUM_CANDIDATES", "3"))

# Warm-up prefix: a throwaway sentence spoken before the real text.
# The audio for this prefix is measured and cut off, so the model's
# start-of-sequence jitter never reaches the final output.
WARMUP_PREFIX = "One moment please. "

# Fade-in duration (seconds) applied after trimming to smooth any remaining edge
FADE_IN_SEC = float(os.getenv("FADE_IN_SEC", "0.08"))

# XTTS v2 output sample rate
OUTPUT_SR = 24000

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("voice-api")

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _load_wav_as_float(wav_path: str) -> np.ndarray:
    """Load a WAV file and return a mono float32 numpy array in [-1, 1]."""
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

    # Mix to mono if stereo
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples


def _mel_envelope(signal: np.ndarray, sr: int, n_fft: int = 2048, n_bands: int = 40) -> np.ndarray:
    """Compute a mean mel-scale spectral envelope (a compact voice fingerprint)."""
    hop = n_fft // 2
    n_hops = max(1, (len(signal) - n_fft) // hop)

    power_sum = np.zeros(n_fft // 2 + 1)
    window = np.hanning(n_fft)
    for i in range(n_hops):
        frame = signal[i * hop: i * hop + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        spectrum = np.abs(np.fft.rfft(frame * window)) ** 2
        power_sum += spectrum

    avg_power = power_sum / max(n_hops, 1)

    # Group into mel-spaced bands
    freqs = np.linspace(0, sr / 2, len(avg_power))
    mel_freqs = 2595.0 * np.log10(1.0 + freqs / 700.0)
    mel_edges = np.linspace(mel_freqs[0], mel_freqs[-1], n_bands + 2)

    envelope = np.zeros(n_bands)
    for b in range(n_bands):
        mask = (mel_freqs >= mel_edges[b]) & (mel_freqs < mel_edges[b + 2])
        if mask.any():
            envelope[b] = np.mean(avg_power[mask])

    return np.log(envelope + 1e-10)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _reference_envelope(sample_paths: list[str], sr: int) -> np.ndarray:
    """Compute the averaged mel envelope across all reference samples."""
    envelopes = []
    for p in sample_paths:
        sig = _load_wav_as_float(p)
        envelopes.append(_mel_envelope(sig, sr))
    return np.mean(envelopes, axis=0)


def _save_wav(waveform: np.ndarray, file_path: str, sr: int = OUTPUT_SR) -> None:
    """Save a float waveform array as a 16-bit mono WAV."""
    wav = np.array(waveform, dtype=np.float32)
    peak = np.max(np.abs(wav))
    if peak > 0:
        wav = wav / peak * 0.95          # normalise to prevent clipping
    wav_int16 = (wav * 32767).astype(np.int16)

    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(wav_int16.tobytes())


def _trim_start_array(waveform: np.ndarray, sr: int, trim_sec: float) -> np.ndarray:
    """Return the waveform with the first *trim_sec* seconds removed."""
    if trim_sec <= 0:
        return waveform
    frames_to_skip = int(sr * trim_sec)
    if frames_to_skip >= len(waveform):
        return waveform
    return waveform[frames_to_skip:]


def _fade_in(waveform: np.ndarray, sr: int, duration_sec: float) -> np.ndarray:
    """Apply a smooth fade-in over the first *duration_sec* seconds."""
    if duration_sec <= 0:
        return waveform
    n_samples = min(int(sr * duration_sec), len(waveform))
    fade = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    out = waveform.copy()
    out[:n_samples] *= fade
    return out


def _generate_one(model: TTS, text: str, samples: list[str]) -> np.ndarray:
    """Run TTS for *text* and return the waveform as a float32 array."""
    wav_list = model.tts(
        text=text,
        speaker_wav=samples,
        language="en",
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        speed=SPEED,
    )
    return np.array(wav_list, dtype=np.float32)


def _measure_warmup_duration(model: TTS, samples: list[str]) -> float:
    """Generate the warm-up prefix in isolation and return its duration in seconds.

    Called once at startup so we know exactly how many samples to cut.
    """
    wav = _generate_one(model, WARMUP_PREFIX, samples)
    duration = len(wav) / OUTPUT_SR
    return duration


# ---------------------------------------------------------------------------
# Voice sample collection
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


# ---------------------------------------------------------------------------
# App lifespan — load model once, create temp dir
# ---------------------------------------------------------------------------
tts_model: TTS | None = None
voice_samples: list[str] = []
ref_envelope: np.ndarray | None = None
warmup_duration: float = 0.0          # measured at startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_model, voice_samples, ref_envelope, warmup_duration
    os.makedirs(TMP_DIR, exist_ok=True)

    voice_samples = _collect_voice_samples()
    if voice_samples:
        logger.info("Using %d voice sample(s): %s", len(voice_samples), voice_samples)
        ref_envelope = _reference_envelope(voice_samples, OUTPUT_SR)
        logger.info("Reference voice envelope computed.")
    else:
        logger.warning("No voice samples found — /generate will fail until at least one .wav is provided.")

    logger.info("Loading TTS model '%s' …", MODEL_NAME)
    tts_model = TTS(MODEL_NAME).to("cpu")
    logger.info("Model loaded and ready.")

    # Measure warm-up prefix duration so we can trim it precisely later
    if voice_samples:
        logger.info("Measuring warm-up prefix duration …")
        warmup_duration = _measure_warmup_duration(tts_model, voice_samples)
        # Add a small safety margin so we don't leave any prefix residue
        warmup_duration += 0.15
        logger.info("Warm-up prefix duration: %.2fs (with margin).", warmup_duration)

    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="My Voice API",
    description="Generate speech audio that sounds like your cloned voice.",
    version="2.0.0",
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
            "POST /generate": "Send JSON {'text': '...'} to generate audio (best-of-N selection).",
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
        "candidates_per_request": NUM_CANDIDATES,
    }


@app.post("/generate")
def generate_audio(request: TextRequest):
    """Generate a WAV file spoken in the cloned voice.

    Internally generates NUM_CANDIDATES versions and returns the one whose
    spectral envelope is closest to the reference voice.
    """
    # --- pre-flight checks ------------------------------------------------
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model is still loading. Please retry in a moment.")

    if not voice_samples:
        raise HTTPException(status_code=500, detail="No voice sample files found on the server.")

    # --- generate N candidates --------------------------------------------
    logger.info(
        "Generating %d candidate(s) for %d chars | temp=%.2f  top_k=%d  top_p=%.2f  rep=%.1f  speed=%.1f",
        NUM_CANDIDATES, len(request.text),
        TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY, SPEED,
    )

    # Build the full text with warm-up prefix so jitter lands on the throwaway part
    full_text = WARMUP_PREFIX + request.text

    candidates: list[np.ndarray] = []
    for i in range(NUM_CANDIDATES):
        try:
            wav_arr = _generate_one(tts_model, full_text, voice_samples)
            candidates.append(wav_arr)
            logger.info("  Candidate %d/%d generated (%d samples).", i + 1, NUM_CANDIDATES, len(wav_arr))
        except Exception as exc:
            logger.warning("  Candidate %d/%d failed: %s", i + 1, NUM_CANDIDATES, exc)

    if not candidates:
        raise HTTPException(status_code=500, detail="All generation attempts failed.")

    # --- pick the best candidate ------------------------------------------
    if len(candidates) == 1 or ref_envelope is None:
        best_wav = candidates[0]
        logger.info("Using the only available candidate.")
    else:
        best_idx = 0
        best_score = -1.0
        for i, cand in enumerate(candidates):
            cand_env = _mel_envelope(cand, OUTPUT_SR)
            score = _cosine_sim(ref_envelope, cand_env)
            logger.info("  Candidate %d similarity: %.4f", i + 1, score)
            if score > best_score:
                best_score = score
                best_idx = i
        best_wav = candidates[best_idx]
        logger.info("Selected candidate %d (score %.4f).", best_idx + 1, best_score)

    # --- post-processing: cut warm-up prefix + fade-in --------------------
    best_wav = _trim_start_array(best_wav, OUTPUT_SR, warmup_duration)
    logger.info("Trimmed warm-up prefix (%.2fs).", warmup_duration)

    best_wav = _fade_in(best_wav, OUTPUT_SR, FADE_IN_SEC)
    logger.info("Applied %.0fms fade-in.", FADE_IN_SEC * 1000)

    # --- save & return ----------------------------------------------------
    filename = f"{uuid.uuid4().hex}.wav"
    output_path = os.path.join(TMP_DIR, filename)
    _save_wav(best_wav, output_path, OUTPUT_SR)

    logger.info("Audio saved → %s", output_path)
    return FileResponse(output_path, media_type="audio/wav", filename="output.wav")
