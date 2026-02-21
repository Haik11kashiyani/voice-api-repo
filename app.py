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
TEMPERATURE = float(os.getenv("TTS_TEMPERATURE", "0.15"))
TOP_K = int(os.getenv("TTS_TOP_K", "30"))
TOP_P = float(os.getenv("TTS_TOP_P", "0.75"))
REPETITION_PENALTY = float(os.getenv("TTS_REP_PENALTY", "5.0"))
SPEED = float(os.getenv("TTS_SPEED", "1.0"))

# Best-of-N: generate N candidates, pick the closest to reference voice
NUM_CANDIDATES = int(os.getenv("TTS_NUM_CANDIDATES", "3"))

# Fade-in applied to final output (milliseconds)
FADE_IN_MS = float(os.getenv("FADE_IN_MS", "50"))

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

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples


def _mel_envelope(signal: np.ndarray, sr: int, n_fft: int = 2048, n_bands: int = 40) -> np.ndarray:
    """Compute a mean mel-scale spectral envelope (voice fingerprint)."""
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
    """Average mel envelope across all reference voice samples."""
    envelopes = []
    for p in sample_paths:
        sig = _load_wav_as_float(p)
        envelopes.append(_mel_envelope(sig, sr))
    return np.mean(envelopes, axis=0)


def _save_wav(waveform: np.ndarray, file_path: str, sr: int = OUTPUT_SR) -> None:
    """Save a float32 waveform as a 16-bit mono WAV."""
    wav = np.array(waveform, dtype=np.float32)
    peak = np.max(np.abs(wav))
    if peak > 0:
        wav = wav / peak * 0.95
    wav_int16 = (wav * 32767).astype(np.int16)

    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(wav_int16.tobytes())


def _find_speech_start(waveform: np.ndarray, sr: int,
                       frame_ms: int = 20,
                       energy_threshold: float = 0.005,
                       required_frames: int = 3,
                       lookback_ms: int = 30) -> int:
    """Find the sample index where real speech begins.

    Scans the waveform in small frames.  Once *required_frames* consecutive
    frames exceed *energy_threshold* (RMS), we mark that as speech onset and
    step back by *lookback_ms* so we keep the natural attack of the first
    consonant/vowel.

    Returns the sample index to start from (0 if no leading jitter found).
    """
    frame_len = int(sr * frame_ms / 1000)
    if frame_len == 0:
        return 0

    consecutive = 0
    for start in range(0, len(waveform) - frame_len, frame_len):
        frame = waveform[start: start + frame_len]
        rms = float(np.sqrt(np.mean(frame ** 2)))

        if rms >= energy_threshold:
            consecutive += 1
            if consecutive >= required_frames:
                onset = start - (required_frames - 1) * frame_len
                lookback_samples = int(sr * lookback_ms / 1000)
                cut_point = max(0, onset - lookback_samples)
                return cut_point
        else:
            consecutive = 0

    return 0


def _fade_in(waveform: np.ndarray, sr: int, duration_ms: float) -> np.ndarray:
    """Apply a smooth fade-in over the first *duration_ms* milliseconds."""
    if duration_ms <= 0:
        return waveform
    n_samples = min(int(sr * duration_ms / 1000), len(waveform))
    fade = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    out = waveform.copy()
    out[:n_samples] *= fade
    return out


def _clean_start(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Trim any leading jitter/noise and apply a gentle fade-in."""
    cut = _find_speech_start(waveform, sr)
    if cut > 0:
        logger.info("  Trimming %d leading samples (%.3fs of jitter).", cut, cut / sr)
        waveform = waveform[cut:]
    waveform = _fade_in(waveform, sr, FADE_IN_MS)
    return waveform


def _generate_one(model: TTS, text: str, samples: list[str]) -> np.ndarray:
    """Run TTS and return the waveform as a float32 array."""
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


# ---------------------------------------------------------------------------
# Voice sample collection
# ---------------------------------------------------------------------------

def _collect_voice_samples() -> list[str]:
    """Return voice sample WAV paths (folder first, single file fallback)."""
    samples: list[str] = []

    if os.path.isdir(VOICE_SAMPLES_DIR):
        samples = sorted(glob.glob(os.path.join(VOICE_SAMPLES_DIR, "*.wav")))

    if not samples and os.path.isfile(VOICE_SAMPLE):
        samples = [VOICE_SAMPLE]

    return samples


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------
tts_model: TTS | None = None
voice_samples: list[str] = []
ref_envelope: np.ndarray | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_model, voice_samples, ref_envelope
    os.makedirs(TMP_DIR, exist_ok=True)

    voice_samples = _collect_voice_samples()
    if voice_samples:
        logger.info("Using %d voice sample(s): %s", len(voice_samples), voice_samples)
        ref_envelope = _reference_envelope(voice_samples, OUTPUT_SR)
        logger.info("Reference voice envelope computed.")
    else:
        logger.warning("No voice samples found — /generate will fail.")

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
    version="2.1.0",
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
            "POST /generate": "Generate audio (best-of-N).",
            "GET  /health": "Health check.",
        },
    }


@app.get("/health")
def health_check():
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
    """Generate a WAV file in the cloned voice (best-of-N selection)."""

    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model still loading.")

    if not voice_samples:
        raise HTTPException(status_code=500, detail="No voice samples found.")

    logger.info(
        "Generating %d candidate(s) for %d chars | temp=%.2f  top_k=%d  top_p=%.2f",
        NUM_CANDIDATES, len(request.text), TEMPERATURE, TOP_K, TOP_P,
    )

    # --- generate N candidates, clean each one ----------------------------
    candidates: list[np.ndarray] = []
    for i in range(NUM_CANDIDATES):
        try:
            wav = _generate_one(tts_model, request.text, voice_samples)
            wav = _clean_start(wav, OUTPUT_SR)
            candidates.append(wav)
            logger.info("  Candidate %d/%d — %d samples (%.2fs), cleaned.",
                        i + 1, NUM_CANDIDATES, len(wav), len(wav) / OUTPUT_SR)
        except Exception as exc:
            logger.warning("  Candidate %d/%d failed: %s", i + 1, NUM_CANDIDATES, exc)

    if not candidates:
        raise HTTPException(status_code=500, detail="All generation attempts failed.")

    # --- pick the best candidate ------------------------------------------
    if len(candidates) == 1 or ref_envelope is None:
        best_wav = candidates[0]
    else:
        best_idx, best_score = 0, -1.0
        for i, cand in enumerate(candidates):
            env = _mel_envelope(cand, OUTPUT_SR)
            score = _cosine_sim(ref_envelope, env)
            logger.info("  Candidate %d similarity: %.4f", i + 1, score)
            if score > best_score:
                best_score, best_idx = score, i
        best_wav = candidates[best_idx]
        logger.info("Selected candidate %d (score %.4f).", best_idx + 1, best_score)

    # --- save & return ----------------------------------------------------
    filename = f"{uuid.uuid4().hex}.wav"
    output_path = os.path.join(TMP_DIR, filename)
    _save_wav(best_wav, output_path, OUTPUT_SR)

    logger.info("Audio saved → %s", output_path)
    return FileResponse(output_path, media_type="audio/wav", filename="output.wav")
