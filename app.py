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

# ── XTTS v2 generation parameters ────────────────────────────────────────
TEMPERATURE = float(os.getenv("TTS_TEMPERATURE", "0.22"))
TOP_K = int(os.getenv("TTS_TOP_K", "50"))
TOP_P = float(os.getenv("TTS_TOP_P", "0.92"))
REPETITION_PENALTY = float(os.getenv("TTS_REP_PENALTY", "2.4"))
SPEED = float(os.getenv("TTS_SPEED", "0.99"))

# Generate N candidates for the FULL text, pick the cleanest one
NUM_CANDIDATES = int(os.getenv("TTS_CANDIDATES", "6"))

# XTTS v2 native sample rate
OUTPUT_SR = 24000

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("voice-api")


# ═══════════════════════════════════════════════════════════════════════════
# Audio helpers  (read-only measurement + minimal trim/fade — nothing else)
# ═══════════════════════════════════════════════════════════════════════════

def _load_wav_as_float(wav_path: str) -> np.ndarray:
    """Load any WAV file → mono float32 in [-1, 1]."""
    with wave.open(wav_path, "rb") as wf:
        n_ch = wf.getnchannels()
        sw = wf.getsampwidth()
        nf = wf.getnframes()
        raw = wf.readframes(nf)
    if sw == 2:
        s = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        s = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        s = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if n_ch > 1:
        s = s.reshape(-1, n_ch).mean(axis=1)
    return s


def _mel_envelope(signal: np.ndarray, sr: int,
                  n_fft: int = 2048, n_bands: int = 40) -> np.ndarray:
    """Compute a rough mel-scale spectral envelope for voice similarity."""
    hop = n_fft // 2
    n_hops = max(1, (len(signal) - n_fft) // hop)
    power_sum = np.zeros(n_fft // 2 + 1)
    win = np.hanning(n_fft)
    for i in range(n_hops):
        f = signal[i * hop: i * hop + n_fft]
        if len(f) < n_fft:
            f = np.pad(f, (0, n_fft - len(f)))
        power_sum += np.abs(np.fft.rfft(f * win)) ** 2
    avg = power_sum / max(n_hops, 1)
    freqs = np.linspace(0, sr / 2, len(avg))
    mf = 2595.0 * np.log10(1.0 + freqs / 700.0)
    edges = np.linspace(mf[0], mf[-1], n_bands + 2)
    env = np.zeros(n_bands)
    for b in range(n_bands):
        mask = (mf >= edges[b]) & (mf < edges[b + 2])
        if mask.any():
            env[b] = np.mean(avg[mask])
    return np.log(env + 1e-10)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _reference_envelope(paths: list[str], sr: int) -> np.ndarray:
    """Average mel envelope across all reference voice samples."""
    return np.mean([_mel_envelope(_load_wav_as_float(p), sr) for p in paths],
                   axis=0)


def _save_wav(wav: np.ndarray, path: str, sr: int = OUTPUT_SR) -> None:
    """Normalize to −0.95 … +0.95 and write 16-bit WAV."""
    w = np.array(wav, dtype=np.float32)
    peak = np.max(np.abs(w))
    if peak > 0:
        w = w / peak * 0.95
    w16 = (w * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(w16.tobytes())


# ── Quality scoring (never modifies the waveform) ────────────────────────

def _smoothness(wav: np.ndarray, sr: int, frame_ms: float = 10.0) -> float:
    """Score 0-1: how smooth the energy envelope is (1 = smooth, 0 = glitchy)."""
    fl = max(1, int(sr * frame_ms / 1000))
    nf = len(wav) // fl
    if nf < 3:
        return 1.0
    rms = np.array([float(np.sqrt(np.mean(wav[i*fl:(i+1)*fl] ** 2)))
                     for i in range(nf)])
    d = np.abs(np.diff(rms))
    m = float(np.mean(rms)) + 1e-10
    return float(np.clip(1.0 / (1.0 + np.mean(d / m) * 5.0), 0.0, 1.0))


def _glitch_penalty(wav: np.ndarray, sr: int, frame_ms: float = 10.0) -> float:
    """Penalize sudden energy spikes (clicks / pops).  0 = no glitches, 1 = terrible."""
    fl = max(1, int(sr * frame_ms / 1000))
    nf = len(wav) // fl
    if nf < 5:
        return 0.0
    rms = np.array([float(np.sqrt(np.mean(wav[i*fl:(i+1)*fl] ** 2)))
                     for i in range(nf)])
    d = np.abs(np.diff(rms))
    if d.std() < 1e-10:
        return 0.0
    z = (d - d.mean()) / (d.std() + 1e-10)
    n_bad = int(np.sum(z > 4.0))  # only penalize very strong spikes
    return float(np.clip(n_bad / max(nf, 1) * 15.0, 0.0, 1.0))


def _score(wav: np.ndarray, ref_env: np.ndarray | None, sr: int) -> float:
    """Combined quality score (higher = better).

    35 % voice similarity  +  45 % smoothness  +  20 % (1 − glitch penalty)
    """
    sm = _smoothness(wav, sr)
    gp = _glitch_penalty(wav, sr)
    if ref_env is None:
        return 0.6 * sm + 0.4 * (1.0 - gp)
    env = _mel_envelope(wav, sr)
    sim = _cosine_sim(ref_env, env)
    return 0.35 * sim + 0.45 * sm + 0.20 * (1.0 - gp)


# ── Minimal cleanup: trim silence + fade edges ───────────────────────────

def _find_speech_start(wav: np.ndarray, sr: int,
                       frame_ms: int = 8, thresh: float = 0.006,
                       need: int = 4, back_ms: int = 12) -> int:
    """Scan forward to find where speech energy begins.

    Keeps ~12 ms pre-speech padding to avoid cutting leading consonants.
    """
    fl = max(1, int(sr * frame_ms / 1000))
    consec = 0
    for s in range(0, len(wav) - fl, fl):
        rms = float(np.sqrt(np.mean(wav[s:s+fl] ** 2)))
        if rms >= thresh:
            consec += 1
            if consec >= need:
                onset = s - (need - 1) * fl
                return max(0, onset - int(sr * back_ms / 1000))
        else:
            consec = 0
    return 0


def _find_speech_end(wav: np.ndarray, sr: int,
                     frame_ms: int = 8, thresh: float = 0.006,
                     need: int = 4, tail_ms: int = 22) -> int:
    """Scan backward to find where speech energy ends.

    Keeps a short 22 ms tail so vowels decay naturally.
    """
    fl = max(1, int(sr * frame_ms / 1000))
    consec = 0
    for s in range(len(wav) - fl, 0, -fl):
        rms = float(np.sqrt(np.mean(wav[s:s+fl] ** 2)))
        if rms >= thresh:
            consec += 1
            if consec >= need:
                last = s + fl + (need - 1) * fl
                return min(len(wav), last + int(sr * tail_ms / 1000))
        else:
            consec = 0
    return len(wav)


def _trim_and_fade(wav: np.ndarray, sr: int) -> np.ndarray:
    """Trim leading/trailing silence  →  apply gentle fade-in / fade-out.

    This is the ONLY modification applied to the raw TTS output.
    """
    start = _find_speech_start(wav, sr)
    end = _find_speech_end(wav, sr)
    if start >= end:
        trimmed = wav
    else:
        trimmed = wav[start:end]

    out = trimmed.copy()
    # 80 ms fade-in / 60 ms fade-out to smooth the edges cleanly
    n_in = min(int(sr * 0.080), len(out))
    n_out = min(int(sr * 0.060), len(out))
    if n_in > 0:
        out[:n_in] *= np.linspace(0, 1, n_in, dtype=np.float32)
    if n_out > 0:
        out[-n_out:] *= np.linspace(1, 0, n_out, dtype=np.float32)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# TTS generation — full-text best-of-N
# ═══════════════════════════════════════════════════════════════════════════

def _generate_one(model: TTS, text: str, samples: list[str]) -> np.ndarray:
    """Single TTS call → float32 waveform (no post-processing yet)."""
    wav = model.tts(
        text=text,
        speaker_wav=samples,
        language="en",
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        speed=SPEED,
    )
    return np.array(wav, dtype=np.float32)


def _best_of_n(model: TTS, text: str, samples: list[str],
               ref_env: np.ndarray | None, n: int) -> np.ndarray:
    """Generate *n* full candidates, trim+fade each, return the highest-scored."""
    candidates: list[tuple[float, np.ndarray]] = []
    for i in range(n):
        try:
            raw = _generate_one(model, text, samples)
            cleaned = _trim_and_fade(raw, OUTPUT_SR)
            sc = _score(cleaned, ref_env, OUTPUT_SR)
            dur = len(cleaned) / OUTPUT_SR
            candidates.append((sc, cleaned))
            logger.info("  Candidate %d/%d — %.2fs, score %.4f", i+1, n, dur, sc)
        except Exception as e:
            logger.warning("  Candidate %d/%d failed: %s", i+1, n, e)

    if not candidates:
        raise RuntimeError(f"All {n} candidates failed.")

    # Pick the best
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score = candidates[0][0]
    best_wav = candidates[0][1]
    logger.info("  Winner: score %.4f, %.2fs",
                best_score, len(best_wav) / OUTPUT_SR)
    return best_wav


# ═══════════════════════════════════════════════════════════════════════════
# Voice sample collection
# ═══════════════════════════════════════════════════════════════════════════

def _collect_voice_samples() -> list[str]:
    samples: list[str] = []
    if os.path.isdir(VOICE_SAMPLES_DIR):
        samples = sorted(glob.glob(os.path.join(VOICE_SAMPLES_DIR, "*.wav")))
    if not samples and os.path.isfile(VOICE_SAMPLE):
        samples = [VOICE_SAMPLE]
    return samples


# ═══════════════════════════════════════════════════════════════════════════
# App lifecycle
# ═══════════════════════════════════════════════════════════════════════════

tts_model: TTS | None = None
voice_samples: list[str] = []
ref_envelope: np.ndarray | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_model, voice_samples, ref_envelope
    os.makedirs(TMP_DIR, exist_ok=True)

    voice_samples = _collect_voice_samples()
    if voice_samples:
        logger.info("Voice samples: %s", voice_samples)
        ref_envelope = _reference_envelope(voice_samples, OUTPUT_SR)
    else:
        logger.warning("No voice samples found.")

    logger.info("Loading model '%s' …", MODEL_NAME)
    tts_model = TTS(MODEL_NAME).to("cpu")
    logger.info("Model ready.")
    yield
    logger.info("Shutting down.")


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI app
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="My Voice API",
    description="Generate speech in your cloned voice.",
    version="4.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH,
                      description="Text to speak.",
                      json_schema_extra={"example": "Hello, this is a test."})


@app.get("/")
def root():
    return {
        "message": "Voice API is running.",
        "endpoints": {
            "POST /generate": "Generate audio.",
            "GET  /health": "Health check.",
        },
    }


@app.get("/health")
def health():
    ok = tts_model is not None and len(voice_samples) > 0
    return {
        "status": "ok" if ok else "degraded",
        "model_loaded": tts_model is not None,
        "voice_samples": len(voice_samples),
        "candidates": NUM_CANDIDATES,
    }


@app.post("/generate")
def generate_audio(request: TextRequest):
    """Generate speech from the full text.

    Produces NUM_CANDIDATES independent takes of the complete text,
    picks the one with the best quality score, and returns it as WAV.

    Post-processing is limited to silence trimming + edge fades — the
    waveform is NEVER otherwise modified.
    """
    if tts_model is None:
        raise HTTPException(503, "Model still loading.")
    if not voice_samples:
        raise HTTPException(500, "No voice samples configured.")

    text = request.text.strip()
    logger.info("Generate request: %d chars, %d candidates",
                len(text), NUM_CANDIDATES)

    best = _best_of_n(tts_model, text, voice_samples,
                      ref_envelope, NUM_CANDIDATES)

    out_path = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}.wav")
    _save_wav(best, out_path, OUTPUT_SR)
    logger.info("Saved %s (%.2fs)", out_path, len(best) / OUTPUT_SR)

    return FileResponse(out_path, media_type="audio/wav", filename="output.wav")
