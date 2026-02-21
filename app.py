import os
import re
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

# XTTS v2 generation — use the model's recommended defaults
TEMPERATURE = float(os.getenv("TTS_TEMPERATURE", "0.65"))
TOP_K = int(os.getenv("TTS_TOP_K", "50"))
TOP_P = float(os.getenv("TTS_TOP_P", "0.85"))
REPETITION_PENALTY = float(os.getenv("TTS_REP_PENALTY", "2.0"))
SPEED = float(os.getenv("TTS_SPEED", "1.0"))

# Per-sentence: generate N takes, pick the smoothest
TAKES_PER_SENTENCE = int(os.getenv("TTS_TAKES", "2"))

# Crossfade duration between sentences (milliseconds)
CROSSFADE_MS = int(os.getenv("CROSSFADE_MS", "80"))

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


# ═══════════════════════════════════════════════════════════════════════════
# Audio utilities (no waveform manipulation — only measurement + assembly)
# ═══════════════════════════════════════════════════════════════════════════

def _load_wav_as_float(wav_path: str) -> np.ndarray:
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
    return np.mean([_mel_envelope(_load_wav_as_float(p), sr) for p in paths], axis=0)


def _save_wav(wav: np.ndarray, path: str, sr: int = OUTPUT_SR) -> None:
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


# ── Quality measurement (read-only — never modifies audio) ───────────────

def _smoothness(wav: np.ndarray, sr: int, frame_ms: float = 10.0) -> float:
    """1.0 = perfectly smooth energy curve, 0.0 = very glitchy."""
    fl = max(1, int(sr * frame_ms / 1000))
    nf = len(wav) // fl
    if nf < 3:
        return 1.0
    rms = np.array([float(np.sqrt(np.mean(wav[i*fl:(i+1)*fl] ** 2))) for i in range(nf)])
    d = np.abs(np.diff(rms))
    m = float(np.mean(rms)) + 1e-10
    return float(np.clip(1.0 / (1.0 + np.mean(d / m) * 5.0), 0.0, 1.0))


def _score(wav: np.ndarray, ref_env: np.ndarray | None, sr: int) -> float:
    """Combined quality score: 60% voice match + 40% smoothness."""
    sm = _smoothness(wav, sr)
    if ref_env is None:
        return sm
    env = _mel_envelope(wav, sr)
    sim = _cosine_sim(ref_env, env)
    return 0.6 * sim + 0.4 * sm


# ── Minimal cleanup: trim leading silence, fade edges ─────────────────────

def _find_speech_start(wav: np.ndarray, sr: int,
                       frame_ms: int = 15, thresh: float = 0.004,
                       need: int = 3, back_ms: int = 20) -> int:
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
                     frame_ms: int = 15, thresh: float = 0.004,
                     need: int = 3, tail_ms: int = 40) -> int:
    """Find where speech ends (scan from the back)."""
    fl = max(1, int(sr * frame_ms / 1000))
    consec = 0
    last = len(wav)
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


def _trim_silence(wav: np.ndarray, sr: int) -> np.ndarray:
    """Trim leading and trailing silence only — no waveform modification."""
    start = _find_speech_start(wav, sr)
    end = _find_speech_end(wav, sr)
    if start >= end:
        return wav
    return wav[start:end]


def _fade(wav: np.ndarray, sr: int, in_ms: float = 30, out_ms: float = 30) -> np.ndarray:
    """Apply gentle fade-in and fade-out to avoid hard edges."""
    out = wav.copy()
    n_in = min(int(sr * in_ms / 1000), len(out))
    n_out = min(int(sr * out_ms / 1000), len(out))
    if n_in > 0:
        out[:n_in] *= np.linspace(0, 1, n_in, dtype=np.float32)
    if n_out > 0:
        out[-n_out:] *= np.linspace(1, 0, n_out, dtype=np.float32)
    return out


# ── Sentence splitting ────────────────────────────────────────────────────

_SENT_RE = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Keep very short fragments merged."""
    raw = _SENT_RE.split(text.strip())
    sentences: list[str] = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        # Merge very short fragments (< 20 chars) with the previous sentence
        if sentences and len(s) < 20:
            sentences[-1] = sentences[-1] + " " + s
        else:
            sentences.append(s)
    return sentences if sentences else [text.strip()]


# ── Crossfade join ────────────────────────────────────────────────────────

def _crossfade_join(chunks: list[np.ndarray], sr: int, ms: int = CROSSFADE_MS) -> np.ndarray:
    """Join audio chunks with smooth crossfades between them."""
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    xf = min(int(sr * ms / 1000), min(len(c) for c in chunks) // 2)
    if xf < 2:
        return np.concatenate(chunks)

    result = chunks[0].copy()
    for nxt in chunks[1:]:
        # fade out the tail of result
        fade_out = np.linspace(1, 0, xf, dtype=np.float32)
        fade_in = np.linspace(0, 1, xf, dtype=np.float32)
        tail = result[-xf:] * fade_out + nxt[:xf] * fade_in
        result = np.concatenate([result[:-xf], tail, nxt[xf:]])

    return result


# ── TTS generation ────────────────────────────────────────────────────────

def _tts_raw(model: TTS, text: str, samples: list[str]) -> np.ndarray:
    """Single TTS call → float32 waveform."""
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


def _best_take(model: TTS, text: str, samples: list[str],
               ref_env: np.ndarray | None, n_takes: int) -> np.ndarray:
    """Generate *n_takes* for the same text, return the best one.

    Each take is trimmed of silence and scored.  No other modification.
    """
    takes: list[tuple[float, np.ndarray]] = []
    for t in range(n_takes):
        try:
            raw = _tts_raw(model, text, samples)
            trimmed = _trim_silence(raw, OUTPUT_SR)
            sc = _score(trimmed, ref_env, OUTPUT_SR)
            takes.append((sc, trimmed))
            logger.info("    Take %d/%d  %.2fs  score=%.4f",
                        t + 1, n_takes, len(trimmed) / OUTPUT_SR, sc)
        except Exception as e:
            logger.warning("    Take %d/%d failed: %s", t + 1, n_takes, e)

    if not takes:
        raise RuntimeError(f"All {n_takes} takes failed for: {text[:60]}")

    takes.sort(key=lambda x: x[0], reverse=True)
    return takes[0][1]


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
# FastAPI
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="My Voice API",
    description="Generate speech in your cloned voice.",
    version="3.0.0",
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
        "takes_per_sentence": TAKES_PER_SENTENCE,
    }


@app.post("/generate")
def generate_audio(request: TextRequest):
    """Generate clean speech: sentence-by-sentence best-of-N, crossfade joined."""

    if tts_model is None:
        raise HTTPException(503, "Model still loading.")
    if not voice_samples:
        raise HTTPException(500, "No voice samples.")

    # 1. Split into sentences
    sentences = _split_sentences(request.text)
    logger.info("Text split into %d sentence(s): %s",
                len(sentences), [s[:40] + "…" if len(s) > 40 else s for s in sentences])

    # 2. Generate best take per sentence
    sentence_wavs: list[np.ndarray] = []
    for idx, sent in enumerate(sentences):
        logger.info("  Sentence %d/%d: \"%s\"", idx + 1, len(sentences),
                    sent[:50] + "…" if len(sent) > 50 else sent)
        try:
            best = _best_take(tts_model, sent, voice_samples,
                              ref_envelope, TAKES_PER_SENTENCE)
            # Only apply fade to edges — do NOT modify the waveform otherwise
            best = _fade(best, OUTPUT_SR, in_ms=20, out_ms=20)
            sentence_wavs.append(best)
        except RuntimeError as e:
            logger.error("  Sentence %d failed entirely: %s", idx + 1, e)
            raise HTTPException(500, f"Generation failed for sentence {idx + 1}.")

    # 3. Crossfade-join all sentences
    final = _crossfade_join(sentence_wavs, OUTPUT_SR, CROSSFADE_MS)
    logger.info("Final audio: %.2fs", len(final) / OUTPUT_SR)

    # 4. Save and return
    out_path = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}.wav")
    _save_wav(final, out_path, OUTPUT_SR)

    return FileResponse(out_path, media_type="audio/wav", filename="output.wav")
