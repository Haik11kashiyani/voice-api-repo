import os
import glob
import time
import uuid
import wave
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
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
TEMPERATURE = float(os.getenv("TTS_TEMPERATURE", "0.35"))
TOP_K = int(os.getenv("TTS_TOP_K", "50"))
TOP_P = float(os.getenv("TTS_TOP_P", "0.85"))
REPETITION_PENALTY = float(os.getenv("TTS_REP_PENALTY", "2.5"))
SPEED = float(os.getenv("TTS_SPEED", "1.0"))

# XTTS v2 voice conditioning — how many seconds of reference audio to use
GPT_COND_LEN = int(os.getenv("TTS_GPT_COND_LEN", "12"))
GPT_COND_CHUNK_LEN = int(os.getenv("TTS_GPT_COND_CHUNK_LEN", "4"))

# Generate N candidates for the FULL text, pick the cleanest one
NUM_CANDIDATES = int(os.getenv("TTS_CANDIDATES", "5"))

# XTTS v2 always produces a warm-up artifact at the start (distorted first
# few phonemes). Hard-cut this many milliseconds BEFORE energy-based trimming.
START_CUT_MS = int(os.getenv("TTS_START_CUT_MS", "180"))

# Short texts (fewer chars than this) get trailing padding so XTTS v2 has
# enough context to produce a clean waveform.
SHORT_TEXT_THRESHOLD = int(os.getenv("TTS_SHORT_TEXT_THRESH", "15"))

# ── Primer word ──────────────────────────────────────────────────────────
# XTTS v2 garbles the first ~1s of every generation (warm-up artifact).
# We prepend a tiny throwaway word that absorbs the artifact, then
# hard-cut a fixed duration from the front.  No gap detection needed.
PRIMER_WORD = os.getenv("TTS_PRIMER", "So,")
# How many milliseconds to hard-cut from the start to remove the primer
# word + warm-up artifact.  Must be long enough to cover both but short
# enough not to eat into the real speech.  ~1.4s is safe for a 1-word
# primer at normal speaking speed.
PRIMER_CUT_MS = int(os.getenv("TTS_PRIMER_CUT_MS", "1400"))

# XTTS v2 native sample rate
OUTPUT_SR = 24000

# ── Audio polish parameters ────────────────────────────────────────────────
TARGET_LUFS = float(os.getenv("TTS_TARGET_LUFS", "-18.0"))  # loudness target
HIGH_SHELF_FREQ = float(os.getenv("TTS_HSHELF_HZ", "6000"))  # de-ess shelf
HIGH_SHELF_GAIN_DB = float(os.getenv("TTS_HSHELF_DB", "-2.5"))  # gentle cut
FADE_IN_MS = int(os.getenv("TTS_FADE_IN_MS", "40"))   # cosine fade-in
FADE_OUT_MS = int(os.getenv("TTS_FADE_OUT_MS", "60"))  # cosine fade-out

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


# ── Audio polish helpers ─────────────────────────────────────────────────

def _high_shelf(signal: np.ndarray, sr: int,
                freq: float, gain_db: float) -> np.ndarray:
    """Apply a single-pole high-shelf filter.

    Frequencies above *freq* are boosted/cut by *gain_db*.
    Uses a simple first-order IIR — cheap, zero-latency, no ringing.
    """
    if abs(gain_db) < 0.1:
        return signal  # nothing to do
    g = 10.0 ** (gain_db / 20.0)
    w0 = 2.0 * np.pi * freq / sr
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / 2.0 * np.sqrt(max((g + 1.0 / g) * (1.0 / 1.0 - 1.0) + 2.0, 0.001))
    sq = 2.0 * np.sqrt(g) * alpha

    b0 = g * ((g + 1) + (g - 1) * cos_w0 + sq)
    b1 = -2 * g * ((g - 1) + (g + 1) * cos_w0)
    b2 = g * ((g + 1) + (g - 1) * cos_w0 - sq)
    a0 = (g + 1) - (g - 1) * cos_w0 + sq
    a1 = 2 * ((g - 1) - (g + 1) * cos_w0)
    a2 = (g + 1) - (g - 1) * cos_w0 - sq

    # Normalize
    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])

    # Direct-form II transposed (forward pass only — no look-ahead)
    out = np.zeros_like(signal)
    z1, z2 = 0.0, 0.0
    for i in range(len(signal)):
        x = float(signal[i])
        y = b[0] * x + z1
        z1 = b[1] * x - a[1] * y + z2
        z2 = b[2] * x - a[2] * y
        out[i] = y
    return out


def _normalize_loudness(signal: np.ndarray,
                        target_lufs: float = -18.0) -> np.ndarray:
    """RMS-based loudness normalization (mono, simplified LUFS).

    Measures the RMS of the signal, converts to approximate LUFS,
    and scales so the output sits at *target_lufs*.
    """
    rms = float(np.sqrt(np.mean(signal ** 2)))
    if rms < 1e-10:
        return signal
    current_lufs = 20.0 * np.log10(rms) - 0.691  # approximate LUFS for mono
    diff = target_lufs - current_lufs
    gain = 10.0 ** (diff / 20.0)
    return signal * gain


def _save_wav(wav: np.ndarray, path: str, sr: int = OUTPUT_SR) -> None:
    """Apply final polish chain and write 16-bit WAV.

    1. Remove DC offset
    2. Gentle high-shelf cut (tames sibilance / XTTS hiss)
    3. RMS-based loudness normalization to TARGET_LUFS
    4. Peak-limit to −0.5 dBFS (prevents clipping)
    """
    w = np.array(wav, dtype=np.float32)

    # 1 — DC offset removal
    w = w - np.mean(w)

    # 2 — gentle high-shelf EQ (single-pole IIR)
    w = _high_shelf(w, sr, HIGH_SHELF_FREQ, HIGH_SHELF_GAIN_DB)

    # 3 — RMS loudness normalization
    w = _normalize_loudness(w, TARGET_LUFS)

    # 4 — peak-limit to −0.5 dBFS (≈ 0.944)
    peak = np.max(np.abs(w))
    if peak > 0.944:
        w = w * (0.944 / peak)

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
    # Z-score of frame-to-frame jumps — large spikes indicate glitches
    d = np.abs(np.diff(rms))
    if d.std() < 1e-10:
        return 0.0
    z = (d - d.mean()) / (d.std() + 1e-10)
    # Count frames with z > 3 (severe spike)
    n_bad = int(np.sum(z > 3.0))
    return float(np.clip(n_bad / max(nf, 1) * 20.0, 0.0, 1.0))


def _pacing_score(wav: np.ndarray, sr: int, frame_ms: float = 25.0) -> float:
    """Score 0-1: how natural the speech pacing is.

    Measures the ratio of voiced frames to total frames.  Natural speech
    at normal speed has ~55-75 % voiced content.  Too high means the
    model is rushing; too low means excessive pauses / silence.
    """
    fl = max(1, int(sr * frame_ms / 1000))
    nf = len(wav) // fl
    if nf < 4:
        return 0.5
    rms = np.array([float(np.sqrt(np.mean(wav[i*fl:(i+1)*fl] ** 2)))
                     for i in range(nf)])
    voiced = float(np.sum(rms > 0.008)) / nf
    # Ideal range 0.55 – 0.75; score drops outside
    if 0.55 <= voiced <= 0.75:
        return 1.0
    if voiced < 0.55:
        return float(np.clip(voiced / 0.55, 0.0, 1.0))
    return float(np.clip((1.0 - voiced) / 0.25, 0.0, 1.0))


def _score(wav: np.ndarray, ref_env: np.ndarray | None, sr: int) -> float:
    """Combined quality score (higher = better).

    35 % voice similarity  +  25 % smoothness  +  25 % (1 − glitch)  +  15 % pacing
    """
    sm = _smoothness(wav, sr)
    gp = _glitch_penalty(wav, sr)
    pc = _pacing_score(wav, sr)
    if ref_env is None:
        return 0.40 * sm + 0.40 * (1.0 - gp) + 0.20 * pc
    env = _mel_envelope(wav, sr)
    sim = _cosine_sim(ref_env, env)
    return 0.35 * sim + 0.25 * sm + 0.25 * (1.0 - gp) + 0.15 * pc


# ── Minimal cleanup: trim silence + fade edges ───────────────────────────

def _find_speech_start(wav: np.ndarray, sr: int,
                       frame_ms: int = 8, thresh: float = 0.006,
                       need: int = 4, back_ms: int = 6) -> int:
    """Scan forward to find where speech energy begins.

    Slightly aggressive: 4 consecutive frames above threshold,
    ~6 ms pre-speech padding.
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
                     need: int = 4, tail_ms: int = 15) -> int:
    """Scan backward to find where speech energy ends.

    Allows a small 15 ms tail to keep natural decay.
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


def _find_first_silence_gap(wav: np.ndarray, sr: int,
                            min_speech_ms: int = 120,
                            frame_ms: int = 10,
                            silence_thresh: float = 0.008,
                            gap_ms: int = 150,
                            tail_ms: int = 30) -> int | None:
    """Scan forward and find the END of the first real speech region.

    After locating at least *min_speech_ms* of speech, look for the first
    silence gap of at least *gap_ms*.  Return the sample index where that
    gap starts (+ a small tail), or None if no clear gap is found.

    This is used for padded short texts: the gap marks the boundary
    between the real word and the trailing filler — everything after it
    should be discarded.
    """
    fl = max(1, int(sr * frame_ms / 1000))
    min_speech_frames = max(1, int(min_speech_ms / frame_ms))
    gap_frames_needed = max(1, int(gap_ms / frame_ms))

    speech_frames = 0
    silence_run = 0
    found_speech = False

    for i in range(0, len(wav) - fl, fl):
        rms = float(np.sqrt(np.mean(wav[i:i+fl] ** 2)))
        if rms >= silence_thresh:
            speech_frames += 1
            silence_run = 0
            if speech_frames >= min_speech_frames:
                found_speech = True
        else:
            if found_speech:
                silence_run += 1
                if silence_run >= gap_frames_needed:
                    # Gap starts where the silence run began
                    gap_start = i - (silence_run - 1) * fl
                    return min(len(wav), gap_start + int(sr * tail_ms / 1000))
            else:
                silence_run = 0  # ignore silence before speech
    return None


def _trim_and_fade(wav: np.ndarray, sr: int,
                   has_primer: bool = True,
                   was_padded: bool = False) -> np.ndarray:
    """Clean up raw TTS output:

    0. If a primer was prepended, hard-cut PRIMER_CUT_MS from the front
       to remove the throwaway word + warm-up artifact.
    1. Energy-based silence trim (start & end)
    1b. For padded short text: forward-scan for the first silence
        gap after the real speech and hard-cut there to remove filler.
    2. Gentle fade-in / fade-out
    """
    # Step 0 — discard primer word + warm-up artifact via fixed hard-cut
    if has_primer:
        primer_cut = int(sr * PRIMER_CUT_MS / 1000)
        if primer_cut < len(wav):
            logger.info("  Primer hard-cut: removing first %.3fs",
                        primer_cut / sr)
            wav = wav[primer_cut:]
        # Also apply the legacy START_CUT_MS on top in case there's
        # residual artifact right after the primer cut
        hard_cut = int(sr * START_CUT_MS / 1000)
        if hard_cut < len(wav):
            wav = wav[hard_cut:]
    else:
        hard_cut = int(sr * START_CUT_MS / 1000)
        if hard_cut < len(wav):
            wav = wav[hard_cut:]

    # Step 1 — energy-based silence trim
    start = _find_speech_start(wav, sr)
    end = _find_speech_end(wav, sr)
    if start >= end:
        trimmed = wav
    else:
        trimmed = wav[start:end]

    # Step 1b — for padded text, cut at the first silence gap after
    #           the real speech to remove any filler the model produced
    if was_padded:
        gap_cut = _find_first_silence_gap(trimmed, sr)
        if gap_cut is not None and gap_cut > int(sr * 0.10):  # keep ≥100ms
            logger.info("  Padded-text trim: cutting at %.3fs (of %.3fs)",
                        gap_cut / sr, len(trimmed) / sr)
            trimmed = trimmed[:gap_cut]

    # Step 2 — cosine fade edges (smoother than linear)
    out = trimmed.copy()
    n_in = min(int(sr * FADE_IN_MS / 1000), len(out))
    n_out = min(int(sr * FADE_OUT_MS / 1000), len(out))
    if n_in > 0:
        # cosine curve: 0 → 1 with smooth acceleration
        out[:n_in] *= (1.0 - np.cos(np.linspace(0, np.pi, n_in))) / 2.0
    if n_out > 0:
        out[-n_out:] *= (1.0 + np.cos(np.linspace(0, np.pi, n_out))) / 2.0
    return out


# ═══════════════════════════════════════════════════════════════════════════
# TTS generation — full-text best-of-N
# ═══════════════════════════════════════════════════════════════════════════

def _pad_short_text(text: str) -> str:
    """Pad very short text so XTTS v2 can articulate it cleanly.

    XTTS v2 needs enough trailing phonetic context to produce stable
    output.  If the input is too short (e.g. "Hello") the model rushes
    and garbles the word.  We append a trailing filler sentence that
    gives the model enough context to finish naturally.  The silence
    trimmer then strips the filler from the final audio, so the
    listener only hears the intended word(s).

    IMPORTANT: padding is ONLY added after the text, never before,
    because leading filler causes a long garbled warm-up artifact
    that START_CUT_MS cannot fully remove.
    """
    stripped = text.strip()
    if len(stripped) < SHORT_TEXT_THRESHOLD:
        # Ensure proper sentence punctuation so the model treats it as
        # a complete utterance.
        if stripped and stripped[-1] not in ".!?":
            stripped += "."
        # Trailing filler — long enough to give XTTS context, bland
        # enough that the silence trimmer can detect the gap after the
        # real speech ends.
        padded = f"{stripped}  ...  ...  ..."
        logger.info("Short text detected (%d chars) — padded to: %r",
                     len(text.strip()), padded)
        return padded
    return text


def _generate_one(model: TTS, text: str, samples: list[str]) -> tuple[np.ndarray, bool]:
    """Single TTS call → (float32 waveform, was_padded) tuple.

    Prepends a primer sentence so the XTTS warm-up artifact lands on
    the throwaway primer instead of the real text.  Also pads short
    texts with trailing filler.
    """
    # Pad very short text so XTTS v2 has enough context to articulate
    # cleanly instead of producing garbled filler sounds.
    synth_text = _pad_short_text(text)
    was_padded = (synth_text != text)

    # Prepend a short throwaway word that absorbs the warm-up artifact.
    # A fixed-duration hard-cut in _trim_and_fade removes it reliably.
    full_text = f"{PRIMER_WORD} {synth_text}"
    logger.info("  Synth text: %r", full_text)

    wav = model.tts(
        text=full_text,
        speaker_wav=samples,
        language="en",
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        speed=SPEED,
        gpt_cond_len=GPT_COND_LEN,
        gpt_cond_chunk_len=GPT_COND_CHUNK_LEN,
        enable_text_splitting=True,
    )
    return np.array(wav, dtype=np.float32), was_padded


def _best_of_n(model: TTS, text: str, samples: list[str],
               ref_env: np.ndarray | None, n: int) -> np.ndarray:
    """Generate *n* full candidates, trim+fade each, return the highest-scored."""
    candidates: list[tuple[float, np.ndarray]] = []
    for i in range(n):
        try:
            raw, was_padded = _generate_one(model, text, samples)
            cleaned = _trim_and_fade(raw, OUTPUT_SR,
                                     has_primer=True,
                                     was_padded=was_padded)
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


# ── Temp file cleanup ────────────────────────────────────────────────────

def _cleanup_tmp(path: str, delay: float = 5.0) -> None:
    """Delete a temp WAV file after a short delay (gives time for streaming)."""
    try:
        time.sleep(delay)
        if os.path.exists(path):
            os.remove(path)
            logger.debug("Cleaned up %s", path)
    except OSError:
        pass  # non-critical — /tmp is ephemeral anyway


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI app
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="My Voice API",
    description="Generate speech in your cloned voice using XTTS v2.",
    version="5.0.0",
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
        "primer": PRIMER_WORD,
        "primer_cut_ms": PRIMER_CUT_MS,
        "model": MODEL_NAME,
    }


@app.post("/generate")
def generate_audio(request: TextRequest, bg: BackgroundTasks):
    """Generate speech from the full text.

    Produces NUM_CANDIDATES independent takes of the complete text,
    picks the one with the best quality score, and returns it as WAV.

    Post-processing: primer removal → silence trim → edge fades.
    The waveform is never otherwise modified.
    """
    if tts_model is None:
        raise HTTPException(503, detail="Model still loading.")
    if not voice_samples:
        raise HTTPException(500, detail="No voice samples configured.")

    text = request.text.strip()
    logger.info("Generate request: %d chars, %d candidates",
                len(text), NUM_CANDIDATES)

    t0 = time.monotonic()
    best = _best_of_n(tts_model, text, voice_samples,
                      ref_envelope, NUM_CANDIDATES)
    elapsed = time.monotonic() - t0

    out_path = os.path.join(TMP_DIR, f"{uuid.uuid4().hex}.wav")
    _save_wav(best, out_path, OUTPUT_SR)
    duration = len(best) / OUTPUT_SR
    logger.info("Saved %s (%.2fs audio, generated in %.1fs)",
                out_path, duration, elapsed)

    # Clean up the temp file after the response has been sent
    bg.add_task(_cleanup_tmp, out_path)

    return FileResponse(out_path, media_type="audio/wav", filename="output.wav")
