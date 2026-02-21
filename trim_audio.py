"""
trim_audio.py — Trim a WAV file to the first N seconds.

Usage:
    python trim_audio.py                                           # defaults
    python trim_audio.py voice_sample.wav trimmed.wav 8.0          # custom
"""

import os
import sys
import wave

DEFAULT_INPUT = "voice_sample.wav"
DEFAULT_OUTPUT = "voice_sample_trimmed.wav"
DEFAULT_DURATION = 10.0


def trim_wav(input_path: str, output_path: str, duration_sec: float = 10.0) -> bool:
    """Trim *input_path* to the first *duration_sec* seconds and write to *output_path*.

    Returns True on success, False on failure.
    """
    if not os.path.isfile(input_path):
        print(f"Error: input file '{input_path}' not found.")
        return False

    try:
        with wave.open(input_path, "rb") as infile:
            nchannels = infile.getnchannels()
            sampwidth = infile.getsampwidth()
            framerate = infile.getframerate()
            nframes = infile.getnframes()

            original_duration = nframes / framerate
            frames_to_keep = min(int(framerate * duration_sec), nframes)
            data = infile.readframes(frames_to_keep)

        with wave.open(output_path, "wb") as outfile:
            outfile.setnchannels(nchannels)
            outfile.setsampwidth(sampwidth)
            outfile.setframerate(framerate)
            outfile.writeframes(data)

        actual_sec = frames_to_keep / framerate
        print(f"Original : {original_duration:.2f}s")
        print(f"Trimmed  : {actual_sec:.2f}s")
        print(f"Saved to : {output_path}")
        return True

    except wave.Error as exc:
        print(f"WAV error: {exc}")
    except Exception as exc:
        print(f"Unexpected error: {exc}")
    return False


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_DURATION

    success = trim_wav(in_path, out_path, dur)
    sys.exit(0 if success else 1)
