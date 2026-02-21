"""
test_api.py — Quick smoke-test for the Voice API.

Usage:
    python test_api.py                           # uses default Space URL
    python test_api.py https://your-url.hf.space # custom base URL
"""

import sys
import time
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "https://haikkashiyani-my-voice-api.hf.space"
BASE_URL = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL

HEALTH_URL = f"{BASE_URL}/health"
GENERATE_URL = f"{BASE_URL}/generate"

TEXT_TO_SAY = (
    "Hello! This is a test of your custom voice API. "
    "If you can hear this, the deployment was a success."
)

MAX_RETRIES = 5          # retries for cold-start wake-up
RETRY_DELAY_SECS = 15    # seconds between retries
REQUEST_TIMEOUT = 300     # seconds (sentence-by-sentence on CPU is slow)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def print_header(title: str):
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}\n")


def check_health() -> bool:
    """Ping /health (preferred) or fall back to root / to confirm the service is up."""
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[{attempt}/{MAX_RETRIES}] Pinging service …")
        try:
            # Try the dedicated /health endpoint first
            resp = requests.get(HEALTH_URL, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status", "unknown")
                print(f"  Status: {status}")
                print(f"  Model loaded : {data.get('model_loaded')}")
                print(f"  Voice sample : {data.get('voice_sample_found')}")
                if status == "ok":
                    return True
                print("  Service is degraded — retrying …")

            elif resp.status_code == 404:
                # Older deployment without /health — fall back to root
                print("  /health not found, trying root endpoint …")
                root_resp = requests.get(BASE_URL + "/", timeout=30)
                if root_resp.status_code == 200:
                    print("  Service is ONLINE (root responded OK).")
                    return True
                else:
                    print(f"  Root returned HTTP {root_resp.status_code}.")
            else:
                print(f"  Unexpected HTTP {resp.status_code}: {resp.text[:200]}")

        except requests.ConnectionError:
            print("  Could not connect (Space may still be booting).")
        except requests.Timeout:
            print("  Request timed out.")
        except Exception as exc:
            print(f"  Error: {exc}")

        if attempt < MAX_RETRIES:
            print(f"  Waiting {RETRY_DELAY_SECS}s before next attempt …\n")
            time.sleep(RETRY_DELAY_SECS)

    return False


def generate_audio() -> bool:
    """Send text to /generate and save the returned WAV."""
    print(f"Sending text ({len(TEXT_TO_SAY)} chars) to {GENERATE_URL} …")
    start = time.time()

    try:
        resp = requests.post(
            GENERATE_URL,
            json={"text": TEXT_TO_SAY},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.Timeout:
        print(f"Request timed out after {REQUEST_TIMEOUT}s.")
        return False
    except Exception as exc:
        print(f"Request failed: {exc}")
        return False

    elapsed = time.time() - start

    if resp.status_code == 200:
        out_file = "test_output.wav"
        with open(out_file, "wb") as f:
            f.write(resp.content)
        size_kb = len(resp.content) / 1024
        print(f"Success! Audio received in {elapsed:.1f}s ({size_kb:.1f} KB).")
        print(f"Saved to '{out_file}' — open it to listen.")
        return True
    else:
        print(f"Error HTTP {resp.status_code} (took {elapsed:.1f}s)")
        print(f"Response: {resp.text[:300]}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print_header("Voice API Smoke Test")
    print(f"Target: {BASE_URL}\n")

    # Step 1 — Health check
    print_header("Step 1: Health Check")
    if not check_health():
        print("\nHealth check failed after all retries.")
        print("Troubleshooting:")
        print("  1. Is the Space set to Public?  (Settings -> Visibility)")
        print("  2. Is the Space still Building? (Check the status badge)")
        print("  3. Did the app crash?           (Check the Logs tab)")
        sys.exit(1)

    # Step 2 — Generate audio
    print_header("Step 2: Generate Audio")
    if not generate_audio():
        print("\nAudio generation failed.")
        sys.exit(1)

    print_header("All tests passed!")


if __name__ == "__main__":
    main()
