import requests
import json
import time

# YOUR SPACE DETAILS
SPACE_URL = "https://haikkashiyani-my-voice-api.hf.space/generate"
TEXT_TO_SAY = "Hello! This is a test of your custom voice API. If you can hear this, the deployment was a success."

print(f"Testing API at: {SPACE_URL}")
print("Sending request... (this might take a minute if the server is waking up)")

try:
    # 1. Health Check (Try to access the docs)
    print("Checking if service is online...")
    try:
        health = requests.get(SPACE_URL.replace("/generate", "/docs"), timeout=10)
        if health.status_code == 200:
            print("Service is ONLINE! (Docs found)")
        elif health.status_code == 404:
             print("Service unreachable (404). Possible causes:")
             print("1. The Space is 'Private'. Go to Settings -> Make Public.")
             print("2. The Space is still 'Building'. Check the status on Hugging Face.")
             print("3. The App crashed. Check the 'Logs' tab on Hugging Face.")
             # We can't proceed if we can't look at the docs
    except Exception:
        print("Could not connect to service root.")

    print(f"\nSending generation request to {SPACE_URL}...")
    response = requests.post(
        SPACE_URL,
        json={"text": TEXT_TO_SAY},
        timeout=120
    )
    
    if response.status_code == 200:
        print("Success! Audio received.")
        with open("test_output.wav", "wb") as f:
            f.write(response.content)
        print("Saved to 'test_output.wav'. Check your folder and play it!")
    else:
        print(f"Error: {response.status_code}")
        print("Response:", response.text[:200])

except Exception as e:
    print(f"An error occurred: {e}")
    print("\nTroubleshooting Tip:")
    print("If this failed, go to your Hugging Face Space page and check if the 'Building' status has changed to 'Running'.")
