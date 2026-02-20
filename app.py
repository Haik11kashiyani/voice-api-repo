from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from TTS.api import TTS
import os

# Bypass the terms prompt
os.environ["COQUI_TOS_AGREED"] = "1"

app = FastAPI()

# Load the AI model into the server's memory
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")

class TextRequest(BaseModel):
    text: str

@app.post("/generate")
def generate_audio(request: TextRequest):
    # Hugging Face requires temporary files to be saved in /tmp
    output_path = "/tmp/output.wav"
    
    tts.tts_to_file(
        text=request.text,
        speaker_wav="voice_sample.wav", 
        language="en",
        file_path=output_path
    )
    
    return FileResponse(output_path, media_type="audio/wav")
