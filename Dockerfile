FROM ghcr.io/coqui-ai/tts-cpu

# Keep essential system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# The base image installs TTS in editable mode into /root/TTS.
# Hugging Face runs containers as non-root (uid 1000), which causes Permission Denied.
# We make /root and /root/TTS accessible to all users to fix this.
RUN chmod a+rx /root && \
    chmod -R a+rX /root/TTS || true

# Create a specific user to avoid Hugging Face permission errors
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Open the default Hugging Face API port
EXPOSE 7860

# Override the base image's default entrypoint ("tts") 
# so that it runs our FastAPI server instead.
ENTRYPOINT []
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
