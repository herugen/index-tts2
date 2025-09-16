FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*

# CUDA 12.8 wheels for torch/torchaudio, then IndexTTS from default branch, then service deps (no DeepSpeed)
RUN pip install "torch==2.8.*" "torchaudio==2.8.*" && \
    pip install "indextts @ git+https://github.com/index-tts/index-tts" && \
    pip install fastapi==0.115.0 "uvicorn[standard]==0.30.6"

COPY main.py .
EXPOSE 9010
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9010", "--workers", "1", "--timeout-keep-alive", "1800"]