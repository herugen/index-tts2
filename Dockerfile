FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*

# install index-tts and dependencies
WORKDIR /
RUN git clone https://github.com/index-tts/index-tts.git \
    && pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple "torch==2.8.*" "torchaudio==2.8.*" \
    && pip install /index-tts \
    && rm -rf /index-tts

WORKDIR /app
COPY main.py .
EXPOSE 9010
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9010", "--workers", "1", "--timeout-keep-alive", "1800"]
