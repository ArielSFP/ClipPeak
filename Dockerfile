FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
        python3 \
        python3-pip \
        ffmpeg \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

# (Optional but recommended) upgrade pip so it understands modern wheels well
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Copy requirements and install all pinned deps (excluding torch stack)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA 12.1 — pinned to the versions that worked in your build
# Final versions after the "uninstall/reinstall" dance:
#   torch        2.5.1+cu121
#   torchvision  0.20.1+cu121
#   torchaudio   2.5.1+cu121 :contentReference[oaicite:0]{index=0}
RUN pip3 install --no-cache-dir \
        torch==2.5.1+cu121 \
        torchvision==0.20.1+cu121 \
        torchaudio==2.5.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121

# Copy the rest of the app
COPY . .

# Install auto-subtitle ONLY from the reelsfy folder, as requested
RUN if [ -d "reelsfy_folder/utils/auto-subtitle" ]; then \
        python3 -m pip install -e reelsfy_folder/utils/auto-subtitle/; \
    else \
        echo "Warning: auto-subtitle not found in reelsfy_folder!"; \
    fi

# (Optional) If you *really* want to pre-download models at build time,
# you can re-add a step like this. I’d *not* recommend baking the big
# models into the image, but here is the pattern:
#
RUN python3 -c "from faster_whisper import WhisperModel; \
    WhisperModel('tiny', device='cpu'); \
    WhisperModel('turbo', device='cpu'); \
    WhisperModel('ivrit-ai/whisper-large-v3-turbo-ct2', device='cpu')"

EXPOSE 8080
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
