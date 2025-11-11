# ClipPeak - GPU-Accelerated Video Processing
# Base image with CUDA 12.1 and cuDNN 9
FROM nvidia/cuda:12.1.0-cudnn9-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support FIRST
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download TalkNet model at build time (so it's baked into the image)
RUN mkdir -p fast-asd/models && \
    if [ ! -f fast-asd/models/pretrain_TalkSet.model ]; then \
        wget -O fast-asd/models/pretrain_TalkSet.model \
        https://github.com/TaoRuijie/TalkNet-ASD/releases/download/v0.1/pretrain_TalkSet.model || \
        echo "Warning: Could not download TalkNet model, ensure it exists in repo"; \
    fi

# Create necessary directories
RUN mkdir -p tmp results save/pyavi save/pycrop

# Expose port (Cloud Run uses PORT env var)
EXPOSE 8080

# Environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV TALKNET_DEBUG=false

# Health check (optional but recommended)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Run the FastAPI app with uvicorn
CMD exec uvicorn api:app --host 0.0.0.0 --port $PORT --workers 1

