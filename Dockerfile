FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA 12.1
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Make sure auto_subtitles is installed
#RUN if [ -d "utils/auto-subtitle" ]; then \
#       pip3 install -e utils/auto-subtitle/; \
#    else \
#       echo "Warning: auto-subtitle directory not found!"; \
#    fi

# Copy application code
COPY . .

EXPOSE 8080

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

CMD uvicorn api:app --host 0.0.0.0 --port $PORT
