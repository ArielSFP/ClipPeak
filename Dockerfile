FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies (excluding ffmpeg - we'll install NVENC-enabled version)
RUN apt-get update && apt-get install -y \
        python3 \
        python3-pip \
        git \
        wget \
        curl \
        bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Install build dependencies for FFmpeg compilation
RUN apt-get update && apt-get install -y \
        build-essential \
        yasm \
        nasm \
        libx264-dev \
        libx265-dev \
        libvpx-dev \
        libfdk-aac-dev \
        libmp3lame-dev \
        libopus-dev \
        libvorbis-dev \
        libtheora-dev \
        libvdpau-dev \
        libva-dev \
        libxcb1-dev \
        libxcb-shm0-dev \
        libxcb-xfixes0-dev \
        pkg-config \
        libass-dev \
        libfreetype6-dev \
        libsdl2-dev \
        texinfo \
        zlib1g-dev \
        cmake \
        git \
        autoconf \
        automake \
        libtool \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA Video Codec SDK headers (required for NVENC compilation)
# The nv-codec-headers provide the necessary headers for NVENC support
RUN cd /tmp && \
    git clone --depth 1 https://github.com/FFmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && \
    make && \
    make install && \
    cd / && \
    rm -rf /tmp/nv-codec-headers

# Compile FFmpeg from source with NVENC support
RUN cd /tmp && \
    git clone --depth 1 https://git.ffmpeg.org/ffmpeg.git ffmpeg-src && \
    cd ffmpeg-src && \
    ./configure \
        --enable-nonfree \
        --enable-gpl \
        --enable-libx264 \
        --enable-libx265 \
        --enable-libvpx \
        --enable-libfdk-aac \
        --enable-libmp3lame \
        --enable-libopus \
        --enable-libvorbis \
        --enable-libtheora \
        --enable-libass \
        --enable-libfreetype \
        --enable-cuda \
        --enable-cuvid \
        --enable-nvenc \
        --enable-libnpp \
        --extra-cflags="-I/usr/local/cuda/include -I/usr/local/cuda/include/nv-codec-headers" \
        --extra-ldflags="-L/usr/local/cuda/lib64" \
        --prefix=/usr/local \
        --enable-shared \
        --disable-static \
        --pkg-config-flags="--static" && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /tmp/ffmpeg-src && \
    ldconfig && \
    # Verify FFmpeg installation and NVENC support
    ffmpeg -version && \
    ffmpeg -encoders | grep -i nvenc && \
    echo "✅ FFmpeg with NVENC support compiled and installed successfully"

# Clean up build dependencies to reduce image size
RUN apt-get purge -y \
        build-essential \
        yasm \
        nasm \
        cmake \
        git \
        autoconf \
        automake \
        libtool \
    && apt-get autoremove -y \
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
