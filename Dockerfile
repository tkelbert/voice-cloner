FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip \
        git git-lfs \
        ffmpeg \
        espeak-ng && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python3

RUN pip3 install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir coqui-tts PySide6 sounddevice scipy

WORKDIR /app

COPY . /app/

CMD ["python3", "gui.py"]
