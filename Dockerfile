FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 가상 환경 생성 및 활성화
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 가상 환경에서 패키지 설치
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir numpy opencv-python-headless Pillow flask flask-cors segment-anything tqdm requests

WORKDIR /app

COPY . /app

# 모델 파일 다운로드
RUN if [ ! -f /app/models/vit_b_model.pth ]; then \
      wget -O /app/models/vit_b_model.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth; \
    fi

CMD ["python3", "main.py"]