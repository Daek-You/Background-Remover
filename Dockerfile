FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04


RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 가상 환경 생성 및 활성화
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 가상 환경에서 패키지 설치
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir numpy opencv-python Pillow flask flask-cors segment-anything tqdm

WORKDIR /app

COPY . /app

# 모델 파일 다운로드
RUN wget -O /app/models/vit_b_model.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

CMD ["python3", "main.py"]