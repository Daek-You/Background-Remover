# Background-Remover
![Remover](/images/MainImage.png)  

아이폰에서 사진 속 대상을 손가락으로 클릭하여, 피사체만 배경과 분리한 경험이 있을 겁니다. [누끼](https://namu.wiki/w/%EB%88%84%EB%81%BC)를 딴다고도 하죠. 이와 비슷한 효과를 낼 수 있게 만들었습니다. 사진과 대상 위치 정보를 받아, 해당 대상을 누끼딸 수 있게 해줍니다.  

## ✏️기술 및 모델
### 📌[Segment Anything Model (SAM)](https://segment-anything.com/)

Meta AI에서 개발한 모델로, 이미지에서 객체를 분할(Segmentation)하기 위해 설계된 인공지능 모델입니다. 일반적인 세그멘테이션 모델과 다르게, SAM은 다음과 같은 특징이 있습니다.  
- <b>프롬프트 기반 세그멘테이션</b>: 사용자가 클릭한 지점을 입력으로 받아 해당 객체를 식별합니다.
- <b>범용성</b>: 다양한 종류의 객체를 인식할 수 있도록 대규모 데이터셋으로 학습되었습니다.
- <b>Zero-shot 학습</b>: 새로운 객체 유형을 별도 학습 없이 인식할 수 있습니다.  

SAM에는 `ViT-B`, `ViT-L`, `ViT-H` 이렇게 3가지 버전이 있습니다.  

| 특성 | ViT-B (Base) | ViT-L (Large) | ViT-H (Huge) |
|------|-------------|--------------|--------------|
| **모델 크기** | 91M 파라미터 | 308M 파라미터 | 636M 파라미터 |
| **파일 크기** | 358MB | 1.2GB | 2.4GB |
| **다운로드 URL** | [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) | [sam_vit_l_0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) | [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| **정확도** | 가장 낮음 | 중간 | 가장 높음 |
| **속도** | 가장 빠름 | 중간 | 가장 느림 |
| **메모리 요구사항** | 낮음 | 중간 | 높음 |
| **적합한 사용 환경** | 모바일 장치, 저사양 환경 | 일반 데스크톱, 중간 성능 요구 | 고성능 워크스테이션, 서버 |
| **IoU 점수(성능 지표)** | 86.8 | 88.8 | 90.4 |
| **추론 시간(상대적)** | 1x | 약 2.5x | 약 4x |  

### 📌[CUDA(Compute Unified Device Architecture)](https://developer.nvidia.com/cuda-toolkit)

NVIDIA에서 개발한 병렬 컴퓨팅 플랫폼이자 프로그래밍 모델로, GPU(Graphics Processing Unit)를 활용하여 복잡한 계산을 빠르게 수행할 수 있도록 설계되었습니다. CUDA는 GPU의 강력한 병렬 처리 능력을 활용하여 CPU보다 훨씬 빠르게 대량의 데이터를 처리할 수 있습니다. 사진의 배경 제거 성능을 대폭 향상시키고자 도입하였습니다.  

<br>

## 📜의존성 관리

- [<b>[DockerHub] nvidi/cuda:12.1.0-cudnn8-runtime-ubuntu20.04</b>](https://hub.docker.com/r/nvidia/cuda/tags?name=12.1)  
25년 5월을 기준으로, `PyTorch`가 CUDA 12.1까지 지원
- [<b>Python 3.11.12</b>](https://www.python.org/downloads/release/python-31112/)
- [<b>python3.11-venv</b>](https://docs.python.org/ko/3.11/tutorial/venv.html)  
`externally-managed-environment` 오류 문제를 해결하고자 가상 환경을 사용
- [<b>deadsnakes PPA</b>](https://github.com/deadsnakes)  
Ubuntu에서 공식적으로 지원하지 않는 Python 버전을 설치할 수 있게 해주는 개인 패키지 아카이브
- [<b>opencv-python-headless 4.11.0.86</b>](https://pypi.org/project/opencv-python-headless/)  
GUI 기능이 없는 OpenCV 버전으로, 서버 환경에서 사용하기 적절 👉 서버 환경에서 불필요한 종속성을 제거  

```txt
Package                  Version
------------------------ ------------
blinker                  1.9.0
certifi                  2025.4.26
charset-normalizer       3.4.2
click                    8.1.8
filelock                 3.13.1
Flask                    3.1.0
flask-cors               5.0.1
fsspec                   2024.6.1
idna                     3.10
itsdangerous             2.2.0
Jinja2                   3.1.4
MarkupSafe               2.1.5
mpmath                   1.3.0
networkx                 3.3
numpy                    2.1.2
nvidia-cublas-cu12       12.1.3.1
nvidia-cuda-cupti-cu12   12.1.105
nvidia-cuda-nvrtc-cu12   12.1.105
nvidia-cuda-runtime-cu12 12.1.105
nvidia-cudnn-cu12        9.1.0.70
nvidia-cufft-cu12        11.0.2.54
nvidia-curand-cu12       10.3.2.106
nvidia-cusolver-cu12     11.4.5.107
nvidia-cusparse-cu12     12.1.0.106
nvidia-nccl-cu12         2.21.5
nvidia-nvjitlink-cu12    12.1.105
nvidia-nvtx-cu12         12.1.105
opencv-python-headless   4.11.0.86
pillow                   11.0.0
pip                      24.0
requests                 2.32.3
segment-anything         1.0
setuptools               65.5.0
sympy                    1.13.1
torch                    2.5.1+cu121
torchaudio               2.5.1+cu121
torchvision              0.20.1+cu121
tqdm                     4.67.1
triton                   3.1.0
typing_extensions        4.12.2
urllib3                  2.4.0
Werkzeug                 3.1.3
```  
