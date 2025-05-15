# Background-Remover
![Remover](/images/MainImage.png)  

아이폰에서 사진 속 대상을 손가락으로 클릭하여 피사체만 배경과 분리한 경험이 있을 겁니다. [누끼](https://namu.wiki/w/%EB%88%84%EB%81%BC)를 딴다고도 하죠. 이와 비슷한 효과를 만들 수 있는 웹 서비스입니다.

## 📖 프로젝트 개요

**Background-Remover**는 Meta AI의 최첨단 Segment Anything Model 2.1 (SAM 2.1)을 활용한 AI 기반 배경 제거 서비스입니다. SAM 2.1의 프롬프트 기반 세그멘테이션 기능을 클릭 인터페이스로 단순화하여, 누구나 쉽게 사용할 수 있는 원클릭 배경 제거 기능을 구현했습니다.

### 🎯 핵심 기능

- **원클릭 배경 제거**: SAM 2.1의 포인트 프롬프트 기능을 활용한 직관적인 클릭 기반 배경 제거
- **다중 포인트 지원**: 복잡한 객체의 경우 여러 지점을 클릭하여 정확도 향상
- **고품질 결과**: SAM 2.1의 뛰어난 세그멘테이션 성능으로 정밀한 경계 추출
- **빠른 처리**: GPU 가속을 통해 평균 2초 내 처리 완료
- **투명 배경**: PNG 형식으로 투명한 배경의 결과 이미지 제공

### 🚀 주요 특징

- **프롬프트 기반 모델의 단순화**: SAM 2.1은 다양한 프롬프트(클릭, 박스, 마스크 등)를 지원하지만, 본 프로젝트에서는 가장 직관적인 클릭 프롬프트 방식으로 단순화
- **높은 정확도**: AI가 클릭한 지점을 기반으로 객체의 경계를 정밀하게 감지하여 자연스러운 결과 생성
- **빠른 성능**: 클라우드 GPU 환경에서 실시간에 가까운 처리 속도 제공
- **다양한 활용**: 프로필 사진, 제품 이미지, 디자인 소재 등 다양한 용도로 활용 가능

## ✏️기술 스택 및 특징

### 📌 [Segment Anything Model 2.1 (SAM 2.1)](https://segment-anything.com/)

Meta AI에서 개발한 최신 세그멘테이션 모델입니다. SAM 2.1은 **프롬프트 기반 세그멘테이션** 모델로, 다양한 형태의 입력(프롬프트)을 받아 객체를 분할합니다:

- **다양한 프롬프트 유형 지원**: 포인트 클릭, 바운딩 박스, 마스크 등 다양한 방식으로 객체 지정 가능
- **인터랙티브 개선**: 사용자가 추가 프롬프트를 제공하여 결과를 반복적으로 개선할 수 있음
- **범용성**: 사람, 동물, 사물 등 다양한 객체를 별도 학습 없이 인식
- **이미지 및 비디오 지원**: 단일 모델로 이미지와 비디오 세그멘테이션 모두 처리

SAM 2.1은 4가지 크기의 모델을 제공하며, 현재 프로젝트에서는 성능과 속도의 균형이 좋은 **Hiera Base+** 모델을 사용합니다.

| 특성 | Hiera Tiny | Hiera Small | Hiera Base+ | Hiera Large |
|------|-----------|-------------|-------------|-------------|
| **모델 크기** | ~30MB | ~40MB | ~160MB | ~240MB |
| **적합한 사용 환경** | 모바일, 저사양 | 일반 데스크톱 | 균형잡힌 성능 | 고성능 서버 |
| **처리 속도** | 가장 빠름 | 빠름 | 중간 | 느림 |
| **정확도** | 기본 | 좋음 | 매우 좋음 | 최고 |

**선택 이유**: OpenCV나 전통적인 이미지 세그멘테이션 방법 대비 사전 학습이 필요 없고, 다양한 프롬프트 방식으로 유연하게 사용할 수 있어 일반 사용자도 쉽게 활용할 수 있습니다. 특히 본 프로젝트에서는 클릭 기반 프롬프트 기능을 활용하여 직관적인 UI를 제공합니다.

### 📌 [FastAPI](https://fastapi.tiangolo.com/)

빠르고 현대적인 Python 웹 프레임워크입니다:

- **자동 API 문서 생성**: 코드만 작성하면 자동으로 Swagger 문서 생성
- **비동기 처리**: 여러 요청을 동시에 처리하여 응답 속도 향상
- **타입 검증**: 업로드 파일과 좌표 데이터를 자동으로 검증

**선택 이유**: 처음에는 간단함 때문에 Flask를 사용했지만, Flask는 동기적으로 요청을 처리하는 방식이어서 서버에 많은 요청이 들어올 때 병목 현상이 발생할 수 있습니다. FastAPI는 비동기 처리를 지원하여 동시에 여러 배경 제거 요청을 효율적으로 처리할 수 있어 전환하게 되었습니다.

### 📌 [CUDA + PyTorch](https://pytorch.org/)

GPU 가속을 통한 빠른 이미지 처리를 위해 사용됩니다:

- **CUDA**: NVIDIA GPU의 병렬 처리 능력 활용 (현재 버전: 12.6)
- **PyTorch**: 딥러닝 모델 실행 및 GPU 연산 관리 (현재 버전: 2.7.0)
- **Mixed Precision (FP16)**: 메모리 사용량을 절반으로 줄이면서도 성능 유지

**선택 이유**: 초기에는 CPU만으로 SAM 모델을 실행했는데, 그리 큰 사진이 아님에도 1장당 평균 20~30초가 소요되었습니다. GPU 병렬 처리 연산을 위해 CUDA를 도입한 결과, 처리 시간이 평균 1초 이내로 대폭 단축되어 실용적인 서비스가 가능해졌습니다.

> **성능 개선 효과**: CPU 처리 대비 약 20-30배 빠른 처리 속도 (평균 20~30초 → 1초 이내)

### 📌 [Docker](https://www.docker.com/)

일관된 실행 환경을 제공하기 위해 컨테이너화했습니다:

- **NVIDIA Container Toolkit**: Docker 컨테이너에서 GPU 사용
- **환경 격리**: 개발/운영 환경 차이로 인한 문제 방지
- **간편한 배포**: 어느 서버에서든 동일하게 실행

**선택 이유**: 복잡한 CUDA, PyTorch, SAM 2.1 의존성을 한 번에 패키징할 수 있어 설치 과정을 대폭 단순화했습니다. 가상 환경이나 직접 설치 대비 환경 차이로 인한 문제를 방지하고, 어떤 시스템에서든 동일한 환경에서 실행할 수 있습니다.

### 📌 [GitHub Actions](https://docs.github.com/en/actions)

자동 배포 시스템을 구축했습니다:

- **자동 CI/CD**: main 브랜치에 코드 푸시 시 자동으로 서버 배포
- **배포 상태 알림**: Mattermost로 배포 결과 실시간 알림
- **리소스 관리**: 배포 전 자동으로 Docker 이미지 및 볼륨 정리

**선택 이유**: Jenkins나 GitLab CI/CD 대비 설정이 간단하고 GitHub과 완벽하게 통합됩니다. 별도 서버 구축이 불필요하며, YAML 파일로 배포 과정을 코드화해 관리할 수 있어 유지보수가 쉽습니다.

### 📌 병렬 처리 및 동시성 관리

프로젝트에는 성능 최적화를 위한 여러 병렬 처리 기법이 적용되었습니다:

- **ThreadPoolExecutor**: SAM 2.1이 생성한 여러 마스크 후보 중 최적 결과를 선택하기 위해 마스크 평가 작업을 병렬로 처리
- **Semaphore를 통한 동시성 제어**: GPU 메모리 관리와 모델 접근을 제어하기 위해 세마포어로 동시 요청 수 제한 (기본값: 2개)
- **AsyncIO Lock**: SAM 모델 로드 및 GPU 메모리 정리 작업에서 스레드 안전성 보장

**구현 배경**: 사용자가 여러 지점을 클릭하면 SAM 2.1이 각 클릭에 대해 여러 마스크 후보를 생성합니다. 최적의 마스크를 빠르게 선택하기 위해 병렬 평가를 도입했으며, GPU 리소스의 안전한 공유를 위해 동시성 제어를 구현했습니다.

## 🛠️설치 및 실행

### 시스템 요구사항
- NVIDIA GPU (CUDA 지원)
- Docker 및 Docker Compose
- 8GB 이상의 GPU 메모리 권장

### 1. GPU 지원 설정

#### Windows (Docker Desktop)
Docker Desktop 설정에서 GPU 지원을 활성화합니다:

```json
{
  "experimental": true,
  "features": {
    "buildkit": true,
    "nvidia-gpu": true
  }
}
```

#### Linux (Ubuntu)
NVIDIA Container Toolkit을 설치합니다:

```bash
# NVIDIA 드라이버 설치
sudo apt-get update
sudo apt-get install -y nvidia-driver-535

# Container Toolkit 설치
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Docker 설정 수정
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF

# Docker 재시작
sudo systemctl restart docker
```

### 2. 프로젝트 실행

#### 로컬 개발 환경
```bash
# 프로젝트 클론
git clone <repository-url>
cd background-remover

# 개발용 컨테이너 실행
docker-compose -f docker-compose-local.yml up --build

# 브라우저에서 http://localhost:5000/docs 접속
```

#### 프로덕션 환경
```bash
# 환경 변수 설정
export EC2_HOST=your-server-ip
export EC2_PORT=5000

# 프로덕션 컨테이너 실행
docker-compose -f docker-compose-prod.yml up -d --build
```

## 📖API 사용법

### 주요 엔드포인트

#### `POST /background-removal/remove`
이미지에서 클릭한 객체의 배경을 제거합니다.

**요청 파라미터:**
- `image`: 업로드할 이미지 파일
- `points`: 클릭한 좌표들 (JSON 배열 형식)
  - 예: `[[100, 200]]` (한 곳 클릭)
  - 예: `[[100, 200], [300, 400]]` (두 곳 클릭)

**예시:**
```bash
curl -X POST "http://localhost:5000/background-removal/remove" \
  -F "image=@example.jpg" \
  -F 'points=[[100, 200], [300, 400]]'
```

**응답:**
- PNG 형식의 배경이 제거된 이미지 (투명 배경)

### API 문서
서버 실행 후 다음 URL에서 대화형 API 문서를 확인할 수 있습니다:
- **Swagger UI**: `http://localhost:5000/docs`
- **ReDoc**: `http://localhost:5000/redoc`

## ⚙️주요 설정

프로젝트의 핵심 설정은 `config/settings.py`에서 관리됩니다:

```python
# 모델 설정
MODEL = {
    'TYPE': 'hiera_b',  # 모델 크기 (hiera_t, hiera_s, hiera_b, hiera_l)
    'DEVICE': 'cuda',   # 'cuda' 또는 'cpu'
}

# 성능 최적화
MODEL_PERFORMANCE = {
    'USE_MIXED_PRECISION': True,  # FP16 사용으로 메모리 절약
}

# 처리 옵션
MAX_IMAGE_SIZE = 1024  # 최대 이미지 크기 (픽셀)
API_CONFIG = {
    'MAX_CONCURRENT_REQUESTS': 2,  # 동시 처리 요청 수
}
```

## 🚀성능 특징

- **빠른 처리**: GPU 가속으로 평균 2초 내 처리
- **정확한 세그멘테이션**: SAM 2.1의 고성능 객체 분할
- **다중 포인트 지원**: 여러 곳 클릭으로 정확도 향상
- **자연스러운 경계**: 마스크 소프트닝으로 부드러운 경계 처리
- **메모리 최적화**: Mixed Precision으로 GPU 메모리 효율적 사용

## 💡사용 팁

1. **객체 전체를 포함하도록 클릭**: 객체의 중심보다는 원하는 영역을 모두 포함할 수 있는 위치를 클릭하는 것이 좋습니다
2. **다중 클릭 활용**: 복잡한 객체나 여러 부분으로 구성된 객체의 경우 각 부분을 클릭하여 정확도를 높이세요
3. **적절한 이미지 크기**: 너무 큰 이미지는 자동으로 리사이즈되어 처리됩니다 (1024픽셀 기준)

## 🧪테스트

양자화 성능을 비교해보고 싶다면:

```bash
# 단일 이미지 테스트 (FP16 vs FP32 비교)
python tests/performance/test_quantization.py --image image.jpg --x 100 --y 200

# 폴더 내 모든 이미지 테스트
python tests/performance/test_quantization.py --folder test_images/
```