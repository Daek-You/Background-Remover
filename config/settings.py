import torch
import multiprocessing

# 이미지 처리 설정
MAX_IMAGE_SIZE = 1024

# API 서버 관련 설정
API_CONFIG = {
    'MAX_CONCURRENT_REQUESTS': 2,     # 동시 처리 가능한 최대 요청 수
    'LARGE_IMAGE_THRESHOLD': 1000,    # 대용량 이미지로 간주하는 기준 (픽셀)
}

# 이미지 분석 파라미터
IMAGE_ANALYSIS = {
    'CLICK_RADIUS': 20,
    'EDGE_CHECK_RADIUS': 5,
    'CANNY_THRESHOLD_LOW': 50,
    'CANNY_THRESHOLD_HIGH': 150,
}

# 마스크 선택 기준
MASK_SELECTION = {
    'MIN_SIZE_PERCENTAGE': 3,
    'MAX_SIZE_PERCENTAGE': 95,
    'SCORE_THRESHOLD': 0.5,
    'TOP_MASKS_COUNT': 5,
    'EDGE_WEIGHT': 0.8,
    'SCORE_WEIGHT': 0.2,
    'SIZE_WEIGHT': 0.6,

    'CLICK_INCLUSION_THRESHOLD': 0.5,  # 클릭 위치가 마스크에 포함되는지 판단하는 임계값
    'SIZE_SCORE_BASELINE': 50,         # 크기 점수 계산시 기준점 (이상적인 크기의 퍼센티지)
    'SIZE_SCORE_DEVIATION': 45,        # 크기 점수 계산시 편차 허용 범위
}

# SAM 2.1 모델 설정
MODEL = {
    'VERSION': 'SAM2',
    'VERSION_NAME': 'SAM 2.1',
    'TYPE': 'hiera_b',
    'SUB_DIRECTORY_NAME': 'sam2',
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # 성능 최적화
    'USE_MIXED_PRECISION': True,
    'QUANTIZATION_DTYPE': torch.bfloat16,
    
    # 모델 다운로드 URL
    'URLS': {
        'hiera_t': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt',
        'hiera_s': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt',
        'hiera_b': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt',
        'hiera_l': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt'
    },
    
    # 설정 파일 매핑
    'CONFIG': {
        'MODEL_TYPE_MAP': {
            'hiera_t': 'sam2.1_hiera_t.yaml',
            'hiera_s': 'sam2.1_hiera_s.yaml',
            'hiera_b': 'sam2.1_hiera_b+.yaml',
            'hiera_l': 'sam2.1_hiera_l.yaml'
        }
    },
    
    # 파일 크기 검증 (MB)
    'MIN_SIZES': {
        'hiera_t': 30,
        'hiera_s': 40,
        'hiera_b': 160,
        'hiera_l': 240,
    },
    
    # 디렉토리 구성
    'DIRS': {
        'ENV_VAR': 'MODELS_DIR',
        'DOCKER_VOLUME': '/app/models',
        'DEFAULT_SUBDIR': 'models'
    },
    
    # 다운로드 옵션
    'DOWNLOAD': {
        'MAX_RETRIES': 3,
        'TIMEOUT': 30,
        'BLOCK_SIZE': 1024 * 1024,
        'RETRY_DELAY': 5
    }
}

# SAM 2.1 추가 옵션
SAM2_OPTIONS = {
    'AUTO_NEGATIVE_POINTS': False,
    'NEGATIVE_POINT_MARGIN': 10,
    'MEMORY_BANK_SIZE': 5,  # 비디오 처리용, 이미지는 미사용
}

# 병렬 처리 설정
THREAD_POOL = {
    'MAX_WORKERS': min(32, multiprocessing.cpu_count() + 4),
    'USE_PROCESS_POOL': False,
}

# 로깅 설정
LOGGING = {
    'LEVEL': 'INFO',
    'FORMAT': '[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s',
    'DATE_FORMAT': '%Y-%m-%d %H:%M:%S',
    'COLORS': {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
}
