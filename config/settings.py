import torch
import multiprocessing

# =============================================================================
# 이미지 처리 관련 설정
# =============================================================================

# 이미지 크기 제한
MAX_IMAGE_SIZE = 1024
LARGE_IMAGE_THRESHOLD = 1000  # 대용량 이미지 기준 (API, BackgroundRemoval 공통 사용)

# 이미지 분석 파라미터
IMAGE_ANALYSIS = {
    'CLICK_RADIUS': 20,
    'EDGE_CHECK_RADIUS': 5,
    'CANNY_THRESHOLD_LOW': 50,
    'CANNY_THRESHOLD_HIGH': 150,
}

# =============================================================================
# 마스크 처리 관련 설정
# =============================================================================

# 마스크 정제 공통 설정
MASK_REFINEMENT = {
    'CLOSE_KERNEL_SIZE': 3,     # Closing 커널 크기
    'OPEN_KERNEL_SIZE': 1,      # Opening 커널 크기
    'THRESHOLD': 0.5,           # 마스크 이진화 임계값
    'SOFTEN_EDGES': True,       # 엣지 소프트닝 활성화
    
    'GAUSSIAN_BLUR_SIGMA': 2.0, # 가우시안 블러 강도
    'FEATHER_RADIUS': 3,        # 페더링 반경 (픽셀)
    'POST_PROCESS_DILATION': 3, # 마스크 팽창 정도
    'EDGE_SMOOTHING': True,     # 엣지 스무딩 활성화
}

# 마스크 선택 기준
MASK_SELECTION = {
    'MIN_SIZE_PERCENTAGE': 2,
    'MAX_SIZE_PERCENTAGE': 85,
    'SCORE_THRESHOLD': 0.4,
    'TOP_MASKS_COUNT': 5,
    'EDGE_WEIGHT': 0.8,
    'SCORE_WEIGHT': 0.2,
    'SIZE_WEIGHT': 0.6,
    'CLICK_INCLUSION_THRESHOLD': 0.1,
    'SIZE_SCORE_BASELINE': 50,
    'SIZE_SCORE_DEVIATION': 45,
}

# =============================================================================
# SAM 2.1 모델 관련 설정
# =============================================================================

# 장치 및 양자화 설정
MODEL_PERFORMANCE = {
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'USE_MIXED_PRECISION': True,            # 양자화 사용 여부
    'QUANTIZATION_DTYPE': torch.bfloat16,   # 양자화 데이터 타입
}

# SAM 2.1 모델 설정
MODEL = {
    'VERSION': 'SAM2',
    'VERSION_NAME': 'SAM 2.1',
    'TYPE': 'hiera_b',
    'SUB_DIRECTORY_NAME': 'sam2',
    
    # 성능 설정 참조
    'DEVICE': MODEL_PERFORMANCE['DEVICE'],
    'USE_MIXED_PRECISION': MODEL_PERFORMANCE['USE_MIXED_PRECISION'],
    'QUANTIZATION_DTYPE': MODEL_PERFORMANCE['QUANTIZATION_DTYPE'],
    
    # 모델 파일 URL
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

# SAM 2.1 추가 기능
SAM2_OPTIONS = {
    'AUTO_NEGATIVE_POINTS': False,       # 자동 negative points 생성
    'NEGATIVE_POINT_MARGIN': 50,         # negative points 여백
    'MEMORY_BANK_SIZE': 5,               # 비디오 처리용 (이미지에서는 미사용)
}

# =============================================================================
# 서비스 및 처리 관련 설정
# =============================================================================

# API 서버 설정
API_CONFIG = {
    'MAX_CONCURRENT_REQUESTS': 2,       # 동시 처리 요청 수
    'LARGE_IMAGE_THRESHOLD': LARGE_IMAGE_THRESHOLD,  # 공통 임계값 참조
}

# 배경 제거 서비스 설정
BACKGROUND_REMOVAL = {
    'TOTAL_STEPS': 5,                   # 처리 단계 수
    'LARGE_IMAGE_THRESHOLD': LARGE_IMAGE_THRESHOLD,  # 공통 임계값 참조
    'DEBUG_INFO_SAVE': False,           # 디버그 정보 저장 여부
    'MASK_REFINEMENT': MASK_REFINEMENT, # 공통 마스크 정제 설정 참조
}

# GPU 메모리 관리 설정
GPU_MEMORY = {
    'CLEANUP_ON_ERROR': True,           # 에러 시 자동 정리
    'ENABLE_AUTO_CLEANUP': True,        # 자동 메모리 정리 활성화
}

# =============================================================================
# 시스템 리소스 관련 설정
# =============================================================================

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