import torch
import multiprocessing

# 이미지 크기 제한
MAX_IMAGE_SIZE = 1024  # 최대 처리 이미지 크기

# 이미지 분석 관련 설정
IMAGE_ANALYSIS = {
    'CLICK_RADIUS': 20,                 # 클릭 위치 주변 분석 반경
    'EDGE_CHECK_RADIUS': 5,             # 에지 확인 반경
    'CANNY_THRESHOLD_LOW': 50,          # Canny 에지 검출 하한 임계값
    'CANNY_THRESHOLD_HIGH': 150,        # Canny 에지 검출 상한 임계값
}

# 마스크 선택 관련 설정 (SAM 2.1 최적화)
MASK_SELECTION = {
    'MIN_SIZE_PERCENTAGE': 3,           # 최소 마스크 크기 (%) - SAM 2.1은 더 작은 객체도 정확히 감지
    'MAX_SIZE_PERCENTAGE': 92,          # 최대 마스크 크기 (%) - 여유 공간 확대
    'SCORE_THRESHOLD': 0.6,             # 최고 점수 대비 최소 점수 비율 - SAM 2.1은 더 안정적
    'TOP_MASKS_COUNT': 3,               # 상위 마스크 선택 개수
    'EDGE_WEIGHT': 0.8,                 # 에지 정렬 가중치 - 강아지 털 등 복잡한 경계 처리 강화
    'SCORE_WEIGHT': 0.2,                # 점수 가중치
    'SIZE_WEIGHT': 0.6,                 # 크기 가중치
}

# SAM 2.1 모델 관련 설정
MODEL = {
    'VERSION': 'SAM2',                  # SAM 버전 (SAM2 = SAM 2.1)
    'VERSION_NAME': 'SAM 2.1',          # 사람이 읽기 쉬운 버전 이름
    'TYPE': 'hiera_l',                  # SAM 2.1 모델 타입 (hiera_large 추천)
    'SUB_DIRECTORY_NAME': 'sam2',       # 모델 파일 저장 서브디렉토리
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # SAM 2.1 양자화 설정 (성능 최적화)
    'USE_MIXED_PRECISION': True,            # SAM 2.1에서는 기본 활성화 권장
    'QUANTIZATION_DTYPE': torch.bfloat16,   # SAM 2.1은 bfloat16 최적화됨
    
    # SAM 2.1 체크포인트 다운로드 URL
    'URLS': {
        'hiera_t': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt',
        'hiera_s': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt',
        'hiera_b': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt',
        'hiera_l': 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt'
    },
    
    # SAM 2.1 설정 파일 매핑 (build_sam2에서 사용)
    'CONFIG_MAP': {
        'hiera_t': 'sam2_hiera_tiny.yaml',
        'hiera_s': 'sam2_hiera_small.yaml', 
        'hiera_b': 'sam2_hiera_base_plus.yaml',
        'hiera_l': 'sam2_hiera_large.yaml'
    },
    
    # 모델 파일 크기 검증용 최소 크기 (MB)
    'MIN_SIZES': {
        'hiera_t': 30,   # tiny: ~38MB
        'hiera_s': 40,   # small: ~46MB  
        'hiera_b': 160,  # base_plus: ~179MB
        'hiera_l': 240,  # large: ~291MB
    },
    
    # 사용 가능한 모델 타입 목록
    'TYPES': ['hiera_t', 'hiera_s', 'hiera_b', 'hiera_l'],
}

# SAM 2.1 특화 설정
SAM2_OPTIONS = {
    # 이미지 모서리에 자동으로 추가할 negative point 설정
    'AUTO_NEGATIVE_POINTS': True,       # 배경 제거 정확도 향상을 위해 활성화
    'NEGATIVE_POINT_MARGIN': 10,        # 이미지 가장자리로부터 떨어뜨릴 픽셀
    
    # SAM 2.1의 메모리 관련 설정 (현재는 이미지만 처리하므로 기본값)
    'MEMORY_BANK_SIZE': 5,              # 비디오 처리용 (이미지는 1)
}

# 스레드 풀 관련 설정
THREAD_POOL = {
    'MAX_WORKERS': min(32, multiprocessing.cpu_count() + 4),  # 스레드 풀 최대 크기
    'USE_PROCESS_POOL': False,                                # 프로세스 풀 사용 여부
}

# 로깅 관련 설정
LOGGING = {
    'LEVEL': 'INFO',                    # 로깅 레벨
    'FORMAT': '[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s',  # 로그 포맷
    'DATE_FORMAT': '%Y-%m-%d %H:%M:%S', # 날짜 포맷
    'COLORS': {                         # 로그 레벨별 색상 (터미널 출력용)
        'DEBUG': '\033[36m',            # Cyan
        'INFO': '\033[32m',             # Green
        'WARNING': '\033[33m',          # Yellow
        'ERROR': '\033[31m',            # Red
        'CRITICAL': '\033[35m',         # Magenta
        'RESET': '\033[0m'              # Reset
    }
}