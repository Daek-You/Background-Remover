"""애플리케이션의 기본 설정값들을 관리하는 모듈"""

# 이미지 분석 관련 설정
IMAGE_ANALYSIS = {
    'CLICK_RADIUS': 20,                 # 클릭 위치 주변 분석 반경
    'EDGE_CHECK_RADIUS': 5,             # 에지 확인 반경
    'CANNY_THRESHOLD_LOW': 50,          # Canny 에지 검출 하한 임계값
    'CANNY_THRESHOLD_HIGH': 150,        # Canny 에지 검출 상한 임계값
}

# 마스크 선택 관련 설정
MASK_SELECTION = {
    'MIN_SIZE_PERCENTAGE': 5,           # 최소 마스크 크기 (%)
    'MAX_SIZE_PERCENTAGE': 90,          # 최대 마스크 크기 (%)
    'SCORE_THRESHOLD': 0.7,             # 최고 점수 대비 최소 점수 비율
    'TOP_MASKS_COUNT': 3,               # 상위 마스크 선택 개수
    'EDGE_WEIGHT': 0.7,                 # 에지 정렬 가중치
    'SCORE_WEIGHT': 0.3,                # 점수 가중치
    'SIZE_WEIGHT': 0.5,                 # 크기 가중치
}

# 모델 관련 설정
MODEL = {
    'TYPE': 'vit_h',                    # SAM 모델 타입
}

# 로깅 관련 설정
LOGGING = {
    'LEVEL': 'INFO',                    # 로깅 레벨
    'FORMAT': '[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s',  # 로그 포맷
    'DATE_FORMAT': '%Y-%m-%d %H:%M:%S', # 날짜 포맷
    'COLORS': {                         # 로그 레벨별 색상 (터미널 출력용)
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
}
