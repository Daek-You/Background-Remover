"""환경별 설정을 관리하는 모듈"""
import os
from config.settings import LOGGING

# 환경 변수에서 현재 환경 가져오기 (기본값: development)
ENV = os.getenv('ENV', 'development')

# 환경별 설정
ENVIRONMENTS = {
    # 로컬 개발용
    'development': {
        'DEBUG': False,
        'TESTING': False,
        'HOST': '0.0.0.0',
        'PORT': 5000,
        'CORS_ORIGINS': ['http://localhost:3000', 'http://127.0.0.1:3000'],
        'LOGGING': {
            'LEVEL': 'DEBUG',
            'FORMAT': LOGGING['FORMAT'],
            'DATE_FORMAT': LOGGING['DATE_FORMAT'],
            'COLORS': LOGGING['COLORS']
        }
    },

    # 운영 서버(실제 서비스 환경)
    'production': {
        'DEBUG': False,
        'TESTING': False,
        'HOST': '0.0.0.0',       # 0.0.0.0은 모든 네트워크 인터페이스에 바인딩한다는 의미로, FastAPI(Uvicorn)가 사용 가능한 모든 IP 주소를 통해 접근 가능
        'PORT': int(os.getenv('EC2_PORT', 5000)),
        'CORS_ORIGINS': [
            "http://s12p31d202.com", 
            "https://s12p31d202.com",
            "http://0v0.co.kr",
            "https://0v0.co.kr",
        ],
        'LOGGING': {
            'LEVEL': 'INFO',
            'FORMAT': LOGGING['FORMAT'],
            'DATE_FORMAT': LOGGING['DATE_FORMAT'],
            'COLORS': LOGGING['COLORS']
        }
    },
}

def get_environment():
    """현재 환경의 설정을 반환하는 함수"""
    if ENV not in ENVIRONMENTS:
        print(f"알 수 없는 환경: '{ENV}'. '개발(development)' 환경으로 설정됩니다.")
        return ENVIRONMENTS['development']
    return ENVIRONMENTS[ENV]

# 현재 환경의 설정 가져오기
current_env = get_environment()