"""환경별 설정을 관리하는 모듈"""
import os
from config.settings import LOGGING

# 환경 변수에서 현재 환경 가져오기 (기본값: development)
ENV = os.getenv('FLASK_ENV', 'development')

# 환경별 설정
ENVIRONMENTS = {
    # 로컬 개발용
    'development': {
        'DEBUG': True,
        'TESTING': False,
        'HOST': '0.0.0.0',
        'PORT': 5000,
        'CORS_ORIGINS': ['http://localhost:*', 'http://127.0.0.1:*'],
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
        'HOST': '0.0.0.0',   # TODO: 우리 도메인으로 변경
        'PORT': 5000,        # TODO: 우리 포트로 변경
        'CORS_ORIGINS': [],  # TODO: 우리 도메인으로 변경
        'LOGGING': {
            'LEVEL': 'INFO',
            'FORMAT': LOGGING['FORMAT'],
            'DATE_FORMAT': LOGGING['DATE_FORMAT'],
            'COLORS': LOGGING['COLORS']
        }
    },

    # 테스트 서버
    'testing': {
        'DEBUG': True,
        'TESTING': True,
        'HOST': '127.0.0.1',                        # TODO: 우리 도메인으로 변경
        'PORT': 5000,                               # TODO: 우리 포트로 변경
        'CORS_ORIGINS': ['http://localhost:3000'],  # TODO: 우리 도메인으로 변경
        'LOGGING': {
            'LEVEL': 'DEBUG',
            'FORMAT': LOGGING['FORMAT'],
            'DATE_FORMAT': LOGGING['DATE_FORMAT'],
            'COLORS': LOGGING['COLORS']
        }
    }
}

def get_environment():
    """현재 환경의 설정을 반환하는 함수"""
    if ENV not in ENVIRONMENTS:
        logger.warning(f"알 수 없는 환경: '{ENV}'. '개발(development)' 환경으로 설정됩니다.")
        return ENVIRONMENTS['development']
    return ENVIRONMENTS[ENV]

# 현재 환경의 설정 가져오기
current_env = get_environment()
