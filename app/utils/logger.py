""" 로깅 설정을 관리하는 모듈 """
import logging
import sys
from config.environments import current_env

# 로거 중복 설정 방지를 위한 플래그
_loggers = {}

class ColoredFormatter(logging.Formatter):
    """색상이 있는 로그 포맷터"""
    
    def format(self, record):
        # 로그 레벨에 따른 색상 적용
        levelname = record.levelname
        if levelname in current_env['LOGGING']['COLORS']:
            record.levelname = f"{current_env['LOGGING']['COLORS'][levelname]}{levelname}{current_env['LOGGING']['COLORS']['RESET']}"
        return super().format(record)

def setup_logger(name):
    """ 로거 설정 및 반환 """
    global _loggers
    
    # 이미 설정된 로거가 있으면 재사용
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    
    # 이미 핸들러가 설정되어 있다면 중복 설정 방지
    if logger.handlers:
        _loggers[name] = logger
        return logger
    
    # 로그 레벨 설정
    logger.setLevel(getattr(logging, current_env['LOGGING']['LEVEL']))
    
    # 프로파게이션 비활성화 (부모 로거로 전파 방지)
    logger.propagate = False
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, current_env['LOGGING']['LEVEL']))
    
    # 포맷터 설정 (터미널이 색상을 지원하는 경우에만 색상 적용)
    if sys.stdout.isatty():
        formatter = ColoredFormatter(current_env['LOGGING']['FORMAT'], datefmt=current_env['LOGGING']['DATE_FORMAT'])
    else:
        formatter = logging.Formatter(current_env['LOGGING']['FORMAT'], datefmt=current_env['LOGGING']['DATE_FORMAT'])
    
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(console_handler)
    
    # 로거 저장
    _loggers[name] = logger
    
    return logger

def setup_werkzeug_logger():
    """Werkzeug 로거 설정"""
    # Werkzeug 로거 설정
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(getattr(logging, current_env['LOGGING']['LEVEL']))
    
    # 기존 핸들러 제거
    for handler in werkzeug_logger.handlers[:]:
        werkzeug_logger.removeHandler(handler)
    
    # 새 핸들러 추가
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, current_env['LOGGING']['LEVEL']))
    
    # 포맷터 설정
    if sys.stdout.isatty():
        formatter = ColoredFormatter(current_env['LOGGING']['FORMAT'], datefmt=current_env['LOGGING']['DATE_FORMAT'])
    else:
        formatter = logging.Formatter(current_env['LOGGING']['FORMAT'], datefmt=current_env['LOGGING']['DATE_FORMAT'])
    
    console_handler.setFormatter(formatter)
    werkzeug_logger.addHandler(console_handler)
    werkzeug_logger.propagate = False