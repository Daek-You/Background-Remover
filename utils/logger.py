""" 로깅 설정을 관리하는 모듈 """
import logging
import sys
from config.settings import LOGGING

class ColoredFormatter(logging.Formatter):
    """색상이 있는 로그 포맷터"""
    
    def format(self, record):
        # 로그 레벨에 따른 색상 적용
        levelname = record.levelname
        if levelname in LOGGING['COLORS']:
            record.levelname = f"{LOGGING['COLORS'][levelname]}{levelname}{LOGGING['COLORS']['RESET']}"
        return super().format(record)

def setup_logger(name):
    """ 로거 설정 및 반환 """
    logger = logging.getLogger(name)
    
    # 이미 핸들러가 설정되어 있다면 중복 설정 방지
    if logger.handlers:
        return logger
        
    logger.setLevel(LOGGING['LEVEL'])
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOGGING['LEVEL'])
    
    # 포맷터 설정 (터미널이 색상을 지원하는 경우에만 색상 적용)
    if sys.stdout.isatty():
        formatter = ColoredFormatter(LOGGING['FORMAT'], datefmt=LOGGING['DATE_FORMAT'])
    else:
        formatter = logging.Formatter(LOGGING['FORMAT'], datefmt=LOGGING['DATE_FORMAT'])
    
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(console_handler)
    
    return logger