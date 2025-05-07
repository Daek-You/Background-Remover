""" 이미지 처리 관련 유틸리티 함수들을 관리하는 모듈 """
import os
import time
import glob
from utils.logger import setup_logger

# 로거 설정
logger = setup_logger(__name__)

def save_image(img, folder, filename=None):
    """ 이미지를 저장하고 저장된 경로를 반환합니다. """
    if filename is None:
        # 타임스탬프를 포함하여 더 안전한 고유 파일명 생성
        timestamp = int(time.time())
        filename = f"result_{timestamp}_{os.urandom(4).hex()}.png"
    
    # 폴더가 없으면 생성
    os.makedirs(folder, exist_ok=True)
    
    # 전체 파일 경로
    file_path = os.path.join(folder, filename)
    
    # 이미지 저장
    img.save(file_path)
    return file_path

def cleanup_old_files(folder, pattern="result_*.png", max_age_hours=24):
    """오래된 임시 파일들을 정리합니다."""
    try:
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        # 패턴에 맞는 파일 목록
        file_list = glob.glob(os.path.join(folder, pattern))
        
        removed_count = 0
        for file_path in file_list:
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                os.remove(file_path)
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"{removed_count}개의 오래된 파일이 정리되었습니다.")
            
        return removed_count
    except Exception as e:
        logger.error(f"파일 정리 중 오류 발생: {str(e)}")
        return 0 