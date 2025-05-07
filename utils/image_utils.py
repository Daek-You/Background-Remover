""" 이미지 처리 관련 유틸리티 함수들을 관리하는 모듈 """
import os
from pathlib import Path

def save_image(img, folder, filename=None):
    """ 이미지를 저장하고 저장된 경로를 반환합니다. """
    if filename is None:
        filename = f"result_{os.urandom(4).hex()}.png"
    
    # 폴더가 없으면 생성
    os.makedirs(folder, exist_ok=True)
    
    # 전체 파일 경로
    file_path = os.path.join(folder, filename)
    
    # 이미지 저장
    img.save(file_path)
    return file_path 