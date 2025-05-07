""" 모델 관련 유틸리티 함수들을 관리하는 모듈 """
import os
import requests
import tqdm
from pathlib import Path
from config.settings import MODEL
from utils.logger import setup_logger

# 모듈별 로거 설정
logger = setup_logger(__name__)

def download_sam_model(model_type=MODEL['TYPE']):
    """ SAM 모델 파일 다운로드 함수 """
    model_urls = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
    
    if model_type not in model_urls:
        raise ValueError(f"지원되지 않는 모델 유형: {model_type}")
    
    # 모델 파일 경로 설정 (Docker 볼륨 고려)
    model_filename = Path(model_urls[model_type].split("/")[-1])
    
    # Docker 환경에서는 /app/models를, 로컬 환경에서는 프로젝트 루트의 models 디렉토리를 사용
    if os.path.exists('/app/models'):
        model_dir = Path('/app/models')
        logger.info("Docker 볼륨 환경 감지됨: /app/models")
    else:
        base_dir = Path(__file__).parent.parent
        model_dir = base_dir / "models"
        logger.info(f"로컬 환경 감지됨: {model_dir}")
    
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / model_filename
    
    # 모델 파일이 이미 존재하는지 확인
    if model_path.exists():
        logger.info(f"모델 파일이 이미 존재합니다: {model_path}")
        return str(model_path)
    
    # 모델 다운로드
    logger.info(f"SAM {model_type} 모델 다운로드 중... 이 작업은 몇 분 정도 소요될 수 있습니다.")
    
    response = requests.get(model_urls[model_type], stream=True)
    response.raise_for_status()
    
    # 파일 크기 확인 및 진행률 표시 설정
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024
    
    with open(model_path, 'wb') as f:
        with tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc=f"다운로드 중: {model_type}") as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    logger.info(f"모델 다운로드 완료: {model_path}")
    return str(model_path) 