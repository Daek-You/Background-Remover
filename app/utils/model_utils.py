""" 모델 관련 유틸리티 함수들을 관리하는 모듈 """
import os
import requests
import tqdm
from pathlib import Path
from config.settings import MODEL
from app.utils.logger import setup_logger

# 모듈별 로거 설정
logger = setup_logger(__name__)

# 캐시된 모델 경로
_cached_model_path = None

def get_model_dir():
    """모델 디렉토리 경로를 결정하는 함수"""
    # 환경 변수에서 경로 확인
    models_dir = os.getenv('MODELS_DIR')
    if models_dir:
        model_dir = Path(models_dir)
        logger.info(f"환경 변수에서 모델 디렉토리 경로 사용: {model_dir}")
        return model_dir
        
    # Docker 볼륨 경로 확인
    if os.path.exists('/app/models'):
        model_dir = Path('/app/models')
        logger.info("Docker 볼륨 감지됨: /app/models")
        return model_dir
    
    # 프로젝트 루트 기반 경로 찾기
    current_dir = Path(__file__).parent
    while not (current_dir / 'main.py').exists() and current_dir != current_dir.parent:
        current_dir = current_dir.parent
        
    if (current_dir / 'main.py').exists():
        model_dir = current_dir / "models"
        logger.info(f"프로젝트 루트 기반 모델 경로 사용: {model_dir}")
    else:
        # 상대 경로 기반 모델 경로
        base_dir = Path(__file__).parent.parent.parent
        model_dir = base_dir / "models"
        logger.info(f"상대 경로 기반 모델 경로 사용: {model_dir}")
    
    # 디렉토리 생성
    model_dir.mkdir(exist_ok=True)
    return model_dir

def download_model(model_url, model_path):
    """모델 파일을 다운로드하는 함수"""
    logger.info(f"모델 다운로드 중... 이 작업은 몇 분 정도 소요될 수 있습니다.")
    
    try:
        response = requests.get(model_url, stream=True, timeout=30)
        response.raise_for_status()
        
        # 파일 크기 확인 및 진행률 표시 설정
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024
        
        with open(model_path, 'wb') as f:
            with tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc="다운로드 중") as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"모델 다운로드 완료: {model_path}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"모델 다운로드 실패: {str(e)}")
        return False

def download_sam_model(model_type=MODEL['TYPE']):
    """ SAM 모델 파일 다운로드 함수 """
    global _cached_model_path
    
    # 이미 캐시된 경로가 있고, 파일이 존재하면 바로 반환
    if _cached_model_path and os.path.exists(_cached_model_path):
        logger.debug(f"캐시된 모델 경로 사용: {_cached_model_path}")
        return _cached_model_path
    
    # 모델 URL 확인
    if model_type not in MODEL['URLS']:
        raise ValueError(f"지원되지 않는 모델 유형: {model_type}")
    
    model_url = MODEL['URLS'][model_type]
    model_filename = Path(model_url.split("/")[-1])
    
    # 모델 디렉토리 및 파일 경로 설정
    model_dir = get_model_dir()
    model_path = model_dir / model_filename
    
    # 모델 파일이 이미 존재하는지 확인
    if model_path.exists():
        logger.info(f"모델 파일이 이미 존재합니다: {model_path}")
        _cached_model_path = str(model_path)
        return _cached_model_path
    
    # 모델 다운로드
    success = download_model(model_url, model_path)
    if not success:
        raise RuntimeError("모델 다운로드에 실패했습니다.")
    
    # 캐시 업데이트
    _cached_model_path = str(model_path)
    return _cached_model_path
