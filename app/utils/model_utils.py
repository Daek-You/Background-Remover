import os
import requests
import tqdm
from pathlib import Path
from config.settings import MODEL
from app.utils.logger import setup_logger

# 모듈별 로거 설정
logger = setup_logger(__name__)
# 캐시된 모델 경로 (메모리 절약을 위해 단일 모델만 캐시)
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
    
    # 동적 서브디렉토리 생성 (settings에서 가져옴)
    model_dir = model_dir / MODEL['SUB_DIRECTORY_NAME']
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir

def download_model(model_url, model_path):
    """모델 파일을 다운로드하는 함수"""
    version_name = MODEL.get('VERSION_NAME', MODEL['VERSION'])
    logger.info(f"{version_name} 모델 다운로드 중... 이 작업은 몇 분 정도 소요될 수 있습니다.")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(model_url, stream=True, timeout=30)
            response.raise_for_status()
            
            # 파일 크기 확인 및 진행률 표시 설정
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1MB 블록
            
            # 임시 파일로 다운로드 후 원자적 이동 (중단 시 안전성 보장)
            temp_path = model_path.with_suffix('.tmp')
            
            with open(temp_path, 'wb') as f:
                with tqdm.tqdm(
                    total=total_size, 
                    unit='B', 
                    unit_scale=True, 
                    desc=f"다운로드 중 ({model_path.name}) - 시도 {attempt + 1}/{max_retries}"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # 다운로드 완료 후 임시 파일을 최종 파일로 이동
            temp_path.rename(model_path)
            
            logger.info(f"{version_name} 모델 다운로드 완료: {model_path}")
            return True
            
        except (requests.exceptions.RequestException, IOError) as e:
            logger.warning(f"다운로드 시도 {attempt + 1} 실패: {str(e)}")
            
            # 실패한 임시 파일 정리
            if temp_path.exists():
                temp_path.unlink()
            
            # 마지막 시도가 아니면 재시도
            if attempt < max_retries - 1:
                logger.info(f"5초 후 재시도합니다...")
                import time
                time.sleep(5)
            else:
                logger.error(f"{version_name} 모델 다운로드 최종 실패: {str(e)}")
                return False
    
    return False

def get_sam_config_file(model_type):
    """SAM 모델 타입에 따른 설정 파일명 반환 (SAM 2 전용)"""
    # CONFIG_MAP이 있는 경우에만 처리 (SAM2에서만 필요)
    if 'CONFIG_MAP' in MODEL and MODEL['CONFIG_MAP']:
        return MODEL['CONFIG_MAP'].get(model_type, f"{model_type}.yaml")
    else:
        # SAM1의 경우 설정 파일이 필요 없음
        return None

def download_sam_model(model_type=None):
    """SAM 모델 파일 다운로드 함수"""
    global _cached_model_path
    
    # 기본값 설정
    if model_type is None:
        model_type = MODEL['TYPE']
    
    # 동적으로 버전 정보 가져오기
    version_name = MODEL.get('VERSION_NAME', MODEL['VERSION'])
    
    # 이미 캐시된 경로가 있고, 파일이 존재하면 바로 반환
    if _cached_model_path and os.path.exists(_cached_model_path):
        logger.debug(f"캐시된 {version_name} 모델 경로 사용: {_cached_model_path}")
        return _cached_model_path
    
    # SAM URL 확인
    if model_type not in MODEL['URLS']:
        raise ValueError(f"지원되지 않는 {version_name} 모델 타입: {model_type}")
    
    model_url = MODEL['URLS'][model_type]
    model_filename = Path(model_url.split("/")[-1])
    
    # 모델 디렉토리 및 파일 경로 설정
    model_dir = get_model_dir()
    model_path = model_dir / model_filename
    
    # 모델 파일이 이미 존재하는지 확인
    if model_path.exists():
        # 파일 무결성 간단 확인 (크기 체크)
        if _verify_model_file(model_path, model_type):
            logger.info(f"{version_name} 모델 파일이 이미 존재합니다: {model_path}")
            _cached_model_path = str(model_path)
            return _cached_model_path
        else:
            logger.warning(f"기존 모델 파일이 손상되었습니다. 재다운로드합니다.")
            model_path.unlink()  # 손상된 파일 삭제
    
    # 모델 다운로드
    success = download_model(model_url, model_path)
    if not success:
        raise RuntimeError(f"{version_name} 모델 다운로드에 실패했습니다: {model_type}")
    
    # 다운로드 후 무결성 검증
    if not _verify_model_file(model_path, model_type):
        logger.error(f"다운로드된 모델 파일 검증 실패: {model_path}")
        model_path.unlink()  # 손상된 파일 삭제
        raise RuntimeError(f"{version_name} 모델 파일 검증 실패: {model_type}")
    
    # 캐시 업데이트
    _cached_model_path = str(model_path)
    return _cached_model_path

def _verify_model_file(model_path, model_type):
    """모델 파일의 기본적인 무결성 검증"""
    try:
        # 파일 크기 확인 (MB)
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        
        # 동적으로 최소 크기 정보 가져오기
        min_sizes = MODEL.get('MIN_SIZES', {})
        min_size = min_sizes.get(model_type, 50)  # 기본값 50MB
        
        if file_size_mb < min_size:
            logger.error(f"모델 파일 크기가 예상보다 작습니다: {file_size_mb:.1f}MB < {min_size}MB")
            return False
        
        # 동적으로 버전 정보 가져오기
        version_name = MODEL.get('VERSION_NAME', MODEL['VERSION'])
        logger.info(f"{version_name} 모델 파일 검증 완료: {file_size_mb:.1f}MB")
        return True
        
    except Exception as e:
        logger.error(f"모델 파일 검증 중 오류: {str(e)}")
        return False