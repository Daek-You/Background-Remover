""" SAM 모델 다운로더 모듈 """
import os
import requests
import tqdm
from pathlib import Path
from config.settings import MODEL
from app.utils.logger import setup_logger

# 로거 설정
logger = setup_logger(__name__)

class SAM2ModelDownloader:
    """SAM 2.1 모델 파일 다운로드 및 검증을 담당하는 클래스"""
    
    def __init__(self):
        self.version_name = MODEL.get('VERSION_NAME', MODEL['VERSION'])
        self._model_dir = None
    
    def _get_model_dir(self):
        """모델 디렉토리 경로를 결정하는 함수"""
        if self._model_dir is not None:
            return self._model_dir
            
        # 1. 환경 변수 확인
        models_dir = os.getenv(MODEL['DIRS']['ENV_VAR'])
        if models_dir and os.path.exists(models_dir):
            self._model_dir = Path(models_dir)
            logger.info(f"환경 변수에서 모델 디렉토리 경로 사용: {self._model_dir}")
            return self._model_dir
            
        # 2. Docker 볼륨 경로 확인
        docker_volume = MODEL['DIRS']['DOCKER_VOLUME']
        if os.path.exists(docker_volume):
            self._model_dir = Path(docker_volume)
            logger.info(f"Docker 볼륨 감지됨: {docker_volume}")
            return self._model_dir
        
        # 3. 상대 경로 기반 모델 경로 사용
        try:
            current_dir = Path(__file__).parent
            # app/core/model_util -> root/models
            self._model_dir = current_dir.parent.parent.parent / MODEL['DIRS']['DEFAULT_SUBDIR']
            self._model_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"상대 경로 기반 모델 경로 사용: {self._model_dir}")
            
            # 경로 검증
            if not self._model_dir.exists() or not os.access(self._model_dir, os.W_OK):
                raise RuntimeError(f"모델 디렉토리를 생성하거나 접근할 수 없습니다: {self._model_dir}")
                
            return self._model_dir
            
        except Exception as e:
            logger.error(f"모델 디렉토리 생성 실패: {str(e)}")
            raise RuntimeError(f"모델 디렉토리를 생성할 수 없습니다: {str(e)}")
    
    def _download_file(self, url, save_path, description="파일"):
        """파일 다운로드 함수"""
        max_retries = MODEL['DOWNLOAD']['MAX_RETRIES']
        timeout = MODEL['DOWNLOAD']['TIMEOUT']
        block_size = MODEL['DOWNLOAD']['BLOCK_SIZE']
        retry_delay = MODEL['DOWNLOAD']['RETRY_DELAY']
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, stream=True, timeout=timeout)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                temp_path = save_path.with_suffix('.tmp')
                
                with open(temp_path, 'wb') as f:
                    with tqdm.tqdm(
                        total=total_size, 
                        unit='B', 
                        unit_scale=True, 
                        desc=f"{description} 다운로드 중 - 시도 {attempt + 1}/{max_retries}"
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=block_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                temp_path.rename(save_path)
                logger.info(f"{description} 다운로드 완료: {save_path}")
                return True
                
            except (requests.exceptions.RequestException, IOError) as e:
                logger.warning(f"다운로드 시도 {attempt + 1} 실패: {str(e)}")
                
                if temp_path.exists():
                    temp_path.unlink()
                
                if attempt < max_retries - 1:
                    logger.info(f"{retry_delay}초 후 재시도합니다...")
                    import time
                    time.sleep(retry_delay)
                else:
                    logger.error(f"{description} 다운로드 최종 실패: {str(e)}")
                    return False
        
        return False
    
    def _verify_file_size(self, file_path, min_size_mb, description="파일"):
        """파일 크기 검증"""
        try:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            if file_size_mb < min_size_mb:
                logger.error(f"{description} 크기가 예상보다 작습니다: {file_size_mb:.1f}MB < {min_size_mb}MB")
                return False
            
            logger.info(f"{description} 크기 검증 완료: {file_size_mb:.1f}MB")
            return True
            
        except Exception as e:
            logger.error(f"{description} 크기 검증 중 오류: {str(e)}")
            return False
    
    def download_model(self, model_type):
        """모델 파일 다운로드 및 검증"""
        logger.info(f"{self.version_name} 모델 파일 다운로드 중: {model_type}")
        
        # SAM URL 확인
        if model_type not in MODEL['URLS']:
            raise ValueError(f"지원되지 않는 {self.version_name} 모델 타입: {model_type}")
        
        model_url = MODEL['URLS'][model_type]
        model_filename = Path(model_url.split("/")[-1])
        
        # 모델 디렉토리 및 파일 경로 설정
        model_dir = self._get_model_dir()
        model_path = model_dir / model_filename
        
        # 모델 파일이 이미 존재하는지 확인
        if model_path.exists():
            # 파일 무결성 간단 확인 (크기 체크)
            min_size = MODEL['MIN_SIZES'].get(model_type, 50)  # 기본값 50MB
            if self._verify_file_size(model_path, min_size, f"{self.version_name} 모델"):
                logger.info(f"{self.version_name} 모델 파일이 이미 존재합니다: {model_path}")
                return str(model_path)
            else:
                logger.warning(f"기존 모델 파일이 손상되었습니다. 재다운로드합니다.")
                model_path.unlink()  # 손상된 파일 삭제
        
        # 모델 다운로드
        if not self._download_file(model_url, model_path, f"{self.version_name} 모델"):
            raise RuntimeError(f"{self.version_name} 모델 다운로드에 실패했습니다: {model_type}")
        
        # 다운로드 후 무결성 검증
        min_size = MODEL['MIN_SIZES'].get(model_type, 50)
        if not self._verify_file_size(model_path, min_size, f"{self.version_name} 모델"):
            logger.error(f"다운로드된 모델 파일 검증 실패: {model_path}")
            model_path.unlink()  # 손상된 파일 삭제
            raise RuntimeError(f"{self.version_name} 모델 파일 검증 실패: {model_type}")
        
        return str(model_path) 