""" SAM 모델 초기화 모듈 """
import os
from pathlib import Path
from config.settings import MODEL
from app.utils.logger import setup_logger
from app.core.model_util.sam2_model_downloader import SAM2ModelDownloader

# 로거 설정
logger = setup_logger(__name__)

# SAM2 시리즈 import
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2ModelInitializer:
    """SAM 2.1 모델 초기화 및 설정 파일 관리를 담당하는 클래스"""
    
    def __init__(self):
        self.version_name = MODEL.get('VERSION_NAME', MODEL['VERSION'])
        self.downloader = SAM2ModelDownloader()

    def _build_sam2_model(self, model_cfg_name: str, model_path: Path):
        device = MODEL['DEVICE']
        logger.debug(f"{self.version_name} 모델 빌드 중... (device: {device})")
        try:
            # 단순히 설정 파일 이름만 전달 (경로 없이)
            config_name = model_cfg_name.replace('.yaml', '')
            sam = build_sam2(config_name, str(model_path), device=device)
            
            logger.debug(f"{self.version_name} 모델 빌드 완료")
            return sam
        except Exception as e:
            logger.error(f"{self.version_name} 모델 빌드 실패: {str(e)}")
            raise RuntimeError(f"모델 빌드 실패: {str(e)}")
        
        try:
            sam = build_sam2(config_file, str(model_path), device=device)
            logger.debug(f"{self.version_name} 모델 빌드 완료")
            return sam
        except Exception as e:
            logger.error(f"{self.version_name} 모델 빌드 실패: {str(e)}")
            raise RuntimeError(f"모델 빌드 실패: {str(e)}")

    def _create_predictor(self, sam_model):
        """Predictor 생성"""
        if sam_model is None:
            raise ValueError("SAM 모델이 초기화되지 않았습니다")
        logger.debug(f"{self.version_name} Predictor 생성 중...")
        try:
            predictor = SAM2ImagePredictor(sam_model=sam_model)
            logger.debug(f"{self.version_name} Predictor 생성 완료")
            return predictor
        except Exception as e:
            logger.error(f"{self.version_name} Predictor 생성 실패: {str(e)}")
            raise RuntimeError(f"Predictor 생성 실패: {str(e)}")

    def initialize_model(self, model_type=None):
        """전체 모델 초기화 프로세스"""
        if model_type is None:
            model_type = MODEL['TYPE']
        logger.info(f"{self.version_name} 모델을 초기화합니다... (타입: {model_type})")
        try:
            # 1. 모델 파일 다운로드
            model_path = Path(self.downloader.download_model(model_type))
            if not model_path.exists():
                raise RuntimeError(f"모델 파일이 존재하지 않습니다: {model_path}")
            # 2. 설정 파일 이름만 추출
            model_cfg_name = MODEL['CONFIG']['MODEL_TYPE_MAP'][model_type]
            # 3. SAM 2.1 모델 빌드
            sam = self._build_sam2_model(model_cfg_name, model_path)
            # 4. Predictor 생성
            predictor = self._create_predictor(sam)
            logger.info(f"{self.version_name} 모델 초기화 완료 (타입: {model_type})")
            return predictor
        except Exception as e:
            logger.error(f"{self.version_name} 모델 초기화 실패: {str(e)}")
            raise RuntimeError(f"모델 초기화 실패: {str(e)}") 