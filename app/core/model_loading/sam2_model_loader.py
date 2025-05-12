"""
SAM 모델 로딩 전담 모듈
"""
from app.utils.model_utils import download_sam_model, get_sam_config_file
from app.utils.logger import setup_logger
from config.settings import MODEL

# 로거 설정
logger = setup_logger(__name__)

# SAM2 시리즈 import
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2ModelLoader:
    """SAM2 시리즈 모델 로딩 전담 클래스"""
    
    @staticmethod
    def download_model_file(model_type):
        """
        모델 파일 다운로드
        
        Args:
            model_type: 다운로드할 모델 타입
            
        Returns:
            str: 다운로드된 모델 파일 경로
        """
        version_name = MODEL.get('VERSION_NAME', MODEL['VERSION'])
        logger.info(f"{version_name} 모델 파일 다운로드 중: {model_type}")
        model_path = download_sam_model(model_type=model_type)
        logger.debug(f"모델 파일 경로: {model_path}")
        return model_path
    
    @staticmethod
    def get_config_file(model_type):
        """
        모델 설정 파일명 가져오기
        
        Args:
            model_type: 모델 타입
            
        Returns:
            str: 설정 파일명
        """
        config_file = get_sam_config_file(model_type)
        logger.debug(f"설정 파일: {config_file}")
        return config_file
    
    @staticmethod
    def build_sam2_model(model_cfg, model_path):
        """
        모델 빌드
        
        Args:
            model_cfg: 모델 설정 파일명
            model_path: 모델 체크포인트 경로
            
        Returns:
            모델 객체
        """
        device = MODEL['DEVICE']
        version_name = MODEL.get('VERSION_NAME', MODEL['VERSION'])
        logger.debug(f"{version_name} 모델 빌드 중... (device: {device})")
        
        try:
            sam = build_sam2(model_cfg, model_path, device=device)
            logger.debug(f"{version_name} 모델 빌드 완료")
            return sam
        except Exception as e:
            logger.error(f"{version_name} 모델 빌드 실패: {str(e)}")
            raise
    
    @staticmethod
    def create_predictor(sam_model):
        """
        Predictor 생성
        
        Args:
            sam_model: 모델 객체
            
        Returns:
            생성된 predictor
        """
        version_name = MODEL.get('VERSION_NAME', MODEL['VERSION'])
        logger.debug(f"{version_name} Predictor 생성 중...")
        
        try:
            predictor = SAM2ImagePredictor(sam_model=sam_model)
            logger.debug(f"{version_name} Predictor 생성 완료")
            return predictor
        except Exception as e:
            logger.error(f"{version_name} Predictor 생성 실패: {str(e)}")
            raise
    
    @staticmethod
    def load_model(model_type):
        """
        전체 모델 로딩 프로세스
        
        Args:
            model_type: 로드할 모델 타입
            
        Returns:
            로드된 predictor
        """
        version_name = MODEL.get('VERSION_NAME', MODEL['VERSION'])
        logger.info(f"{version_name} 모델을 로드합니다... (타입: {model_type})")
        
        try:
            # 1. 모델 파일 다운로드
            model_path = SAM2ModelLoader.download_model_file(model_type)
            
            # 2. 설정 파일 가져오기
            model_cfg = SAM2ModelLoader.get_config_file(model_type)
            
            # 3. SAM 2.1 모델 빌드
            sam = SAM2ModelLoader.build_sam2_model(model_cfg, model_path)
            
            # 4. Predictor 생성
            predictor = SAM2ModelLoader.create_predictor(sam)
            
            logger.info(f"{version_name} 모델 로드 완료 (타입: {model_type})")
            return predictor
            
        except Exception as e:
            logger.error(f"{version_name} 모델 로드 실패: {str(e)}")
            raise