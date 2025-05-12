""" 이미지 크기 조정 전담 클래스 """
from PIL import Image
from app.utils.logger import setup_logger
from config.settings import MAX_IMAGE_SIZE

# 로거 설정
logger = setup_logger(__name__)

class ImageResizer:
    """이미지 크기 조정 전담 클래스"""
    
    @staticmethod
    def resize_image_if_needed(image: Image.Image) -> tuple[Image.Image, float]:
        """
        이미지가 너무 큰 경우 리사이징하고 스케일 비율 반환
        
        Args:
            image: 원본 이미지
            
        Returns:
            tuple: (리사이즈된_이미지, 스케일_비율)
        """
        max_size = MAX_IMAGE_SIZE
        
        if max(image.size) > max_size:
            scale = max_size / max(image.size)
            new_size = tuple(int(dim * scale) for dim in image.size)
            resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"이미지 리사이즈: {image.size} -> {new_size} (스케일: {scale:.2f})")
            return resized_image, scale
        
        logger.debug("이미지 리사이즈 불필요")
        return image, 1.0
    
    @staticmethod
    def get_resized_coordinates(x, y, scale):
        """
        좌표를 스케일에 맞게 조정
        
        Args:
            x: 원본 x 좌표
            y: 원본 y 좌표
            scale: 스케일 비율
            
        Returns:
            tuple: (조정된_x, 조정된_y)
        """
        return int(x * scale), int(y * scale)
    
    @staticmethod
    def validate_image_size(image: Image.Image) -> bool:
        """
        이미지 크기가 유효한지 확인
        
        Args:
            image: 확인할 이미지
            
        Returns:
            bool: 유효하면 True
        """
        if not image or image.size[0] <= 0 or image.size[1] <= 0:
            logger.error("유효하지 않은 이미지 크기")
            return False
        
        # 최대 크기 제한 확인
        if max(image.size) > MAX_IMAGE_SIZE * 10:  # 10배 이상이면 너무 큼
            logger.warning(f"이미지가 매우 큽니다: {image.size}")
        
        return True