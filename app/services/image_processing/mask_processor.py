""" 마스크 처리 전담 클래스 """
import numpy as np
from PIL import Image
from app.utils.logger import setup_logger

# 로거 설정
logger = setup_logger(__name__)

class MaskProcessor:
    """마스크 처리 전담 클래스"""
    
    @staticmethod
    def apply_mask_to_image(image: Image.Image, mask: np.ndarray, resize_scale: float = 1.0) -> Image.Image:
        """
        마스크를 이미지에 적용하여 배경이 제거된 결과 반환
        
        Args:
            image: 원본 이미지
            mask: 적용할 마스크
            resize_scale: 리사이즈 스케일 (기본값: 1.0)
            
        Returns:
            Image.Image: 마스크가 적용된 이미지
        """
        # 마스크를 PIL 이미지로 변환
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        
        # 원본 크기로 복원 필요시
        if resize_scale != 1.0:
            mask_img = mask_img.resize(image.size, Image.Resampling.LANCZOS)
            logger.debug(f"마스크 크기 조정: 스케일={resize_scale}")
        
        # 마스크를 원본 이미지에 적용
        result = image.convert('RGBA')
        result.putalpha(mask_img)
        
        logger.debug("마스크 적용 완료")
        return result
    
    @staticmethod
    def refine_mask(mask: np.ndarray, close_kernel_size: int = 5, open_kernel_size: int = 3) -> np.ndarray:
        """
        형태학적 연산을 통한 마스크 정제
        
        Args:
            mask: 정제할 마스크
            close_kernel_size: Closing 커널 크기
            open_kernel_size: Opening 커널 크기
            
        Returns:
            numpy.ndarray: 정제된 마스크
        """
        import cv2
        
        # boolean 마스크를 uint8로 변환
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Closing으로 구멍 메우기
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                               (close_kernel_size, close_kernel_size))
        refined_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, close_kernel)
        
        # Opening으로 노이즈 제거
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                              (open_kernel_size, open_kernel_size))
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, open_kernel)
        
        # boolean 마스크로 변환
        result = refined_mask > 0
        
        logger.debug("마스크 정제 완료")
        return result
    
    @staticmethod
    def validate_mask(mask: np.ndarray, image_shape: tuple) -> bool:
        """
        마스크의 유효성 검증
        
        Args:
            mask: 검증할 마스크
            image_shape: 이미지 크기 (height, width)
            
        Returns:
            bool: 유효하면 True
        """
        if mask is None:
            logger.error("마스크가 None입니다")
            return False
        
        if mask.shape[:2] != image_shape:
            logger.error(f"마스크 크기 불일치: {mask.shape[:2]} != {image_shape}")
            return False
        
        # 마스크에 유효한 영역이 있는지 확인
        if np.sum(mask) == 0:
            logger.warning("마스크에 선택된 영역이 없습니다")
            return False
        
        logger.debug("마스크 유효성 검증 통과")
        return True
    
    @staticmethod
    def create_transparent_image(size: tuple) -> Image.Image:
        """
        투명 이미지 생성
        
        Args:
            size: 이미지 크기 (width, height)
            
        Returns:
            Image.Image: 투명한 RGBA 이미지
        """
        transparent_image = Image.new('RGBA', size, (0, 0, 0, 0))
        logger.debug(f"투명 이미지 생성: {size}")
        return transparent_image