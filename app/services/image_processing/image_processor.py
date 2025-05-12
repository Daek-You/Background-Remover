""" 이미지 처리 서비스 클래스 (기존 코드 호환용) """
from PIL import Image
import numpy as np
from app.services.image_processing.image_resizer import ImageResizer
from app.services.image_processing.image_analyzer import ImageAnalyzer
from app.services.image_processing.mask_processor import MaskProcessor

class ImageProcessor:
    """이미지 처리 서비스 클래스 (기존 코드 호환용)"""
    
    @staticmethod
    def resize_image_if_needed(image: Image.Image) -> tuple[Image.Image, float]:
        """기존 코드 호환용 메서드"""
        return ImageResizer.resize_image_if_needed(image)
    
    @staticmethod
    def analyze_image_for_object(img_np, x, y, radius=None):
        """기존 코드 호환용 메서드"""
        return ImageAnalyzer.analyze_image_for_object(img_np, x, y, radius)
    
    @staticmethod
    def analyze_mask_edge_alignment(mask, img_np):
        """기존 코드 호환용 메서드"""
        return ImageAnalyzer.analyze_mask_edge_alignment(mask, img_np)
    
    @staticmethod
    def apply_mask_to_image(image: Image.Image, mask: np.ndarray, resize_scale: float = 1.0) -> Image.Image:
        """기존 코드 호환용 메서드"""
        return MaskProcessor.apply_mask_to_image(image, mask, resize_scale)
