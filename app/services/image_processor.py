"""
이미지 처리 관련 기능을 제공하는 서비스 모듈
"""
import numpy as np
import cv2
from PIL import Image
from app.utils.logger import setup_logger
from config.settings import IMAGE_ANALYSIS, MAX_IMAGE_SIZE

# 로거 설정
logger = setup_logger(__name__)

class ImageProcessor:
    """이미지 처리 서비스"""
    
    @staticmethod
    def resize_image_if_needed(image: Image.Image) -> tuple[Image.Image, float]:
        """이미지가 너무 큰 경우 리사이징하고 스케일 비율 반환"""
        if max(image.size) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(image.size)
            new_size = tuple(int(dim * scale) for dim in image.size)
            return image.resize(new_size, Image.Resampling.LANCZOS), scale
        return image, 1.0
    
    @staticmethod
    def analyze_image_for_object(img_np, x, y, radius=IMAGE_ANALYSIS['CLICK_RADIUS']):
        """클릭 위치 주변을 분석하여 객체 특성 파악"""
        # 클릭 주변 영역 추출 (반경 내의 픽셀 분석)
        h, w = img_np.shape[:2]
        x_min = max(0, x - radius)
        y_min = max(0, y - radius)
        x_max = min(w, x + radius)
        y_max = min(h, y + radius)
        
        # 관심 영역 추출
        roi = img_np[y_min:y_max, x_min:x_max]
        
        # 간단한 에지 검출로 객체 경계 강화
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, IMAGE_ANALYSIS['CANNY_THRESHOLD_LOW'], IMAGE_ANALYSIS['CANNY_THRESHOLD_HIGH'])
        
        # 클릭 위치 주변 에지 확인
        center_y, center_x = radius, radius
        edge_radius = IMAGE_ANALYSIS['EDGE_CHECK_RADIUS']
        nearby_region = edges[max(0, center_y-edge_radius):min(2*radius, center_y+edge_radius), 
                             max(0, center_x-edge_radius):min(2*radius, center_x+edge_radius)]
        
        # 에지 존재 여부 및 강도
        is_near_edge = np.any(nearby_region > 0)
        edge_strength = np.sum(nearby_region) / (nearby_region.size * 255) if nearby_region.size > 0 else 0
        
        return is_near_edge, edge_strength
    
    @staticmethod
    def analyze_mask_edge_alignment(mask, img_np):
        """마스크가 이미지의 에지와 얼마나 잘 일치하는지 분석"""
        # 마스크 경계 추출
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 마스크 경계 주변만 에지 검출
        mask_edge = np.zeros_like(img_np[:,:,0])
        cv2.drawContours(mask_edge, mask_contours, -1, 255, 1)
        
        # 마스크 경계 주변 영역만 추출
        kernel = np.ones((3,3), np.uint8)
        mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=2)
        roi = np.where(mask_dilated > 0)
        
        if len(roi[0]) == 0:
            return 0.0
        
        # 관심 영역만 에지 검출
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 
                         IMAGE_ANALYSIS['CANNY_THRESHOLD_LOW'],
                         IMAGE_ANALYSIS['CANNY_THRESHOLD_HIGH'])
        
        # 마스크 경계와 에지의 겹치는 정도를 관심 영역 부분만 계산
        overlap = np.logical_and(edges[roi] > 0, mask_edge[roi] > 0)
        edge_alignment_score = np.sum(overlap) / np.sum(mask_edge[roi] > 0)
        
        return edge_alignment_score
    
    @staticmethod
    def apply_mask_to_image(image: Image.Image, mask: np.ndarray, resize_scale: float = 1.0) -> Image.Image:
        """마스크를 이미지에 적용하여 배경이 제거된 결과 반환"""
        # 마스크를 PIL 이미지로 변환
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        
        # 원본 크기로 복원 필요시
        if resize_scale != 1.0:
            mask_img = mask_img.resize(image.size, Image.Resampling.LANCZOS)
        
        # 마스크를 원본 이미지에 적용
        result = image.convert('RGBA')
        result.putalpha(mask_img)
        
        return result 