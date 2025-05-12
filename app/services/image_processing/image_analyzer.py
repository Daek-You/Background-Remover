""" 이미지 분석 전담 클래스 """
import numpy as np
import cv2
from app.utils.logger import setup_logger
from config.settings import IMAGE_ANALYSIS

# 로거 설정
logger = setup_logger(__name__)

class ImageAnalyzer:
    """이미지 분석 전담 클래스"""
    
    @staticmethod
    def analyze_image_for_object(img_np, x, y, radius=None):
        """
        클릭 위치 주변을 분석하여 객체 특성 파악
        
        Args:
            img_np: 이미지 numpy 배열
            x: 클릭한 x 좌표
            y: 클릭한 y 좌표
            radius: 분석 반경 (기본값: IMAGE_ANALYSIS['CLICK_RADIUS'])
            
        Returns:
            tuple: (에지_근접_여부, 에지_강도)
        """
        if radius is None:
            radius = IMAGE_ANALYSIS['CLICK_RADIUS']
        
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
        edges = cv2.Canny(gray, 
                         IMAGE_ANALYSIS['CANNY_THRESHOLD_LOW'], 
                         IMAGE_ANALYSIS['CANNY_THRESHOLD_HIGH'])
        
        # 클릭 위치 주변 에지 확인
        center_y, center_x = radius, radius
        edge_radius = IMAGE_ANALYSIS['EDGE_CHECK_RADIUS']
        nearby_region = edges[max(0, center_y-edge_radius):min(2*radius, center_y+edge_radius), 
                             max(0, center_x-edge_radius):min(2*radius, center_x+edge_radius)]
        
        # 에지 존재 여부 및 강도
        is_near_edge = np.any(nearby_region > 0)
        edge_strength = np.sum(nearby_region) / (nearby_region.size * 255) if nearby_region.size > 0 else 0
        
        logger.debug(f"에지 분석 결과: 근접={is_near_edge}, 강도={edge_strength:.3f}")
        return is_near_edge, edge_strength
    
    @staticmethod
    def analyze_mask_edge_alignment(mask, img_np):
        """
        마스크가 이미지의 에지와 얼마나 잘 일치하는지 분석
        
        Args:
            mask: 분석할 마스크
            img_np: 이미지 numpy 배열
            
        Returns:
            float: 에지 정렬 점수
        """
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
        
        logger.debug(f"에지 정렬 점수: {edge_alignment_score:.3f}")
        return edge_alignment_score
    
    @staticmethod
    def detect_edges(image_np, low_threshold=None, high_threshold=None):
        """
        이미지에서 에지 검출
        
        Args:
            image_np: 이미지 numpy 배열
            low_threshold: Canny 하한 임계값
            high_threshold: Canny 상한 임계값
            
        Returns:
            numpy.ndarray: 에지 이미지
        """
        if low_threshold is None:
            low_threshold = IMAGE_ANALYSIS['CANNY_THRESHOLD_LOW']
        if high_threshold is None:
            high_threshold = IMAGE_ANALYSIS['CANNY_THRESHOLD_HIGH']
        
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        logger.debug(f"에지 검출 완료: {np.sum(edges > 0)}개 에지 픽셀")
        return edges