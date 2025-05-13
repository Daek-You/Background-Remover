""" 마스크 처리 전담 클래스 """
import numpy as np
import cv2
from PIL import Image, ImageFilter
from app.utils.logger import setup_logger
from config.settings import MASK_REFINEMENT
from scipy.ndimage import gaussian_filter, binary_dilation

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
        # 1. 마스크 소프트닝 적용 (설정에 따라)
        if MASK_REFINEMENT.get('SOFTEN_EDGES', True):
            processed_mask = MaskProcessor._soften_mask_edges(mask)
        else:
            processed_mask = mask
        
        # 2. 마스크를 PIL 이미지로 변환
        mask_img = Image.fromarray((processed_mask * 255).astype(np.uint8), mode="L")
        
        # 3. 원본 크기로 복원 필요시
        if resize_scale != 1.0:
            mask_img = mask_img.resize(image.size, Image.Resampling.LANCZOS)
            logger.debug(f"마스크 크기 조정: 스케일={resize_scale}")
        
        # 4. 엣지 스무딩 적용 (설정에 따라)
        if MASK_REFINEMENT.get('EDGE_SMOOTHING', True):
            feather_radius = MASK_REFINEMENT.get('FEATHER_RADIUS', 2)
            mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=feather_radius))
        
        # 5. 마스크를 원본 이미지에 적용
        result = image.convert('RGBA')
        result.putalpha(mask_img)
        
        logger.debug("소프트닝된 마스크 적용 완료")
        return result
    
    @staticmethod
    def _soften_mask_edges(mask: np.ndarray) -> np.ndarray:
        """
        마스크 경계를 부드럽게 처리 (강화된 버전)
        
        Args:
            mask: 원본 마스크
            
        Returns:
            np.ndarray: 소프트닝된 마스크
        """
        # 설정값 가져오기
        gaussian_sigma = MASK_REFINEMENT.get('GAUSSIAN_BLUR_SIGMA', 1.5)
        dilation_size = MASK_REFINEMENT.get('POST_PROCESS_DILATION', 1)
        
        # 1. 마스크를 실수형으로 변환
        soft_mask = mask.astype(np.float32)
        
        # 2. 더 강한 팽창 (하얀 경계 제거용)
        if dilation_size > 0:
            # 더 큰 커널과 더 많은 반복
            kernel = np.ones((5, 5), np.uint8)
            dilated = (soft_mask * 255).astype(np.uint8)
            
            # 팽창을 더 많이 적용
            for _ in range(dilation_size + 1):
                dilated = cv2.dilate(dilated, kernel, iterations=1)
            
            soft_mask = dilated.astype(np.float32) / 255.0
        
        # 3. 추가 전처리 - 에지 검출 기반 마스크 확장
        # 경계 근처의 유사한 색상도 포함시키기
        soft_mask = MaskProcessor._edge_aware_expansion(soft_mask)
        
        # 4. 이중 가우시안 블러로 더 부드러운 경계 생성
        # 첫 번째: 강한 블러로 경계 확산
        soft_mask = gaussian_filter(soft_mask, sigma=gaussian_sigma * 1.5)
        
        # 두 번째: 약한 블러로 자연스러운 그라데이션 생성
        soft_mask = gaussian_filter(soft_mask, sigma=gaussian_sigma * 0.7)
        
        # 5. 삼중 블러로 더욱 부드러운 전환
        soft_mask = gaussian_filter(soft_mask, sigma=gaussian_sigma * 0.3)
        
        # 6. 경계 부근 알파 값 조정 (더 자연스러운 전환)
        # 더 강한 감마 보정 적용
        soft_mask = np.power(soft_mask, 0.6)  # 0.8 → 0.6 (더 강한 효과)
        
        # 7. 값 범위 정규화 (0-1)
        soft_mask = np.clip(soft_mask, 0, 1)
        
        logger.debug(f"강화된 마스크 소프트닝 완료 (sigma={gaussian_sigma}, dilation={dilation_size})")
        return soft_mask
    
    @staticmethod
    def _edge_aware_expansion(mask: np.ndarray) -> np.ndarray:
        """에지 인식 기반 마스크 확장"""
        # 현재 마스크 경계 찾기
        mask_uint8 = (mask * 255).astype(np.uint8)
        edges = cv2.Canny(mask_uint8, 50, 150)
        
        # 경계 근처 확장
        kernel_small = np.ones((3, 3), np.uint8)
        expanded_edges = cv2.dilate(edges, kernel_small, iterations=2)
        
        # 원본 마스크와 결합
        expanded_mask = np.maximum(mask, expanded_edges.astype(np.float32) / 255.0)
        
        return expanded_mask
    
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