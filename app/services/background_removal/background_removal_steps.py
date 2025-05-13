""" 배경 제거 과정의 단계별 로직 관리 """
import numpy as np
from typing import Tuple, List, Dict, Any
from app.utils.logger import setup_logger
from app.core.model_management.model_manager import get_model_manager
from app.services.image_processing.image_processor import ImageProcessor
from app.services.mask_selection.mask_selector import MaskSelector
from config.settings import MODEL, BACKGROUND_REMOVAL

logger = setup_logger(__name__)

class BackgroundRemovalSteps:
    """배경 제거 과정의 단계별 로직을 관리하는 클래스"""
    
    # 클래스 레벨에서 단계 수 관리
    TOTAL_STEPS = BACKGROUND_REMOVAL['TOTAL_STEPS']
    
    @staticmethod
    def _log_step(image_id: str, step: int, description: str):
        """단계별 로그를 일관된 형식으로 출력"""
        logger.info(f"[{image_id}] {step}/{BackgroundRemovalSteps.TOTAL_STEPS}: {description}")
    
    @staticmethod
    def preprocess_image(
        image, 
        click_points: List[Tuple[int, int]], 
        image_id: str
    ) -> Tuple[np.ndarray, List[Tuple[int, int]], float]:
        """1단계: 이미지 전처리"""
        BackgroundRemovalSteps._log_step(image_id, 1, "이미지 전처리")
        
        # 크기 조정 및 좌표 스케일 조정
        resized_image, scale = ImageProcessor.resize_image_if_needed(image)
        img_np = np.array(resized_image.convert("RGB"))
        
        # 모든 좌표 변환
        h, w = img_np.shape[:2]
        scaled_points = []
        
        for i, (x, y) in enumerate(click_points):
            # 좌표 스케일 조정
            scaled_x = int(x * scale)
            scaled_y = int(y * scale)
            
            # 좌표 범위 검증
            scaled_x = max(0, min(scaled_x, w - 1))
            scaled_y = max(0, min(scaled_y, h - 1))
            
            scaled_points.append((scaled_x, scaled_y))
            logger.debug(f"[{image_id}] 좌표 #{i+1} 변환: ({x}, {y}) -> ({scaled_x}, {scaled_y})")
        
        logger.debug(f"[{image_id}] 처리 크기: {img_np.shape[:2]}, 스케일: {scale:.2f}")
        return img_np, scaled_points, scale
    
    @staticmethod
    def analyze_image(
        img_np: np.ndarray, 
        x: int, 
        y: int, 
        image_id: str
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """2단계: 이미지 분석"""
        BackgroundRemovalSteps._log_step(image_id, 2, "이미지 분석")
        
        # 기본 분석
        is_near_edge, edge_strength = ImageProcessor.analyze_image_for_object(img_np, x, y)
        
        # SAM 2.1용 추가 분석 (확장 가능)
        context_info = {}
        if hasattr(ImageProcessor, 'analyze_image_context'):
            context_info = ImageProcessor.analyze_image_context(img_np, x, y)
        
        logger.debug(f"[{image_id}] 분석 결과 - 에지 근처: {is_near_edge}, 강도: {edge_strength:.3f}")
        return is_near_edge, edge_strength, context_info
    
    @staticmethod
    def predict_masks(
        img_np: np.ndarray, 
        points: List[Tuple[int, int]], 
        image_id: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """3단계: 모델 예측"""
        BackgroundRemovalSteps._log_step(image_id, 3, "모델 예측")
        
        # 모델 관리자 가져오기
        model_manager = get_model_manager()
        
        # 입력 포인트 설정 (모든 포인트를 positive로 설정)
        input_points = np.array(points)
        input_labels = np.ones(len(points))
        
        logger.debug(f"[{image_id}] 예측 포인트 수: {len(points)}")
        
        # SAM 2.1의 negative points는 모델 관리자에서 자동 처리
        masks, scores, logits = model_manager.predict(img_np, input_points, input_labels)
        
        logger.debug(f"[{image_id}] 예측 완료 - 마스크 {len(masks)}개 생성")
        return masks, scores, logits
    
    @staticmethod
    def select_best_mask(
        masks: np.ndarray,
        scores: np.ndarray,
        img_np: np.ndarray,
        x: int, y: int,
        is_near_edge: bool,
        edge_strength: float,
        context_info: Dict[str, Any],
        image_id: str
    ) -> np.ndarray:
        """4단계: 마스크 선택"""
        BackgroundRemovalSteps._log_step(image_id, 4, "마스크 선택")
        
        # 기존 마스크 선택 로직 사용
        mask_idx = MaskSelector.select_best_mask(
            masks, scores, img_np, x, y, is_near_edge, edge_strength
        )
        selected_mask = masks[mask_idx]
        
        logger.debug(f"[{image_id}] 선택된 마스크: {mask_idx}번 (점수: {scores[mask_idx]:.3f})")
        return selected_mask
    
    @staticmethod
    def refine_and_apply_mask(
        image, 
        mask: np.ndarray, 
        scale: float, 
        image_id: str
    ) -> Any:
        """5단계: 마스크 정제 및 적용"""
        BackgroundRemovalSteps._log_step(image_id, 5, "마스크 정제 및 적용")
        
        # 마스크 정제
        refined_mask = BackgroundRemovalSteps._refine_mask(mask, image_id)
        
        # 마스크 적용
        result = ImageProcessor.apply_mask_to_image(image, refined_mask, scale)
        
        logger.debug(f"[{image_id}] 마스크 적용 완료")
        return result
    
    @staticmethod
    def _refine_mask(mask: np.ndarray, image_id: str) -> np.ndarray:
        """마스크 정제 (내부 함수)"""
        close_size = BACKGROUND_REMOVAL['MASK_REFINEMENT']['CLOSE_KERNEL_SIZE']
        open_size = BACKGROUND_REMOVAL['MASK_REFINEMENT']['OPEN_KERNEL_SIZE']
        
        # 마스크를 boolean으로 변환 (threshold 적용)
        threshold = BACKGROUND_REMOVAL['MASK_REFINEMENT']['THRESHOLD']
        mask_binary = mask > threshold
        
        # 형태학적 연산으로 정제
        from app.services.image_processing.mask_processor import MaskProcessor
        refined_mask = MaskProcessor.refine_mask(
            mask_binary,
            close_kernel_size=close_size,
            open_kernel_size=open_size
        )
        
        logger.debug(f"[{image_id}] 마스크 정제 - 커널 크기 close:{close_size}, open:{open_size}")
        return refined_mask
    
    # === 하위 호환성을 위한 래퍼 함수들 ===
    
    @staticmethod
    def refine_selected_mask(mask: np.ndarray, image_id: str) -> np.ndarray:
        """하위 호환성을 위한 래퍼 함수"""
        return BackgroundRemovalSteps._refine_mask(mask, image_id)
    
    @staticmethod
    def apply_mask_to_image(image, mask: np.ndarray, scale: float, image_id: str):
        """하위 호환성을 위한 래퍼 함수"""
        logger.debug(f"[{image_id}] 마스크 적용 시작")
        result = ImageProcessor.apply_mask_to_image(image, mask, scale)
        logger.debug(f"[{image_id}] 마스크 적용 완료")
        return result