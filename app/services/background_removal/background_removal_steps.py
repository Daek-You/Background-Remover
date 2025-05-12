"""
배경 제거 과정의 단계별 로직 관리
"""
import numpy as np
from app.utils.logger import setup_logger
from app.core.model_manager import get_model_manager
from app.services.image_processor import ImageProcessor
from app.services.mask_selector import MaskSelector
from config.settings import MODEL

# 로거 설정
logger = setup_logger(__name__)

class BackgroundRemovalSteps:
    """배경 제거 과정의 단계별 로직 관리 클래스"""
    
    @staticmethod
    def log_step(image_id, current_step, total_steps, description):
        """
        단계별 로그 출력
        
        Args:
            image_id: 이미지 식별자
            current_step: 현재 단계
            total_steps: 전체 단계 수
            description: 단계 설명
        """
        logger.info(f"[{image_id}] Step {current_step}/{total_steps}: {description}")
    
    @staticmethod
    def preprocess_image(image, x, y, image_id):
        """
        1단계: 이미지 전처리
        
        Args:
            image: 원본 이미지
            x: 클릭한 x 좌표
            y: 클릭한 y 좌표
            image_id: 이미지 식별자
            
        Returns:
            tuple: (전처리된_이미지_배열, 조정된_x, 조정된_y, 스케일)
        """
        BackgroundRemovalSteps.log_step(image_id, 1, 5, "이미지 전처리 시작")
        
        # 크기 조정 및 좌표 스케일 조정
        resized_image, scale = ImageProcessor.resize_image_if_needed(image)
        img_np = np.array(resized_image.convert("RGB"))
        
        scaled_x = int(x * scale)
        scaled_y = int(y * scale)
        
        BackgroundRemovalSteps.log_step(image_id, 1, 5, "이미지 전처리 완료")
        return img_np, scaled_x, scaled_y, scale
    
    @staticmethod
    def analyze_image(img_np, x, y, image_id):
        """
        2단계: 이미지 분석
        
        Args:
            img_np: 이미지 numpy 배열
            x: 클릭한 x 좌표
            y: 클릭한 y 좌표
            image_id: 이미지 식별자
            
        Returns:
            tuple: (에지_근접_여부, 에지_강도, 컨텍스트_정보)
        """
        BackgroundRemovalSteps.log_step(image_id, 2, 5, "이미지 분석 시작")
        
        # 기본 분석
        is_near_edge, edge_strength = ImageProcessor.analyze_image_for_object(img_np, x, y)
        
        # SAM 2.1용 추가 분석 (향후 확장 가능)
        context_info = {}
        if hasattr(ImageProcessor, 'analyze_image_context'):
            context_info = ImageProcessor.analyze_image_context(img_np, x, y)
        
        BackgroundRemovalSteps.log_step(image_id, 2, 5, "이미지 분석 완료")
        return is_near_edge, edge_strength, context_info
    
    @staticmethod
    def predict_masks(img_np, x, y, image_id):
        """
        3단계: 모델 예측
        
        Args:
            img_np: 이미지 numpy 배열
            x: 클릭한 x 좌표
            y: 클릭한 y 좌표
            image_id: 이미지 식별자
            
        Returns:
            tuple: (마스크들, 점수들, 로짓들)
        """
        BackgroundRemovalSteps.log_step(image_id, 3, 5, "모델 예측 시작")
        
        # 모델 관리자 가져오기
        model_manager = get_model_manager()
        
        # 입력 포인트 설정
        input_point = np.array([[x, y]])
        input_label = np.array([1])
        
        # SAM 2.1의 negative points는 모델 관리자에서 자동 처리
        masks, scores, logits = model_manager.predict(img_np, input_point, input_label)
        
        BackgroundRemovalSteps.log_step(image_id, 3, 5, "모델 예측 완료")
        return masks, scores, logits
    
    @staticmethod
    def select_best_mask(masks, scores, img_np, x, y, is_near_edge, edge_strength, context_info, image_id):
        """
        4단계: 마스크 선택
        
        Args:
            masks: 예측된 마스크들
            scores: 마스크 점수들
            img_np: 이미지 numpy 배열
            x: 클릭한 x 좌표
            y: 클릭한 y 좌표
            is_near_edge: 에지 근접 여부
            edge_strength: 에지 강도
            context_info: 컨텍스트 정보
            image_id: 이미지 식별자
            
        Returns:
            numpy.ndarray: 선택된 마스크
        """
        BackgroundRemovalSteps.log_step(image_id, 4, 5, "마스크 선택 시작")
        
        # 기존 마스크 선택 로직 사용
        mask_idx = MaskSelector.select_best_mask(
            masks, scores, img_np, x, y, is_near_edge, edge_strength
        )
        selected_mask = masks[mask_idx]
        
        BackgroundRemovalSteps.log_step(image_id, 4, 5, "마스크 선택 완료")
        return selected_mask
    
    @staticmethod
    def apply_mask_to_image(image, mask, scale, image_id):
        """
        5단계: 마스크 적용
        
        Args:
            image: 원본 이미지
            mask: 적용할 마스크
            scale: 이미지 스케일
            image_id: 이미지 식별자
            
        Returns:
            PIL.Image: 마스크가 적용된 이미지
        """
        BackgroundRemovalSteps.log_step(image_id, 5, 5, "마스크 적용 시작")
        
        # 마스크 적용
        result = ImageProcessor.apply_mask_to_image(image, mask, scale)
        
        BackgroundRemovalSteps.log_step(image_id, 5, 5, "마스크 적용 완료")
        return result