"""
배경 제거 서비스 메인 클래스
"""
import os
from PIL import Image
from app.utils.logger import setup_logger
from app.core.model_manager import get_model_manager
from app.services.background_removal_steps import BackgroundRemovalSteps
from config.settings import MODEL

# 로거 설정
logger = setup_logger(__name__)

class BackgroundRemover:
    """배경 제거 서비스 메인 클래스"""
    
    @staticmethod
    def remove_background(image: Image.Image, x: int, y: int, image_name: str) -> Image.Image:
        """
        클릭한 위치의 객체 배경을 제거하는 함수
        
        Args:
            image: 원본 이미지
            x: 클릭한 x 좌표
            y: 클릭한 y 좌표
            image_name: 이미지 이름
            
        Returns:
            Image.Image: 배경이 제거된 이미지
        """
        image_id = os.path.basename(image_name)
        version_name = MODEL.get('VERSION_NAME', MODEL['VERSION'])
        
        try:
            logger.info(f"[{image_id}] {version_name} 배경 제거 작업 시작")
            
            # 1. 이미지 전처리
            img_np, scaled_x, scaled_y, scale = BackgroundRemovalSteps.preprocess_image(
                image, x, y, image_id
            )
            
            # 2. 이미지 분석
            is_near_edge, edge_strength, context_info = BackgroundRemovalSteps.analyze_image(
                img_np, scaled_x, scaled_y, image_id
            )
            
            # 3. 모델 예측
            masks, scores, logits = BackgroundRemovalSteps.predict_masks(
                img_np, scaled_x, scaled_y, image_id
            )
            
            # 4. 마스크 선택
            selected_mask = BackgroundRemovalSteps.select_best_mask(
                masks, scores, img_np, scaled_x, scaled_y, 
                is_near_edge, edge_strength, context_info, image_id
            )
            
            # 5. 마스크 적용
            result = BackgroundRemovalSteps.apply_mask_to_image(
                image, selected_mask, scale, image_id
            )
            
            # GPU 메모리 정리 (대용량 이미지 처리 후)
            if max(image.size) > 1000:
                model_manager = get_model_manager()
                model_manager.cleanup()
                logger.debug(f"[{image_id}] 대용량 이미지 처리 후 GPU 메모리 정리")
            
            logger.info(f"[{image_id}] {version_name} 배경 제거 작업 완료")
            return result
            
        except Exception as e:
            logger.error(f"[{image_id}] 배경 제거 중 오류 발생: {str(e)}")
            
            # 오류 발생 시 GPU 메모리 정리
            try:
                model_manager = get_model_manager()
                model_manager.cleanup()
            except:
                pass
            
            # 실패 시 원본 이미지 반환 (RGBA 변환)
            result = image.convert('RGBA') if image.mode != 'RGBA' else image
            logger.warning(f"[{image_id}] 원본 이미지 반환")
            return result