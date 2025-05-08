"""
배경 제거 서비스의 메인 로직
"""
import os
import numpy as np
from PIL import Image
from app.utils.logger import setup_logger
from app.core.model_manager import get_model_manager
from app.services.image_processor import ImageProcessor
from app.services.mask_selector import MaskSelector

# 로거 설정
logger = setup_logger(__name__)

class BackgroundRemover:
    """배경 제거 서비스"""
    
    @staticmethod
    def remove_background(image: Image.Image, x: int, y: int, image_name: str) -> Image.Image:
        """클릭한 위치의 객체 배경을 제거하는 함수"""
        image_id = os.path.basename(image_name)
        total_steps = 5
        
        try:
            # 1. 이미지 전처리
            logger.info(f"[{image_id}] Step 1/{total_steps}: 이미지 전처리 시작")
            resized_image, scale = ImageProcessor.resize_image_if_needed(image)
            img_np = np.array(resized_image.convert("RGB"))
            
            # 좌표 스케일 조정
            x = int(x * scale)
            y = int(y * scale)
            logger.info(f"[{image_id}] Step 1/{total_steps}: 이미지 전처리 완료")
            
            # 2. 이미지 분석
            logger.info(f"[{image_id}] Step 2/{total_steps}: 이미지 분석 시작")
            is_near_edge, edge_strength = ImageProcessor.analyze_image_for_object(img_np, x, y)
            logger.info(f"[{image_id}] Step 2/{total_steps}: 이미지 분석 완료")
            
            # 3. 모델 예측
            logger.info(f"[{image_id}] Step 3/{total_steps}: 모델 예측 시작")
            model_manager = get_model_manager()
            input_point = np.array([[x, y]])
            input_label = np.array([1])
            
            masks, scores, _ = model_manager.predict(img_np, input_point, input_label)
            logger.info(f"[{image_id}] Step 3/{total_steps}: 모델 예측 완료")
            
            # 4. 마스크 선택
            logger.info(f"[{image_id}] Step 4/{total_steps}: 마스크 선택 시작")
            mask_idx = MaskSelector.select_best_mask(masks, scores, img_np, x, y, is_near_edge, edge_strength)
            mask = masks[mask_idx]
            logger.info(f"[{image_id}] Step 4/{total_steps}: 마스크 선택 완료")
            
            # 5. 마스크 적용
            logger.info(f"[{image_id}] Step 5/{total_steps}: 마스크 적용 시작")
            result = ImageProcessor.apply_mask_to_image(image, mask, scale)
            logger.info(f"[{image_id}] Step 5/{total_steps}: 마스크 적용 완료")
            
            # 대용량 이미지 처리 후 GPU 메모리 정리
            if max(image.size) > 1000:
                model_manager.cleanup()
                
            return result
            
        except Exception as e:
            logger.error(f"[{image_id}] 배경 제거 중 오류 발생: {str(e)}")
            model_manager = get_model_manager()
            model_manager.cleanup()
            return image.convert('RGBA') if image.mode != 'RGBA' else image 