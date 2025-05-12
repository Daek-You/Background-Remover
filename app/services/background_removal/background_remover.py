"""
배경 제거 서비스 메인 클래스
"""
import os
import numpy as np
from PIL import Image
from app.utils.logger import setup_logger
from app.core.model_management.model_manager import get_model_manager
from app.services.background_removal.background_removal_steps import BackgroundRemovalSteps
from config.settings import MODEL, LOGGING

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
            logger.info(f"[{image_id}] 입력 매개변수: 이미지 크기={image.size}, 클릭 위치=({x}, {y})")
            
            # 입력 검증
            if x < 0 or x >= image.size[0] or y < 0 or y >= image.size[1]:
                logger.error(f"[{image_id}] 잘못된 클릭 좌표: ({x}, {y}), 이미지 크기: {image.size}")
                return image.convert('RGBA') if image.mode != 'RGBA' else image
            
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
            
            # 디버깅: 예측된 마스크 정보 상세 로깅
            logger.info(f"[{image_id}] 예측된 마스크 수: {len(masks)}")
            for i, (mask, score) in enumerate(zip(masks, scores)):
                unique_values = len(np.unique(mask))
                zero_ratio = np.sum(mask < 0.1) / mask.size
                one_ratio = np.sum(mask > 0.9) / mask.size
                logger.debug(f"[{image_id}] 마스크 {i}: 점수={score:.3f}, "
                           f"크기={mask.shape}, 고유값수={unique_values}, "
                           f"0 근처={zero_ratio:.1%}, 1 근처={one_ratio:.1%}")
            
            # 4. 마스크 선택
            selected_mask = BackgroundRemovalSteps.select_best_mask(
                masks, scores, img_np, scaled_x, scaled_y, 
                is_near_edge, edge_strength, context_info, image_id
            )
            
            # 4.5. 마스크 정제 (새로 추가)
            refined_mask = BackgroundRemovalSteps.refine_selected_mask(selected_mask, image_id)
            
            # 5. 마스크 적용
            result = BackgroundRemovalSteps.apply_mask_to_image(
                image, refined_mask, scale, image_id
            )
            
            # 디버깅: 결과 이미지 저장 (설정 기반)
            if LOGGING.get('LEVEL', 'INFO') == 'DEBUG':
                logger.debug(f"[{image_id}] 디버그 모드 활성화됨")
                # 간단한 디버그 정보 파일 저장 (이미지는 생략)
                debug_info = {
                    'image_size': image.size,
                    'click_position': (x, y),
                    'scaled_position': (scaled_x, scaled_y),
                    'scale': scale,
                    'masks_count': len(masks),
                    'selected_mask_stats': {
                        'min': float(refined_mask.min()),
                        'max': float(refined_mask.max()),
                        'mean': float(refined_mask.mean()),
                        'nonzero_ratio': float(np.sum(refined_mask > 0) / refined_mask.size)
                    }
                }
                
                try:
                    import json
                    debug_file_path = f"debug_{image_id.replace('.', '_')}.json"
                    with open(debug_file_path, 'w') as f:
                        json.dump(debug_info, f, indent=2)
                    logger.debug(f"[{image_id}] 디버그 정보 저장됨: {debug_file_path}")
                except Exception as debug_e:
                    logger.warning(f"[{image_id}] 디버그 정보 저장 실패: {str(debug_e)}")
            
            # GPU 메모리 정리 (대용량 이미지 처리 후)
            if max(image.size) > 1000:
                model_manager = get_model_manager()
                model_manager.cleanup()
                logger.debug(f"[{image_id}] 대용량 이미지 처리 후 GPU 메모리 정리")
            
            logger.info(f"[{image_id}] {version_name} 배경 제거 작업 완료")
            return result
            
        except Exception as e:
            logger.error(f"[{image_id}] 배경 제거 중 오류 발생: {str(e)}")
            import traceback
            logger.error(f"[{image_id}] 전체 스택 트레이스:\n{traceback.format_exc()}")
            
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