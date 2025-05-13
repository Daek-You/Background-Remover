""" 배경 제거 서비스 메인 클래스 """
import os
import numpy as np
from PIL import Image
from typing import Tuple, List
from app.utils.logger import setup_logger
from app.core.model_management.model_manager import get_model_manager
from app.services.background_removal.background_removal_steps import BackgroundRemovalSteps
from config.settings import MODEL, LOGGING, BACKGROUND_REMOVAL, LARGE_IMAGE_THRESHOLD, GPU_MEMORY

logger = setup_logger(__name__)

class BackgroundRemover:
    """배경 제거 서비스 메인 클래스"""
    
    @staticmethod
    def remove_background(
        image: Image.Image, 
        click_points: List[Tuple[int, int]], 
        image_name: str
    ) -> Image.Image:
        """
        여러 클릭 위치의 객체 배경을 제거
        
        Args:
            image: 원본 이미지
            click_points: 클릭한 좌표 리스트
            image_name: 이미지 이름
            
        Returns:
            Image.Image: 배경이 제거된 이미지
        """
        image_id = os.path.basename(image_name)
        version_name = MODEL.get('VERSION_NAME', MODEL['VERSION'])
        
        try:
            logger.info(f"[{image_id}] {version_name} 배경 제거 시작")
            BackgroundRemover._log_input_info(image_id, image, click_points)
            
            # 입력 검증
            if not BackgroundRemover._validate_input(image, click_points, image_id):
                return BackgroundRemover._create_rgba_image(image)
            
            # 5단계 처리 파이프라인
            result = BackgroundRemover._execute_pipeline(image, click_points, image_id)
            
            # 후처리
            BackgroundRemover._post_process(image, image_id)
            
            logger.info(f"[{image_id}] {version_name} 배경 제거 완료")
            return result
            
        except Exception as e:
            return BackgroundRemover._handle_error(e, image, image_id)
    
    @staticmethod
    def _log_input_info(image_id: str, image: Image.Image, click_points: List[Tuple[int, int]]):
        """입력 정보를 로그에 기록"""
        logger.info(f"[{image_id}] 입력 - 이미지 크기: {image.size}, 클릭 위치: {len(click_points)}개")
        if logger.isEnabledFor(10):  # DEBUG level
            logger.debug(f"[{image_id}] 클릭 좌표: {click_points}")
    
    @staticmethod
    def _validate_input(
        image: Image.Image, 
        click_points: List[Tuple[int, int]], 
        image_id: str
    ) -> bool:
        """입력 유효성 검증"""
        for i, (x, y) in enumerate(click_points):
            if x < 0 or x >= image.size[0] or y < 0 or y >= image.size[1]:
                logger.error(f"[{image_id}] 잘못된 클릭 좌표 #{i+1}: ({x}, {y}), 이미지 크기: {image.size}")
                return False
        return True
    
    @staticmethod
    def _execute_pipeline(
        image: Image.Image, 
        click_points: List[Tuple[int, int]], 
        image_id: str
    ) -> Image.Image:
        """배경 제거 파이프라인 실행"""
        # 1. 이미지 전처리
        img_np, scaled_points, scale = BackgroundRemovalSteps.preprocess_image(
            image, click_points, image_id
        )
        
        # 2. 이미지 분석 (첫 번째 포인트 기준)
        x0, y0 = scaled_points[0]
        is_near_edge, edge_strength, context_info = BackgroundRemovalSteps.analyze_image(
            img_np, x0, y0, image_id
        )
        
        # 3. 모델 예측 (다중 포인트)
        masks, scores, logits = BackgroundRemovalSteps.predict_masks(
            img_np, scaled_points, image_id
        )
        
        # 예측 결과 로깅
        BackgroundRemover._log_prediction_results(image_id, masks, scores)
        
        # 4. 마스크 선택
        selected_mask = BackgroundRemovalSteps.select_best_mask(
            masks, scores, img_np, x0, y0, 
            is_near_edge, edge_strength, context_info, image_id
        )
        
        # 5. 마스크 정제 및 적용
        refined_mask = BackgroundRemovalSteps.refine_selected_mask(selected_mask, image_id)
        result = BackgroundRemovalSteps.apply_mask_to_image(image, refined_mask, scale, image_id)
        
        # 디버그 정보 저장 (설정에 따라)
        BackgroundRemover._save_debug_info_if_enabled(
            image, click_points, scaled_points, scale, masks, refined_mask, image_id
        )
        
        return result
    
    @staticmethod
    def _log_prediction_results(image_id: str, masks: np.ndarray, scores: np.ndarray):
        """예측 결과를 상세하게 로그에 기록"""
        logger.info(f"[{image_id}] 예측된 마스크 수: {len(masks)}")
        
        if logger.isEnabledFor(10):  # DEBUG level
            for i, (mask, score) in enumerate(zip(masks, scores)):
                unique_values = len(np.unique(mask))
                zero_ratio = np.sum(mask < 0.1) / mask.size
                one_ratio = np.sum(mask > 0.9) / mask.size
                logger.debug(
                    f"[{image_id}] 마스크 {i}: 점수={score:.3f}, "
                    f"크기={mask.shape}, 고유값={unique_values}, "
                    f"0근처={zero_ratio:.1%}, 1근처={one_ratio:.1%}"
                )
    
    @staticmethod
    def _save_debug_info_if_enabled(
        image: Image.Image,
        click_points: List[Tuple[int, int]],
        scaled_points: List[Tuple[int, int]],
        scale: float,
        masks: np.ndarray,
        refined_mask: np.ndarray,
        image_id: str
    ):
        """설정에 따라 디버그 정보 저장"""
        # 로깅 레벨과 설정 둘 다 체크
        if (LOGGING.get('LEVEL') == 'DEBUG' and 
            BACKGROUND_REMOVAL.get('DEBUG_INFO_SAVE', False)):
            
            logger.debug(f"[{image_id}] 디버그 모드 활성화됨")
            
            # 디버그 정보 생성
            debug_info = {
                'image_size': list(image.size),
                'click_positions': click_points,
                'scaled_positions': scaled_points,
                'scale': scale,
                'masks_count': len(masks),
                'selected_mask_stats': {
                    'min': float(refined_mask.min()),
                    'max': float(refined_mask.max()),
                    'mean': float(refined_mask.mean()),
                    'nonzero_ratio': float(np.sum(refined_mask > 0) / refined_mask.size)
                }
            }
            
            # JSON 파일로 저장
            BackgroundRemover._save_debug_json(debug_info, image_id)
    
    @staticmethod
    def _save_debug_json(debug_info: dict, image_id: str):
        """디버그 정보를 JSON 파일로 저장"""
        try:
            import json
            debug_file_path = f"debug_{image_id.replace('.', '_')}.json"
            with open(debug_file_path, 'w') as f:
                json.dump(debug_info, f, indent=2)
            logger.debug(f"[{image_id}] 디버그 정보 저장: {debug_file_path}")
        except Exception as e:
            logger.warning(f"[{image_id}] 디버그 정보 저장 실패: {str(e)}")
    
    @staticmethod
    def _post_process(image: Image.Image, image_id: str):
        """후처리 작업 (메모리 정리 등)"""
        if max(image.size) > LARGE_IMAGE_THRESHOLD:
            model_manager = get_model_manager()
            model_manager.cleanup()
            logger.debug(f"[{image_id}] 대용량 이미지 처리 후 GPU 메모리 정리")
    
    @staticmethod
    def _handle_error(e: Exception, image: Image.Image, image_id: str) -> Image.Image:
        """에러 처리 및 원본 이미지 반환"""
        logger.error(f"[{image_id}] 배경 제거 중 오류 발생: {str(e)}")
        
        if logger.isEnabledFor(10):  # DEBUG level
            import traceback
            logger.debug(f"[{image_id}] 전체 스택 트레이스:\n{traceback.format_exc()}")
        
        # 오류 발생 시 GPU 메모리 정리
        try:
            model_manager = get_model_manager()
            model_manager.cleanup()
        except:
            pass
        
        logger.warning(f"[{image_id}] 원본 이미지 반환")
        return BackgroundRemover._create_rgba_image(image)
    
    @staticmethod
    def _create_rgba_image(image: Image.Image) -> Image.Image:
        """이미지를 RGBA 형식으로 변환"""
        return image.convert('RGBA') if image.mode != 'RGBA' else image