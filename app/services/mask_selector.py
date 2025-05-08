"""
마스크 선택 로직을 제공하는 서비스 모듈
"""
import numpy as np
from app.utils.logger import setup_logger
from config.settings import MASK_SELECTION
from app.core.parallel_executor import get_executor_manager
from app.services.image_processor import ImageProcessor

# 로거 설정
logger = setup_logger(__name__)

class MaskSelector:
    """마스크 선택 및 평가 서비스"""
    
    @staticmethod
    def evaluate_mask(args):
        """마스크 평가 함수 (병렬 처리용)"""
        i, mask, score, img_np, x, y, is_near_edge, total_pixels = args
        
        mask_pixels = np.sum(mask)
        mask_percentage = mask_pixels / total_pixels * 100
        
        # 크기가 적절한지 빠르게 확인
        if not (MASK_SELECTION['MIN_SIZE_PERCENTAGE'] <= mask_percentage <= MASK_SELECTION['MAX_SIZE_PERCENTAGE']):
            return None
            
        # 점수가 충분히 높은지 확인
        if score < MASK_SELECTION['SCORE_THRESHOLD']:
            return None
        
        # 필요한 경우에만 에지 정렬 분석 수행
        if is_near_edge:
            edge_alignment = ImageProcessor.analyze_mask_edge_alignment(mask, img_np)
            return (i, score, edge_alignment)
        else:
            size_score = 1.0 - abs(50 - mask_percentage) / 45
            return (i, score, size_score)
    
    @staticmethod
    def select_best_mask(masks, scores, img_np, x, y, is_near_edge, edge_strength=0):
        """여러 마스크 중 가장 적합한 마스크를 선택하는 함수"""
        # 1. 빠른 필터링: 클릭 위치가 마스크에 포함되지 않는 것들은 즉시 제외
        valid_masks = [(i, scores[i]) for i, mask in enumerate(masks) if mask[y, x]]
        
        if not valid_masks:
            return np.argmax(scores)
        
        # 2. 점수 기반 초기 필터링 (상위 N개만 선택)
        valid_masks.sort(key=lambda x: x[1], reverse=True)
        top_masks = valid_masks[:MASK_SELECTION['TOP_MASKS_COUNT']]
        
        # 3. 병렬로 마스크 평가
        h, w = img_np.shape[:2]
        total_pixels = h * w
        
        # 병렬 처리를 위한 실행기 가져오기
        executor_manager = get_executor_manager()
        
        # 작업 병렬 처리
        args = [(i, masks[i], score, img_np, x, y, is_near_edge, total_pixels) 
                for i, score in top_masks]
        results = executor_manager.map(MaskSelector.evaluate_mask, args)
        
        # 유효한 결과만 필터링
        candidate_masks = [r for r in results if r is not None]
        
        if candidate_masks:
            if is_near_edge:
                candidate_masks.sort(key=lambda x: (
                    x[2] * MASK_SELECTION['EDGE_WEIGHT'] + 
                    x[1] * MASK_SELECTION['SCORE_WEIGHT']
                ), reverse=True)
            else:
                candidate_masks.sort(key=lambda x: (
                    x[2] * MASK_SELECTION['SIZE_WEIGHT'] + 
                    x[1] * MASK_SELECTION['SCORE_WEIGHT']
                ), reverse=True)
            return candidate_masks[0][0]
        
        return top_masks[0][0] 