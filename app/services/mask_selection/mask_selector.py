"""마스크 선택 로직을 제공하는 서비스 모듈"""
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from app.utils.logger import setup_logger
from config.settings import MASK_SELECTION, MODEL
from app.core.processors.parallel_executor import get_executor_manager
from app.services.image_processing.image_analyzer import ImageAnalyzer

logger = setup_logger(__name__)

class MaskSelector:
    """마스크 선택 및 평가 서비스"""
    
    @staticmethod
    def select_best_mask(
        masks: np.ndarray, 
        scores: np.ndarray, 
        img_np: np.ndarray, 
        x: int, 
        y: int, 
        is_near_edge: bool, 
        edge_strength: float = 0
    ) -> int:
        """
        여러 마스크 중 가장 적합한 마스크를 선택
        
        Args:
            masks: 예측된 마스크 배열
            scores: 각 마스크의 점수
            img_np: 이미지 numpy 배열
            x, y: 클릭한 좌표
            is_near_edge: 에지 근처 여부
            edge_strength: 에지 강도
            
        Returns:
            int: 선택된 마스크의 인덱스
        """
        logger.debug(f"마스크 선택 시작 - 총 {len(masks)}개 마스크")
        
        # 1. 기본 유효성 검사
        valid_indices = MaskSelector._find_valid_masks(masks, x, y)
        if not valid_indices:
            logger.warning("클릭 위치를 포함하는 마스크가 없음")
            return np.argmax(scores)
        
        # 2. 상위 마스크들만 선별 (성능 최적화)
        top_masks = MaskSelector._filter_top_masks(valid_indices, scores)
        if not top_masks:
            logger.warning("점수 필터링 후 남은 마스크가 없음")
            return np.argmax(scores)
        
        # 3. 상세 평가 실행
        candidates = MaskSelector._evaluate_masks_parallel(
            masks, scores, img_np, x, y, is_near_edge, top_masks
        )
        
        # 4. 최종 선택
        if candidates:
            best_idx = MaskSelector._select_final_mask(candidates, is_near_edge)
            logger.info(f"마스크 {best_idx} 선택 완료")
            return best_idx
        
        # 5. 평가 실패 시 기본 선택
        logger.warning("모든 평가 실패, 첫 번째 유효 마스크 선택")
        return top_masks[0][0]
    
    @staticmethod
    def _find_valid_masks(masks: np.ndarray, x: int, y: int) -> List[int]:
        """클릭 위치를 포함하는 마스크들 찾기"""
        valid_masks = []
        threshold = MASK_SELECTION['CLICK_INCLUSION_THRESHOLD']
        
        for i, mask in enumerate(masks):
            if y < mask.shape[0] and x < mask.shape[1] and mask[y, x] > threshold:
                valid_masks.append(i)
        
        logger.debug(f"클릭 위치({x}, {y})를 포함하는 마스크: {len(valid_masks)}개")
        return valid_masks
    
    @staticmethod
    def _filter_top_masks(
        valid_indices: List[int], 
        scores: np.ndarray
    ) -> List[Tuple[int, float]]:
        """점수 기반으로 상위 마스크들 선별"""
        # 점수와 함께 정렬
        mask_scores = [(i, scores[i]) for i in valid_indices]
        mask_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_count = MASK_SELECTION['TOP_MASKS_COUNT']
        top_masks = mask_scores[:top_count]
        
        logger.debug(f"상위 {len(top_masks)}개 마스크 선택")
        return top_masks
    
    @staticmethod
    def _evaluate_single_mask(args: Tuple) -> Optional[Tuple[int, float, float]]:
        """단일 마스크 평가 (병렬 처리용)"""
        i, mask, score, img_np, x, y, is_near_edge, total_pixels = args
        
        try:
            # 마스크 크기 계산
            mask_pixels = np.sum(mask)
            mask_percentage = mask_pixels / total_pixels * 100
            
            min_size = MASK_SELECTION['MIN_SIZE_PERCENTAGE']
            max_size = MASK_SELECTION['MAX_SIZE_PERCENTAGE']
            score_threshold = MASK_SELECTION['SCORE_THRESHOLD']
            
            # 크기 필터링
            if not (min_size <= mask_percentage <= max_size):
                logger.debug(f"마스크 {i}: 크기 부적합 ({mask_percentage:.1f}%)")
                return None
            
            # 점수 필터링
            if score < score_threshold:
                logger.debug(f"마스크 {i}: 점수 부족 ({score:.3f})")
                return None
            
            # 평가 점수 계산
            if is_near_edge:
                eval_score = ImageAnalyzer.analyze_mask_edge_alignment(mask, img_np)
                logger.debug(f"마스크 {i}: 에지 정렬 점수 {eval_score:.3f}")
            else:
                baseline = MASK_SELECTION['SIZE_SCORE_BASELINE']
                deviation = MASK_SELECTION['SIZE_SCORE_DEVIATION']
                eval_score = 1.0 - abs(baseline - mask_percentage) / deviation
                logger.debug(f"마스크 {i}: 크기 점수 {eval_score:.3f}")
            
            return (i, score, eval_score)
            
        except Exception as e:
            logger.error(f"마스크 {i} 평가 실패: {str(e)}")
            return None
    
    @staticmethod
    def _evaluate_masks_parallel(
        masks: np.ndarray,
        scores: np.ndarray,
        img_np: np.ndarray,
        x: int, y: int,
        is_near_edge: bool,
        top_masks: List[Tuple[int, float]]
    ) -> List[Tuple[int, float, float]]:
        """마스크들을 병렬로 평가"""
        h, w = img_np.shape[:2]
        total_pixels = h * w
        
        # 병렬 처리를 위한 인자 준비
        args = [(i, masks[i], score, img_np, x, y, is_near_edge, total_pixels) 
                for i, score in top_masks]
        
        # 병렬 평가 실행
        executor_manager = get_executor_manager()
        results = executor_manager.map(MaskSelector._evaluate_single_mask, args)
        
        # 유효한 결과만 필터링
        candidates = [r for r in results if r is not None]
        logger.debug(f"평가 완료: {len(candidates)}개 후보")
        
        return candidates
    
    @staticmethod
    def _select_final_mask(
        candidates: List[Tuple[int, float, float]], 
        is_near_edge: bool
    ) -> int:
        """최종 마스크 선택"""
        # settings에서 가중치들 가져오기
        edge_weight = MASK_SELECTION['EDGE_WEIGHT']
        score_weight = MASK_SELECTION['SCORE_WEIGHT']
        size_weight = MASK_SELECTION['SIZE_WEIGHT']
        
        if is_near_edge:
            # 에지 근처: 에지 정렬 우선
            candidates.sort(key=lambda x: (
                x[2] * edge_weight + x[1] * score_weight
            ), reverse=True)
            logger.debug("에지 기반 점수로 정렬")
        else:
            # 일반 영역: 크기와 점수 조합
            candidates.sort(key=lambda x: (
                x[2] * size_weight + x[1] * score_weight
            ), reverse=True)
            logger.debug("크기 기반 점수로 정렬")
        
        selected = candidates[0]
        logger.debug(f"최종 선택: 마스크 {selected[0]} (점수: {selected[1]:.3f}, 평가: {selected[2]:.3f})")
        
        return selected[0]