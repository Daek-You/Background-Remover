""" 마스크 선택 로직을 제공하는 서비스 모듈 """
import numpy as np
from app.utils.logger import setup_logger
from config.settings import MASK_SELECTION, MODEL
from app.core.processors.parallel_executor import get_executor_manager
from app.services.image_processing.image_analyzer import ImageAnalyzer

# 로거 설정
logger = setup_logger(__name__)

class MaskSelector:
    """마스크 선택 및 평가 서비스"""
    
    @staticmethod
    def evaluate_mask(args):
        """
        마스크 평가 함수 (병렬 처리용)
        
        Args:
            args: 평가에 필요한 인자들의 튜플
                (index, mask, score, img_np, x, y, is_near_edge, total_pixels)
        
        Returns:
            tuple: (마스크_인덱스, 점수, 평가_결과) 또는 None
        """
        i, mask, score, img_np, x, y, is_near_edge, total_pixels = args
        
        # 마스크 크기 계산
        mask_pixels = np.sum(mask)
        mask_percentage = mask_pixels / total_pixels * 100
        
        # 크기가 적절한지 빠르게 확인
        if not (MASK_SELECTION['MIN_SIZE_PERCENTAGE'] <= mask_percentage <= MASK_SELECTION['MAX_SIZE_PERCENTAGE']):
            logger.debug(f"마스크 {i}: 크기 제외 ({mask_percentage:.1f}%)")
            return None
            
        # 점수가 충분히 높은지 확인
        if score < MASK_SELECTION['SCORE_THRESHOLD']:
            logger.debug(f"마스크 {i}: 점수 제외 ({score:.3f})")
            return None
        
        # SAM 2.1에서 개선된 마스크 평가
        version_name = MODEL.get('VERSION_NAME', MODEL['VERSION'])
        
        # 필요한 경우에만 에지 정렬 분석 수행
        if is_near_edge:
            edge_alignment = ImageAnalyzer.analyze_mask_edge_alignment(mask, img_np)
            logger.debug(f"마스크 {i}: 에지 정렬 점수 {edge_alignment:.3f}")
            return (i, score, edge_alignment)
        else:
            # 에지 근처가 아닌 경우 크기 기반 점수 사용
            size_score = 1.0 - abs(50 - mask_percentage) / 45
            logger.debug(f"마스크 {i}: 크기 점수 {size_score:.3f}")
            return (i, score, size_score)
    
    @staticmethod
    def _check_click_in_mask(masks, x, y):
        """
        클릭 위치가 마스크에 포함되는지 확인
        
        Args:
            masks: 마스크 배열
            x: 클릭한 x 좌표
            y: 클릭한 y 좌표
            
        Returns:
            list: 유효한 마스크들의 (인덱스, 점수) 리스트
        """
        valid_masks = []
        for i, mask in enumerate(masks):
            if y < mask.shape[0] and x < mask.shape[1] and mask[y, x]:
                valid_masks.append(i)
        
        logger.debug(f"클릭 위치를 포함하는 마스크: {len(valid_masks)}개")
        return valid_masks
    
    @staticmethod
    def _filter_top_masks(valid_masks, scores):
        """
        점수 기반으로 상위 마스크들 선택
        
        Args:
            valid_masks: 유효한 마스크 인덱스들
            scores: 각 마스크의 점수
            
        Returns:
            list: 상위 마스크들의 (인덱스, 점수) 리스트
        """
        if not valid_masks:
            return []
        
        # 점수와 함께 정렬
        mask_scores = [(i, scores[i]) for i in valid_masks]
        mask_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 N개만 선택
        top_count = MASK_SELECTION['TOP_MASKS_COUNT']
        top_masks = mask_scores[:top_count]
        
        logger.debug(f"상위 {len(top_masks)}개 마스크 선택")
        return top_masks
    
    @staticmethod
    def _calculate_final_score(candidates, is_near_edge):
        """
        최종 마스크 점수 계산
        
        Args:
            candidates: 후보 마스크들 [(인덱스, 예측점수, 평가점수)]
            is_near_edge: 에지 근처 여부
            
        Returns:
            list: 정렬된 후보 마스크들
        """
        if is_near_edge:
            # 에지 근처인 경우 에지 정렬을 더 중요하게
            candidates.sort(key=lambda x: (
                x[2] * MASK_SELECTION['EDGE_WEIGHT'] + 
                x[1] * MASK_SELECTION['SCORE_WEIGHT']
            ), reverse=True)
            logger.debug("에지 기반 점수로 정렬")
        else:
            # 에지가 아닌 경우 크기와 점수 조합
            candidates.sort(key=lambda x: (
                x[2] * MASK_SELECTION['SIZE_WEIGHT'] + 
                x[1] * MASK_SELECTION['SCORE_WEIGHT']
            ), reverse=True)
            logger.debug("크기 기반 점수로 정렬")
        
        return candidates
    
    @staticmethod
    def select_best_mask(masks, scores, img_np, x, y, is_near_edge, edge_strength=0):
        """
        여러 마스크 중 가장 적합한 마스크를 선택하는 함수
        
        Args:
            masks: 예측된 마스크 배열
            scores: 각 마스크의 점수
            img_np: 이미지 numpy 배열
            x: 클릭한 x 좌표
            y: 클릭한 y 좌표
            is_near_edge: 에지 근처 여부
            edge_strength: 에지 강도 (SAM 2.1에서 사용)
            
        Returns:
            int: 선택된 마스크의 인덱스
        """
        version_name = MODEL.get('VERSION_NAME', MODEL['VERSION'])
        logger.debug(f"{version_name} 마스크 선택 시작")
        
        # 1. 빠른 필터링: 클릭 위치가 마스크에 포함되지 않는 것들은 즉시 제외
        valid_mask_indices = MaskSelector._check_click_in_mask(masks, x, y)
        
        if not valid_mask_indices:
            logger.warning("클릭 위치를 포함하는 마스크가 없음. 최고 점수 마스크 선택")
            return np.argmax(scores)
        
        # 2. 점수 기반 초기 필터링 (상위 N개만 선택)
        top_masks = MaskSelector._filter_top_masks(valid_mask_indices, scores)
        
        if not top_masks:
            logger.warning("필터링 후 남은 마스크가 없음. 최고 점수 마스크 선택")
            return np.argmax(scores)
        
        # 3. 병렬로 마스크 평가
        h, w = img_np.shape[:2]
        total_pixels = h * w
        
        # 병렬 처리를 위한 실행기 가져오기
        executor_manager = get_executor_manager()
        
        # 작업 병렬 처리
        args = [(i, masks[i], score, img_np, x, y, is_near_edge, total_pixels) 
                for i, score in top_masks]
        results = executor_manager.map(MaskSelector.evaluate_mask, args)
        
        # 4. 유효한 결과만 필터링
        candidate_masks = [r for r in results if r is not None]
        
        if candidate_masks:
            # 5. 최종 점수 계산 및 선택
            sorted_candidates = MaskSelector._calculate_final_score(candidate_masks, is_near_edge)
            
            selected_idx = sorted_candidates[0][0]
            selected_score = sorted_candidates[0][1]
            selected_eval = sorted_candidates[0][2]
            
            logger.info(f"마스크 {selected_idx} 선택 (점수: {selected_score:.3f}, 평가: {selected_eval:.3f})")
            return selected_idx
        
        # 6. 모든 평가가 실패한 경우 첫 번째 유효한 마스크 선택
        logger.warning("모든 마스크 평가 실패. 첫 번째 유효한 마스크 선택")
        return top_masks[0][0]