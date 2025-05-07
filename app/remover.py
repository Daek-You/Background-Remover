import threading
import cv2
import numpy as np
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from utils.model_utils import download_sam_model
from utils.logger import setup_logger
from config.settings import IMAGE_ANALYSIS, MASK_SELECTION, MODEL

# 로거 설정
logger = setup_logger(__name__)

# 전역 변수
predictor = None
predictor_lock = threading.Lock()

def get_predictor():
    """predictor 객체 반환 (필요시 초기화, 스레드 안전)"""
    global predictor
    
    # 이미 초기화되었는지 빠르게 확인
    if predictor is not None:
        return predictor
    
    # 잠금 획득 후 다시 확인
    with predictor_lock:
        if predictor is None:
            # 모델 파일 확인 및 다운로드
            model_path = download_sam_model(model_type=MODEL['TYPE'])
            
            # SAM 모델 로드
            logger.info("SAM 모델 로드 중...")
            sam = sam_model_registry[MODEL['TYPE']](checkpoint=model_path)
            predictor = SamPredictor(sam)
            logger.info("SAM 모델 로드 완료!")
    
    return predictor

def analyze_image_for_object(img_np, x, y, radius=IMAGE_ANALYSIS['CLICK_RADIUS']):
    """ 클릭 위치 주변을 분석하여 객체 특성 파악 """
    # 클릭 주변 영역 추출 (반경 내의 픽셀 분석)
    h, w = img_np.shape[:2]
    x_min = max(0, x - radius)
    y_min = max(0, y - radius)
    x_max = min(w, x + radius)
    y_max = min(h, y + radius)
    
    # 관심 영역 추출
    roi = img_np[y_min:y_max, x_min:x_max]
    
    # 간단한 에지 검출로 객체 경계 강화
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 
                     IMAGE_ANALYSIS['CANNY_THRESHOLD_LOW'],
                     IMAGE_ANALYSIS['CANNY_THRESHOLD_HIGH'])
    
    # 클릭 위치 주변 에지 확인
    center_y, center_x = radius, radius
    edge_radius = IMAGE_ANALYSIS['EDGE_CHECK_RADIUS']
    nearby_region = edges[max(0, center_y-edge_radius):min(2*radius, center_y+edge_radius), 
                         max(0, center_x-edge_radius):min(2*radius, center_x+edge_radius)]
    
    # 에지 존재 여부 및 강도
    is_near_edge = np.any(nearby_region > 0)
    edge_strength = np.sum(nearby_region) / (nearby_region.size * 255) if nearby_region.size > 0 else 0
    
    return is_near_edge, edge_strength

def analyze_mask_edge_alignment(mask, img_np):
    """
    마스크가 이미지의 에지와 얼마나 잘 일치하는지 분석
    
    Args:
        mask: 분석할 마스크
        img_np: 원본 이미지
        
    Returns:
        edge_alignment_score: 마스크와 이미지 에지의 일치도 점수 (0~1)
    """
    # 1. 전체 이미지가 아닌, 마스크 경계만 분석
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 2. 마스크 경계 주변만 에지 검출
    mask_edge = np.zeros_like(img_np[:,:,0])
    cv2.drawContours(mask_edge, mask_contours, -1, 255, 1)
    
    # 3. 마스크 경계 주변 영역만 추출
    kernel = np.ones((3,3), np.uint8)
    mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=2)
    roi = np.where(mask_dilated > 0)
    
    if len(roi[0]) == 0:
        return 0.0
    
    # 4. 관심 영역만 에지 검출
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 
                     IMAGE_ANALYSIS['CANNY_THRESHOLD_LOW'],
                     IMAGE_ANALYSIS['CANNY_THRESHOLD_HIGH'])
    
    # 5. 마스크 경계와 에지의 겹치는 정도를 관심 영역 부분만 계산
    overlap = np.logical_and(edges[roi] > 0, mask_edge[roi] > 0)
    edge_alignment_score = np.sum(overlap) / np.sum(mask_edge[roi] > 0)
    
    return edge_alignment_score

def select_best_mask(masks, scores, img_np, x, y, is_near_edge, edge_strength=0):
    """
    여러 마스크 중 가장 적합한 마스크를 선택하는 함수
    
    Args:
        masks: SAM 모델이 생성한 마스크 후보들 (배열)
        scores: 각 마스크의 신뢰도 점수 (배열)
        img_np: 원본 이미지 NumPy 배열
        x, y: 클릭 좌표
        is_near_edge: 클릭 위치가 경계에 가까운지 여부
        edge_strength: 경계의 강도 (0~1)
        
    Returns:
        best_mask_idx: 가장 적합한 마스크의 인덱스
    """
    # 1. 빠른 필터링: 클릭 위치가 마스크에 포함되지 않는 것들은 즉시 제외
    valid_masks = [(i, scores[i]) for i, mask in enumerate(masks) if mask[y, x]]
    
    if not valid_masks:
        return np.argmax(scores)
    
    # 2. 점수 기반 초기 필터링 (상위 N개만 선택)
    valid_masks.sort(key=lambda x: x[1], reverse=True)
    top_masks = valid_masks[:MASK_SELECTION['TOP_MASKS_COUNT']]
    
    # 3. 필요한 경우에만 상세 분석
    if len(top_masks) > 1:
        h, w = img_np.shape[:2]
        total_pixels = h * w
        
        candidate_masks = []
        for i, score in top_masks:
            mask = masks[i]
            mask_pixels = np.sum(mask)
            mask_percentage = mask_pixels / total_pixels * 100
            
            # 크기가 적절한지 빠르게 확인
            if not (MASK_SELECTION['MIN_SIZE_PERCENTAGE'] <= mask_percentage <= MASK_SELECTION['MAX_SIZE_PERCENTAGE']):
                continue
                
            # 점수가 충분히 높은지 확인
            if score < scores[np.argmax(scores)] * MASK_SELECTION['SCORE_THRESHOLD']:
                continue
            
            # 필요한 경우에만 에지 정렬 분석 수행
            if is_near_edge:
                edge_alignment = analyze_mask_edge_alignment(mask, img_np)
                candidate_masks.append((i, score, edge_alignment))
            else:
                size_score = 1.0 - abs(50 - mask_percentage) / 45
                candidate_masks.append((i, score, size_score))
        
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

def remove_background_from_image(image: Image.Image, x: int, y: int) -> Image.Image:
    """
    클릭한 위치의 객체 배경을 제거하는 함수 (아이폰 스타일)
    
    Args:
        image: 입력 이미지 (PIL Image)
        x, y: 사용자가 클릭한 좌표
        
    Returns:
        배경이 제거된 이미지 (투명 배경, RGBA 형식)
    """
    try:
        # 이미지를 numpy 배열로 변환 (모델 입력용)
        img_np = np.array(image.convert("RGB"))
        
        # 에지 검출 및 객체 분석 (향상된 결과를 위한 추가 정보)
        is_near_edge, edge_strength = analyze_image_for_object(img_np, x, y)
        
        # SAM 모델 작업은 스레드 안전하게 처리
        with predictor_lock:
            # SAM 모델 가져오기
            predictor = get_predictor()
            
            # SAM 모델에 이미지 설정
            predictor.set_image(img_np)
            
            # 클릭 위치 프롬프트 설정
            input_point = np.array([[x, y]])
            input_label = np.array([1])  # 1은 전경(객체), 0은 배경
            
            # 마스크 예측 (다중 마스크 출력 활성화)
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True  # 여러 마스크 후보 생성
            )
        
        # 최적의 마스크 선택 (아이폰 스타일 선택 알고리즘)
        mask_idx = select_best_mask(masks, scores, img_np, x, y, is_near_edge, edge_strength)
        mask = masks[mask_idx]
        
        # 마스크를 PIL 이미지로 변환 (알파 채널용)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        
        # 마스크를 원본 이미지에 적용 (배경 투명화)
        img_pil = Image.fromarray(img_np)
        img_pil.putalpha(mask_img)
        return img_pil
        
    except Exception as e:
        logger.error(f"배경 제거 중 오류 발생: {str(e)}")
        # 오류 시 원본 이미지 반환 (RGBA 형식)
        return image.convert('RGBA') if image.mode != 'RGBA' else image