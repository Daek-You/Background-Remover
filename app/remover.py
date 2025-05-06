import threading
import cv2
import numpy as np
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from app.utils import download_sam_model

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
            model_path = download_sam_model(model_type="vit_h")
            
            # SAM 모델 로드
            print(f"SAM 모델 로드 중...")
            sam = sam_model_registry["vit_h"](checkpoint=model_path)
            predictor = SamPredictor(sam)
            print("SAM 모델 로드 완료!")
    
    return predictor

def analyze_image_for_object(img_np, x, y, radius=20):
    """
    클릭 위치 주변을 분석하여 객체 특성 파악
    
    Args:
        img_np: 이미지 NumPy 배열
        x, y: 클릭 좌표
        radius: 분석할 영역의 반경 (픽셀)
        
    Returns:
        is_near_edge: 클릭 위치가 에지(객체 경계)에 가까운지 여부
        edge_strength: 에지의 강도 (0~1 사이 값)
    """
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
    edges = cv2.Canny(gray, 50, 150)
    
    # 클릭 위치 주변 에지 확인 (5픽셀 이내)
    center_y, center_x = radius, radius
    nearby_region = edges[max(0, center_y-5):min(2*radius, center_y+5), 
                          max(0, center_x-5):min(2*radius, center_x+5)]
    
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
    # 이미지 에지 검출
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # 마스크 경계 추출
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 마스크 경계를 이미지에 그리기
    mask_edge = np.zeros_like(gray)
    cv2.drawContours(mask_edge, mask_contours, -1, 255, 1)
    
    # 마스크 경계와 이미지 에지의 겹치는 부분 계산
    overlap = np.logical_and(edges > 0, mask_edge > 0)
    if np.sum(mask_edge) == 0:
        return 0.0
    
    # 얼마나 많은 마스크 경계가 이미지 에지와 일치하는지 계산
    edge_alignment_score = np.sum(overlap) / np.sum(mask_edge > 0)
    return edge_alignment_score

def select_best_mask(masks, scores, img_np, x, y, is_near_edge, edge_strength=0):
    """
    여러 마스크 중 가장 적합한 마스크를 선택하는 함수
    
    아이폰의 주체 분리 기능과 유사한 결과를 얻기 위해 다음 기준을 적용합니다:
    1. 클릭한 좌표가 마스크에 포함되어야 함 (필수 조건)
    2. 점수(신뢰도)가 높은 마스크 우선
    3. 크기가 적절한 마스크 선호 (너무 크거나 작지 않은)
    4. 객체 경계와 일치하는 마스크 선호
    
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
    # 기본적으로 가장 높은 점수의 마스크 선택
    best_mask_idx = np.argmax(scores)
    highest_score = scores[best_mask_idx]
    
    # 클릭 위치에서 마스크가 활성화된 것만 유효한 마스크로 간주
    valid_masks = []
    for i, mask in enumerate(masks):
        if mask[y, x]:  # 클릭한 좌표에서 마스크가 True인지 확인
            valid_masks.append((i, scores[i]))
    
    # 유효한 마스크가 있으면, 그 중에서 가장 높은 점수의 마스크 선택
    if valid_masks:
        valid_masks.sort(key=lambda x: x[1], reverse=True)
        best_mask_idx = valid_masks[0][0]
    
    # 마스크 크기 및 에지 일치도 분석
    h, w = img_np.shape[:2]
    total_pixels = h * w
    
    # 클릭이 객체 경계에 가까운 경우와 아닌 경우 다른 전략 적용
    candidate_masks = []
    
    for i, mask in enumerate(masks):
        # 마스크가 클릭 위치를 포함하는지 확인 (필수 조건)
        if not mask[y, x]:
            continue
            
        # 마스크의 픽셀 수 계산 및 이미지 대비 비율 계산
        mask_pixels = np.sum(mask)
        mask_percentage = mask_pixels / total_pixels * 100
        
        # 마스크와 이미지 에지의 일치도 계산
        edge_alignment = analyze_mask_edge_alignment(mask, img_np)
        
        # 마스크 크기가 적절한지 확인 (5%~90%)
        size_ok = 5 <= mask_percentage <= 90
        # 점수가 충분히 높은지 확인 (최고 점수의 70% 이상)
        score_ok = scores[i] > highest_score * 0.7
        
        if size_ok and score_ok:
            # 경계에 가까운 클릭인 경우: 에지 일치도가 높은 마스크 선호
            if is_near_edge:
                # (마스크 인덱스, 점수, 에지 일치도)
                candidate_masks.append((i, scores[i], edge_alignment))
            # 경계에서 먼 클릭인 경우: 크기가 적당하고 점수가 높은 마스크 선호
            else:
                # 크기가 중간 정도인 마스크 선호 (너무 크거나 작지 않은)
                size_score = 1.0 - abs(50 - mask_percentage) / 45  # 50%에 가까울수록 1에 가까움
                # (마스크 인덱스, 점수, 크기 적절성)
                candidate_masks.append((i, scores[i], size_score))
    
    # 후보 마스크가 있으면, 전략에 따라 최적의 마스크 선택
    if candidate_masks:
        if is_near_edge:
            # 경계 근처 클릭: 에지 일치도와 점수를 모두 고려
            candidate_masks.sort(key=lambda x: (x[2] * 0.7 + x[1] * 0.3), reverse=True)
        else:
            # 객체 내부 클릭: 크기 적절성과 점수를 모두 고려
            candidate_masks.sort(key=lambda x: (x[2] * 0.5 + x[1] * 0.5), reverse=True)
        
        best_mask_idx = candidate_masks[0][0]
    
    return best_mask_idx

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
        print(f"Error: {e}")
        # 오류 시 원본 이미지 반환 (RGBA 형식)
        return image.convert('RGBA') if image.mode != 'RGBA' else image