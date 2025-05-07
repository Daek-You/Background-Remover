import threading
import cv2
import numpy as np
from PIL import Image
import torch
from contextlib import nullcontext
from segment_anything import SamPredictor, sam_model_registry
from utils.model_utils import download_sam_model
from utils.logger import setup_logger
from config.settings import IMAGE_ANALYSIS, MASK_SELECTION, MODEL
from concurrent.futures import ThreadPoolExecutor
import gc

# 로거 설정
logger = setup_logger(__name__)

# 전역 변수
predictor = None
predictor_lock = threading.Lock()

# 최대 처리 이미지 크기
MAX_IMAGE_SIZE = 1024

def get_predictor():
    """predictor 객체 반환 (필요시 초기화, 스레드 안전)"""
    global predictor
    
    # 이미 초기화되었는지 빠르게 확인
    if predictor is not None:
        return predictor
    
    # 잠금 획득 후 다시 확인
    with predictor_lock:
        if predictor is None:
            try:
                # GPU 메모리 확보를 위한 가비지 컬렉션 실행
                if MODEL['DEVICE'] == 'cuda':
                    gc.collect()
                    torch.cuda.empty_cache()
                    logger.info("GPU 메모리 정리 완료")
                
                # 모델 파일 확인 및 다운로드
                model_path = download_sam_model(model_type=MODEL['TYPE'])
                
                # 기기 정보 로깅
                logger.info(f"SAM 모델 로드 중... (장치: {MODEL['DEVICE']})")
                if MODEL['DEVICE'] == 'cuda':
                    logger.info(f"GPU 모델: {torch.cuda.get_device_name(0)}")
                    logger.info(f"가용 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
                    logger.info(f"할당 메모리: {torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024:.2f} GB")
                    logger.info(f"예약 메모리: {torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024:.2f} GB")
                
                # SAM 모델 로드
                sam = sam_model_registry[MODEL['TYPE']](checkpoint=model_path)
                sam.to(device=MODEL['DEVICE'])  # GPU로 모델 이동
                predictor = SamPredictor(sam)
                
                # 모델 로드 후 메모리 사용량
                if MODEL['DEVICE'] == 'cuda':
                    logger.info(f"모델 로드 후 GPU 메모리 사용량: {torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024:.2f} GB")
            except Exception as e:
                logger.error(f"모델 로드 중 오류 발생: {str(e)}")
                raise
    
    return predictor

def resize_image_if_needed(image: Image.Image) -> tuple[Image.Image, float]:
    """이미지가 너무 큰 경우 리사이징하고 스케일 비율 반환"""
    if max(image.size) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(image.size)
        new_size = tuple(int(dim * scale) for dim in image.size)
        return image.resize(new_size, Image.Resampling.LANCZOS), scale
    return image, 1.0

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
    edges = cv2.Canny(gray, IMAGE_ANALYSIS['CANNY_THRESHOLD_LOW'], IMAGE_ANALYSIS['CANNY_THRESHOLD_HIGH'])
    
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
    """ 마스크가 이미지의 에지와 얼마나 잘 일치하는지 분석 """
    # 마스크 경계 추출
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 마스크 경계 주변만 에지 검출
    mask_edge = np.zeros_like(img_np[:,:,0])
    cv2.drawContours(mask_edge, mask_contours, -1, 255, 1)
    
    # 마스크 경계 주변 영역만 추출
    kernel = np.ones((3,3), np.uint8)
    mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=2)
    roi = np.where(mask_dilated > 0)
    
    if len(roi[0]) == 0:
        return 0.0
    
    # 관심 영역만 에지 검출
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 
                     IMAGE_ANALYSIS['CANNY_THRESHOLD_LOW'],
                     IMAGE_ANALYSIS['CANNY_THRESHOLD_HIGH'])
    
    # 마스크 경계와 에지의 겹치는 정도를 관심 영역 부분만 계산
    overlap = np.logical_and(edges[roi] > 0, mask_edge[roi] > 0)
    edge_alignment_score = np.sum(overlap) / np.sum(mask_edge[roi] > 0)
    
    return edge_alignment_score

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
        edge_alignment = analyze_mask_edge_alignment(mask, img_np)
        return (i, score, edge_alignment)
    else:
        size_score = 1.0 - abs(50 - mask_percentage) / 45
        return (i, score, size_score)

def select_best_mask(masks, scores, img_np, x, y, is_near_edge, edge_strength=0):
    """ 여러 마스크 중 가장 적합한 마스크를 선택하는 함수 """
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
    
    with ThreadPoolExecutor() as executor:
        args = [(i, masks[i], score, img_np, x, y, is_near_edge, total_pixels) 
                for i, score in top_masks]
        results = list(executor.map(evaluate_mask, args))
    
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

def remove_background_from_image(image: Image.Image, x: int, y: int) -> Image.Image:
    """ 클릭한 위치의 객체 배경을 제거하는 함수 (아이폰 스타일) """
    global predictor
    
    try:
        # 이미지 리사이징 (필요한 경우)
        resized_image, scale = resize_image_if_needed(image)
        img_np = np.array(resized_image.convert("RGB"))
        logger.info("이미지 리사이징 완료")
        
        # 좌표 스케일 조정
        x = int(x * scale)
        y = int(y * scale)
        
        # 에지 검출 및 객체 분석
        is_near_edge, edge_strength = analyze_image_for_object(img_np, x, y)
        logger.info("이미지 분석 완료")
        
        # SAM 모델 작업은 스레드 안전하게 처리
        with predictor_lock:
            try:
                predictor = get_predictor()
                predictor.set_image(img_np)
                
                input_point = np.array([[x, y]])
                input_label = np.array([1])
                
                # GPU 메모리 최적화를 위한 설정
                with torch.cuda.amp.autocast() if MODEL['DEVICE'] == 'cuda' else nullcontext():
                    masks, scores, _ = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True
                    )
            except torch.cuda.OutOfMemoryError:
                # GPU 메모리 부족 시 메모리 정리 후 재시도
                logger.warning("GPU 메모리 부족. 메모리 정리 후 재시도합니다.")
                gc.collect()
                torch.cuda.empty_cache()
                
                # 전역 predictor 초기화
                predictor = None
                
                # 다시 predictor 획득 및 처리
                predictor = get_predictor()
                predictor.set_image(img_np)
                with torch.cuda.amp.autocast():
                    masks, scores, _ = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True
                    )
        
        logger.info("SAM 모델 예측 완료")
        
        # 최적의 마스크 선택
        mask_idx = select_best_mask(masks, scores, img_np, x, y, is_near_edge, edge_strength)
        mask = masks[mask_idx]
        logger.info("마스크 선택 완료")
        
        # 마스크를 PIL 이미지로 변환
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        
        # 원본 크기로 복원
        if scale != 1.0:
            mask_img = mask_img.resize(image.size, Image.Resampling.LANCZOS)
        
        # 마스크를 원본 이미지에 적용
        result = image.convert('RGBA')
        result.putalpha(mask_img)
        
        logger.info("배경 제거 완료")
        
        # GPU 메모리 정리 (대용량 이미지 처리 후)
        if MODEL['DEVICE'] == 'cuda' and max(image.size) > 1000:
            torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        logger.error(f"배경 제거 중 오류 발생: {str(e)}")
        # 모든 오류 발생 시 메모리 정리
        if MODEL['DEVICE'] == 'cuda':
            torch.cuda.empty_cache()
        return image.convert('RGBA') if image.mode != 'RGBA' else image