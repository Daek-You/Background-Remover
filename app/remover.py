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

def remove_background_from_image(image: Image.Image, x: int, y: int) -> Image.Image:
    """클릭한 위치의 객체 배경을 제거하는 함수"""
    try:
        # 이미지를 numpy 배열로 변환
        img_np = np.array(image.convert("RGB"))
        
        # SAM 모델 작업은 스레드 안전하게 처리
        with predictor_lock:
            # SAM 모델 가져오기
            predictor = get_predictor()
            
            # SAM 모델에 이미지 설정
            predictor.set_image(img_np)
            
            # 클릭 위치 프롬프트 설정
            input_point = np.array([[x, y]])
            input_label = np.array([1])  # 1은 전경(객체), 0은 배경
            
            # 마스크 예측
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
        
        # 가장 높은 점수의 마스크 선택
        mask_idx = np.argmax(scores)
        mask = masks[mask_idx]
        
        # 마스크를 PIL 이미지로 변환
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        
        # 마스크를 원본 이미지에 적용
        img_pil = Image.fromarray(img_np)
        img_pil.putalpha(mask_img)
        return img_pil
        
    except Exception as e:
        print(f"Error: {e}")
        # 오류 시 원본 이미지 반환
        return image.convert('RGBA') if image.mode != 'RGBA' else image