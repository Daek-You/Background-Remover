import cv2
import numpy as np
from rembg import remove
from PIL import Image

def preprocess_image(image: Image.Image) -> np.ndarray:
    """이미지를 전처리하여 numpy 배열로 변환합니다."""
    img_np = np.array(image.convert("RGB"))
    blurred = cv2.GaussianBlur(img_np, (5, 5), 0)
    return img_np, blurred

def create_flood_fill_mask(img: np.ndarray, x: int, y: int) -> np.ndarray:
    """클릭한 지점을 기준으로 플러드필 마스크를 생성합니다."""
    h, w = img.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # 플러드필 설정
    flood_fill_flags = (
        4 |                             # 4방향 연결성 사용
        cv2.FLOODFILL_FIXED_RANGE |     # 고정 범위 사용
        cv2.FLOODFILL_MASK_ONLY |       # 마스크만 채우기
        (1 << 8)                        # 새로운 값
    )
    
    # 클릭한 지점에서 플러드필 수행
    loDiff = (20, 20, 20)  # 색상 차이 하한값
    upDiff = (20, 20, 20)  # 색상 차이 상한값
    cv2.floodFill(
        img, flood_mask, (x, y), 255,
        loDiff=loDiff, upDiff=upDiff,
        flags=flood_fill_flags
    )
    
    # 플러드필 마스크에서 패딩 제거
    return flood_mask[1:-1, 1:-1]

def apply_grabcut(img_np: np.ndarray, initial_mask: np.ndarray) -> np.ndarray:
    """GrabCut 알고리즘을 사용하여 객체 분할을 수행합니다."""
    # GrabCut 모델 초기화
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # 복사본 생성 (원본 데이터 보존)
    mask = initial_mask.copy()
    
    # GrabCut 수행
    cv2.grabCut(img_np, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    
    # 최종 마스크 생성 (0,2: 배경 / 1,3: 전경)
    return np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')

def filter_biggest_contour_at_point(mask: np.ndarray, x: int, y: int) -> np.ndarray:
    """클릭한 지점이 포함된 가장 큰 컨투어만 유지합니다."""
    # 컨투어 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 클릭 지점을 포함하는 컨투어 찾기
    containing_contours = []
    for contour in contours:
        if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
            containing_contours.append(contour)
    
    # 클릭 지점을 포함하는 컨투어 중 가장 큰 것 선택
    if containing_contours:
        max_contour = max(containing_contours, key=cv2.contourArea)
        filtered_mask = np.zeros_like(mask)
        cv2.drawContours(filtered_mask, [max_contour], 0, 255, -1)
        return filtered_mask
    
    return mask

def create_transparent_image(img_np: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """마스크를 적용하여 투명 배경의 이미지를 생성합니다."""
    # RGBA 이미지로 변환
    result_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2RGBA)
    
    # 알파 채널에 마스크 적용
    result_img[:, :, 3] = mask
    
    return result_img

def remove_background_from_image(image: Image.Image, x: int, y: int) -> Image.Image:
    """클릭한 객체를 식별하여 배경을 제거합니다."""
    # 1. 이미지 전처리
    img_np, blurred = preprocess_image(image)
    
    # 2. 초기 마스크 생성
    mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
    mask[:] = 2  # 처음에는 모든 픽셀을 '가능한 배경'으로 설정
    
    # 3. 플러드필 마스크 생성
    flood_mask = create_flood_fill_mask(blurred, x, y)
    
    # 4. 플러드필로 찾은 영역을 전경으로 설정
    mask[flood_mask == 1] = 1
    
    # 5. GrabCut으로 객체 분할
    result_mask = apply_grabcut(img_np, mask)
    
    # 6. 클릭 지점 포함된 최대 컨투어 필터링
    result_mask = filter_biggest_contour_at_point(result_mask, x, y)
    
    # 7. 투명 배경 적용
    result_img = create_transparent_image(img_np, result_mask)
    
    return Image.fromarray(result_img)
