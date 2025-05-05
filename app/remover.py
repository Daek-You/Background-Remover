import cv2
import numpy as np
from rembg import remove
from PIL import Image

# rembg 라이브러리를 사용하여 배경 제거 수행
def remove_background_from_image(image: Image.Image, x: int, y: int) -> Image.Image:
    # 원본 이미지 → numpy 배열
    img_np = np.array(image.convert("RGB"))
    
    # 마스크 초기화 (0: 배경, 1: 전경)
    mask = np.zeros(img_np.shape[:2], dtype=np.uint8)

    # 클릭 좌표 주변을 전경으로 설정 (반지름 15)
    cv2.circle(mask, (x, y), radius=15, color=1, thickness=-1)

    # 마스크를 기반으로 rembg가 내부적으로 U2NET 사용하도록 (Advanced: Custom preprocessing)
    # rembg는 기본적으로 마스크 입력을 받지 않지만, 내부 엔진을 직접 조정하거나 확장 가능
    # 따라서 다음과 같이 수동 처리 권장:

    # 임시 대체: 배경과 전경 마스크 기반으로 grabCut 사용 (OpenCV 기반)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # grabCut 수행
    cv2.grabCut(img_np, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    # 최종 마스크 생성 (0,2: 배경 / 1,3: 전경)
    result_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')

    # 마스크 적용
    result_img = cv2.bitwise_and(img_np, img_np, mask=result_mask)

    # 투명 배경 적용
    result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2RGBA)
    result_img[:, :, 3] = result_mask

    return Image.fromarray(result_img)
