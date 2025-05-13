""" 메인 라우터 - API 엔드포인트만 담당 """
import json
import io
import asyncio
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from typing import List, Tuple

from app.services.background_removal.background_remover import BackgroundRemover
from app.core.model_management.model_manager import get_model_manager
from app.utils.logger import setup_logger
from config.settings import API_CONFIG, LARGE_IMAGE_THRESHOLD

logger = setup_logger(__name__)
router = APIRouter()

MAX_CONCURRENT = API_CONFIG['MAX_CONCURRENT_REQUESTS']
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# === 입력 검증 함수들 ===
def parse_coordinates(points_json: str) -> List[Tuple[int, int]]:
    """JSON 형식의 좌표를 파싱하고 검증"""
    try:
        coordinates = json.loads(points_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="JSON 형식이 올바르지 않습니다")
    
    if not isinstance(coordinates, list) or len(coordinates) < 1:
        raise HTTPException(status_code=400, detail="최소 1개 이상의 좌표가 필요합니다")
    
    # 좌표 검증 및 변환
    valid_points = []
    for i, point in enumerate(coordinates):
        if not isinstance(point, list) or len(point) != 2:
            raise HTTPException(status_code=400, detail=f"좌표 #{i+1}이 잘못된 형식입니다")
        
        try:
            x, y = int(point[0]), int(point[1])
            valid_points.append((x, y))
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail=f"좌표 #{i+1}이 정수가 아닙니다")
    
    return valid_points

async def process_image(image: UploadFile) -> Image.Image:
    """업로드된 이미지를 PIL 객체로 변환"""
    if not image:
        raise HTTPException(status_code=400, detail="이미지가 제공되지 않았습니다")
    
    try:
        contents = await image.read()
        return Image.open(io.BytesIO(contents))
    except Exception as e:
        logger.error(f"이미지 읽기 실패: {str(e)}")
        raise HTTPException(status_code=400, detail="이미지 파일을 읽을 수 없습니다")

# === 응답 생성 함수 ===
def create_image_response(image: Image.Image) -> StreamingResponse:
    """PIL 이미지를 PNG 응답으로 변환"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    return StreamingResponse(
        content=buffer,
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=result.png"}
    )

# === 메모리 관리 함수 ===
async def cleanup_memory_if_needed(image_size: Tuple[int, int]):
    """대용량 이미지 처리 후 메모리 정리"""
    if max(image_size) > LARGE_IMAGE_THRESHOLD:
        try:
            model_manager = get_model_manager()
            await model_manager.cleanup_async()
            logger.debug("메모리 정리 완료")
        except Exception as e:
            logger.warning(f"메모리 정리 중 오류: {str(e)}")

# === API 엔드포인트들 ===
@router.get("/health")
async def health_check():
    """서비스 상태 확인"""
    return {"status": "healthy", "service": "background-remover"}

@router.post("/background-removal/remove")
async def remove_background(
    image: UploadFile = File(...),
    points: str = Form(...)
) -> StreamingResponse:
    """
    클릭한 여러 좌표를 기반으로 배경 제거
    - image: 업로드할 이미지 파일
    - points: JSON 배열 형식의 좌표들 [[x1,y1], [x2,y2], ...]
    
    클릭 좌표는 1개 이상 자유롭게 지정 가능합니다. 
    예) [[100, 200]] 또는 [[100, 200], [300, 400]] 등
    클릭 포인트가 많을수록 선택의 정확도가 향상될 수 있습니다.
    """
    async with semaphore:
        try:
            # 1. 입력 검증
            coordinates = parse_coordinates(points)
            img = await process_image(image)
            
            # 2. 배경 제거 실행
            result = await asyncio.to_thread(
                BackgroundRemover.remove_background,
                img, coordinates, image.filename
            )
            
            # 3. 응답 생성 및 정리
            response = create_image_response(result)
            await cleanup_memory_if_needed(img.size)
            
            return response
            
        except HTTPException:
            # 명확한 에러 메시지가 있는 경우 그대로 전파
            raise
        except Exception as e:
            logger.error(f"배경 제거 중 예상치 못한 오류: {str(e)}")
            
            # 에러 발생 시 메모리 정리
            try:
                model_manager = get_model_manager()
                await model_manager.cleanup_async()
            except:
                pass
            
            raise HTTPException(
                status_code=500, 
                detail="서버 내부 오류가 발생했습니다"
            )