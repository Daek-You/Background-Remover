"""
배경 제거 관련 라우터
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import io
from PIL import Image
import asyncio
import json
from app.schemas.requests import RemoveBackgroundRequest
from app.services.background_removal.background_remover import BackgroundRemover
from app.core.model_management.model_manager import get_model_manager
from app.utils.logger import setup_logger

# 로거 설정
logger = setup_logger(__name__)

# 라우터 정의
router = APIRouter(
    prefix="/background-removal",
    tags=["background-removal"],
    responses={404: {"description": "Not found"}},
)

# 최대 동시 실행 개수를 제한하는 세마포어
MAX_CONCURRENT_TASKS = 2
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

@router.post("/remove")
async def remove_background(
    image: UploadFile = File(...),
    points: str = Form(...)
):
    """
    여러 클릭 좌표로 배경 제거
    
    이 API는 사용자가 이미지에서 클릭한 여러 좌표를 기반으로 해당 객체를 식별하고
    배경을 제거합니다. 복수의 클릭 좌표를 활용하여 더 정확한 객체 분리가 가능합니다.
    
    입력 파라미터:
    - image: 이미지 파일
    - points: JSON 형식의 좌표 배열 문자열 [[x1, y1], [x2, y2], [x3, y3], ...] 
              3개 이상의 점을 제공하는 것이 좋습니다.
    
    반환값:
    - 배경이 제거된 이미지 파일 (PNG, 투명 배경)
    """
    # 이미지 파일 검증
    if not image:
        raise HTTPException(status_code=400, detail="No image provided")
    
    # 클릭 좌표 파싱
    try:
        click_points_raw = json.loads(points)
        if not isinstance(click_points_raw, list) or len(click_points_raw) < 1:
            raise HTTPException(status_code=400, detail="Points must be a list of [x,y] coordinates")
        
        # 각 좌표가 [x, y] 형식인지 확인하고 정수형으로 변환
        click_points = []
        for point in click_points_raw:
            if not isinstance(point, list) or len(point) != 2:
                raise HTTPException(status_code=400, detail="Each point must be an [x,y] coordinate")
            try:
                x = int(point[0])
                y = int(point[1])
                click_points.append((x, y))
            except (ValueError, TypeError):
                raise HTTPException(status_code=400, detail="Coordinates must be integers")
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for points")
    
    async with semaphore:
        try:
            # 이미지 데이터 읽기
            contents = await image.read()
            img = Image.open(io.BytesIO(contents))
            
            # BackgroundRemover 서비스 호출
            result_img = await asyncio.to_thread(
                BackgroundRemover.remove_background,
                img, click_points, image.filename
            )
            
            # 결과 이미지를 메모리 스트림에 저장
            img_byte_arr = io.BytesIO()
            result_img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # 필요시 모델 메모리 정리
            if max(img.size) > 1000:
                model_manager = get_model_manager()
                await model_manager.cleanup_async()
                
            return StreamingResponse(
                content=img_byte_arr,
                media_type="image/png",
                headers={
                    "Content-Disposition": f"attachment; filename=result.png"
                }
            )
            
        except Exception as e:
            logger.error(f"요청 처리 중 오류 발생: {str(e)}")
            
            # 오류 발생 시 메모리 정리
            try:
                model_manager = get_model_manager()
                await model_manager.cleanup_async()
            except:
                pass
            
            raise HTTPException(status_code=500, detail=f"Request processing error: {str(e)}")
