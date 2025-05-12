"""
배경 제거 관련 라우터
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import io
from PIL import Image
import asyncio
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
    x: int = Form(...),
    y: int = Form(...)
):
    """
    단일 클릭 좌표로 배경 제거 (아이폰 스타일)
    
    이 API는 사용자가 이미지에서 클릭한 좌표를 기반으로 해당 객체를 식별하고
    배경을 제거합니다. 아이폰의 주체 분리 기능과 유사하게 단일 클릭만으로도
    정확하게 객체를 분리합니다.
    
    입력 파라미터:
    - image: 이미지 파일
    - x: 클릭한 x 좌표
    - y: 클릭한 y 좌표
    
    반환값:
    - 배경이 제거된 이미지 파일 (PNG, 투명 배경)
    """
    # 이미지 파일 검증
    if not image:
        raise HTTPException(status_code=400, detail="No image provided")
    
    async with semaphore:
        try:
            # 이미지 데이터 읽기
            contents = await image.read()
            img = Image.open(io.BytesIO(contents))
            
            # BackgroundRemover 서비스 호출
            result_img = await asyncio.to_thread(
                BackgroundRemover.remove_background,
                img, x, y, image.filename
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
