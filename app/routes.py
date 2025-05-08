from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import io
from PIL import Image
import asyncio
from app.services.background_remover import BackgroundRemover
from app.utils.logger import setup_logger
from app.core.model_manager import get_model_manager

# 로거 설정
logger = setup_logger(__name__)

# 라우터 정의
router = APIRouter()

# 최대 동시 실행 개수를 제한하는 세마포어
# GPU 메모리를 고려하여 동시에 처리할 요청 수 제한
MAX_CONCURRENT_TASKS = 2  # GPU 메모리 사용량에 따라 조정 필요
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

# 요청 데이터 모델
class RemoveBackgroundRequest:
    def __init__(self, x: int, y: int, image_filename: str, image_data: bytes):
        self.x = x
        self.y = y
        self.image_filename = image_filename
        self.image_data = image_data

# 요청 처리 작업을 큐에 추가하고 실행 결과를 반환
async def process_background_removal(request: RemoveBackgroundRequest):
    """배경 제거 작업을 비동기적으로 처리"""
    img = Image.open(io.BytesIO(request.image_data))
    
    # 작업을 세마포어로 제한하면서 실행
    async with semaphore:
        try:
            # 모델 매니저의 비동기 메서드 활용
            model_manager = get_model_manager()
            # 비동기 to_thread를 통해 이미지 처리 (CPU/GPU 집약적 작업)
            result_img = await asyncio.to_thread(
                BackgroundRemover.remove_background,
                img, request.x, request.y, request.image_filename
            )
            
            # 결과 이미지를 메모리 스트림에 저장
            img_byte_arr = io.BytesIO()
            result_img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)  # 스트림 포인터를 처음으로 이동
            
            # 필요시 모델 메모리 정리
            if max(img.size) > 1000:  # 대용량 이미지인 경우
                await model_manager.cleanup_async()
                
            return img_byte_arr
        except Exception as e:
            logger.error(f"배경 제거 중 오류 발생: {str(e)}")
            await model_manager.cleanup_async()  # 오류 발생 시에도 메모리 정리
            raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

@router.post("/remove-background")
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
    
    try:
        # 이미지 데이터 읽기
        contents = await image.read()
        
        # 요청 객체 생성
        request = RemoveBackgroundRequest(x, y, image.filename, contents)
        
        # 배경 제거 처리
        img_byte_arr = await process_background_removal(request)
        
        # 스트리밍 응답으로 이미지 반환
        return StreamingResponse(
            content=img_byte_arr,
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=result.png"
            }
        )
        
    except Exception as e:
        logger.error(f"요청 처리 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Request processing error: {str(e)}")

@router.get("/health")
async def health_check():
    """서버 상태 확인을 위한 헬스체크 엔드포인트"""
    return {"status": "healthy", "service": "background-remover"}