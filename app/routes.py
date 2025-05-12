"""
메인 라우터 - 모든 서브 라우터를 통합
"""
from fastapi import APIRouter
from app.routers.background_removal import router as background_removal_router

# 메인 라우터 생성
router = APIRouter()

# 서브 라우터들 포함
router.include_router(background_removal_router)

# 전체 서비스 상태 확인
@router.get("/health")
async def health_check():
    """서비스 상태 확인을 위한 헬스체크 엔드포인트"""
    return {"status": "healthy", "service": "background-remover"}