from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
import time
from app.utils.logger import setup_logger
from config.environments import current_env

# 로거 설정
logger = setup_logger(__name__)

# 모델이 이미 초기화되었는지 확인하는 플래그
_model_initialized = False

# 모델 초기화 락 파일 경로
MODEL_LOCK_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model_initialized.lock')

def init_models(app):
    """모델을 초기화하는 함수"""
    global _model_initialized
    
    # 이미 초기화되었으면 다시 초기화하지 않음
    if _model_initialized or os.path.exists(MODEL_LOCK_FILE) or os.environ.get('APP_INITIALIZED') == '1':
        logger.debug("모델이 이미 초기화되어 있습니다.")
        return
    
    # 초기화 시작 표시
    os.environ['APP_INITIALIZED'] = '1'
    
    try:
        # 모델 관리자 초기화
        from app.core.model_manager import get_model_manager
        model_manager = get_model_manager()
        predictor = model_manager.get_predictor()
        
        # 병렬 처리 관리자 초기화
        from app.core.parallel_executor import get_executor_manager
        executor_manager = get_executor_manager()
        executor_manager.get_executor()
        
        # 초기화 완료 표시
        with open(MODEL_LOCK_FILE, 'w') as f:
            f.write(f"Initialized: {time.time()}")
        _model_initialized = True
    except Exception as e:
        logger.error(f"모델 초기화 중 오류 발생: {str(e)}")
        # 초기화 실패 시 플래그 초기화
        os.environ['APP_INITIALIZED'] = '0'

def configure_logging():
    """애플리케이션의 로깅 설정"""
    # 다른 라이브러리의 로거 레벨 설정
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('uvicorn').setLevel(logging.INFO)

def create_app():
    app = FastAPI(
        title="Background Remover API",
        description="SAM 모델을 사용한 이미지 배경 제거 API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # 로깅 설정
    configure_logging()
    
    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=current_env['CORS_ORIGINS'],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=3600,
    )
    
    # 이미지 업로드 경로 설정
    upload_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static/images')
    os.makedirs(upload_folder, exist_ok=True)
    
    # 모델 초기화 (동기적으로 실행)
    if not (os.environ.get('APP_INITIALIZED') == '1' or os.path.exists(MODEL_LOCK_FILE)):
        # main.py에서 로깅을 이미 했으므로 여기서는 최소한의 로그만 출력
        init_models(app)
    
    # 라우트 등록
    from app.routes import router
    app.include_router(router)
    
    return app