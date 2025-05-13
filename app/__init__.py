""" FastAPI 애플리케이션 생성 및 초기화를 담당하는 모듈 """
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import time
from app.utils.logger import setup_logger
from config.environments import current_env

# 로거 설정
logger = setup_logger(__name__)

# 모델 초기화 상태를 관리하는 클래스
class ModelInitializer:
    """모델 초기화 상태를 체계적으로 관리하는 클래스"""
    
    def __init__(self):
        # 락 파일 경로를 명확하게 설정
        self.lock_file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'model_initialized.lock'
        )
    
    def is_already_initialized(self) -> bool:
        """모델이 이미 초기화되었는지 확인"""
        # 환경변수 체크
        if os.environ.get('APP_INITIALIZED') == '1':
            logger.debug("환경변수를 통해 모델 초기화 확인")
            return True
        
        # 락 파일 체크
        if os.path.exists(self.lock_file_path):
            logger.debug("락 파일을 통해 모델 초기화 확인")
            return True
        
        return False
    
    def mark_as_initialized(self):
        """모델 초기화 완료를 표시"""
        # 환경변수 설정
        os.environ['APP_INITIALIZED'] = '1'
        
        # 락 파일 생성 (프로세스 재시작 시에도 유지)
        try:
            with open(self.lock_file_path, 'w') as f:
                f.write(f"Initialized at: {time.time()}\n")
            logger.debug("모델 초기화 완료 표시됨")
        except Exception as e:
            logger.warning(f"락 파일 생성 실패: {str(e)}")
    
    def clear_initialization_state(self):
        """초기화 실패 시 상태 정리"""
        os.environ.pop('APP_INITIALIZED', None)  # 환경변수 제거
        try:
            if os.path.exists(self.lock_file_path):
                os.remove(self.lock_file_path)
        except Exception as e:
            logger.warning(f"락 파일 제거 실패: {str(e)}")

def initialize_models():
    """모델을 안전하게 초기화하는 함수"""
    model_init = ModelInitializer()
    
    # 이미 초기화되었으면 skip
    if model_init.is_already_initialized():
        logger.debug("모델이 이미 초기화되어 있습니다")
        return
    
    try:
        logger.info("모델 초기화를 시작합니다...")
        
        # 모델 관리자 초기화
        from app.core.model_management.model_manager import get_model_manager
        model_manager = get_model_manager()
        predictor = model_manager.get_predictor()
        
        # 병렬 처리 관리자 초기화
        from app.core.processors.parallel_executor import get_executor_manager
        executor_manager = get_executor_manager()
        executor_manager.get_executor()
        
        # 초기화 완료 표시
        model_init.mark_as_initialized()
        logger.info("모델 초기화가 완료되었습니다")
        
    except Exception as e:
        logger.error(f"모델 초기화 실패: {str(e)}")
        # 실패 시 상태를 깔끔하게 정리
        model_init.clear_initialization_state()
        raise

def configure_cors(app: FastAPI):
    """CORS 미들웨어를 설정하는 함수"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=current_env['CORS_ORIGINS'],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=3600,
    )
    logger.info("CORS 설정이 완료되었습니다")

def configure_logging():
    """로깅 레벨을 조정하는 함수"""
    import logging
    
    # 외부 라이브러리 로거 레벨 조정 (너무 verbose한 로그 억제)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    
    logger.debug("로깅 설정이 완료되었습니다")

def create_app() -> FastAPI:
    """FastAPI 애플리케이션을 생성하고 설정하는 메인 함수"""
    # 1. FastAPI 인스턴스 생성
    app = FastAPI(
        title="Background Remover API",
        description="SAM 2.1 모델을 사용한 이미지 배경 제거 API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # 2. 기본 설정들 적용
    configure_logging()
    configure_cors(app)
    
    # 3. 정적 파일을 위한 디렉토리 생성
    upload_folder = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        'static/images'
    )
    os.makedirs(upload_folder, exist_ok=True)
    logger.debug(f"업로드 폴더 확인/생성: {upload_folder}")
    
    # 4. 모델 초기화 (어플리케이션 시작 시 한 번만 실행)
    initialize_models()
    
    # 5. 라우터 등록
    from app.routes import router
    app.include_router(router)
    
    logger.info("FastAPI 애플리케이션 생성이 완료되었습니다")
    return app