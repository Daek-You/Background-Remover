from flask import Flask
from flask_cors import CORS
from app.routes import api_bp
from config.environments import current_env
import os
import logging
from app.utils.logger import setup_logger, setup_werkzeug_logger
import threading

# 로거 설정
logger = setup_logger(__name__)

def init_models_async(app):
    """비동기적으로 모델을 초기화하는 함수"""
    with app.app_context():
        try:
            # 모델 관리자 초기화
            from app.core.model_manager import get_model_manager
            model_manager = get_model_manager()
            predictor = model_manager.get_predictor()
            logger.info("모델 초기화 완료")
            
            # 병렬 처리 관리자 초기화
            from app.core.parallel_executor import get_executor_manager
            executor_manager = get_executor_manager()
            executor_manager.get_executor()
            logger.info("병렬 처리 관리자 초기화 완료")
        except Exception as e:
            logger.error(f"모델 초기화 중 오류 발생: {str(e)}")

def configure_logging():
    """Flask 애플리케이션의 로깅 설정"""
    # Flask 기본 로거 비활성화
    flask_logger = logging.getLogger('flask')
    flask_logger.propagate = False
    
    # Werkzeug 로거 설정
    setup_werkzeug_logger()
    
    # 다른 라이브러리의 로거 레벨 설정
    logging.getLogger('PIL').setLevel(logging.WARNING)

def create_app():
    app = Flask(__name__, static_folder='../static')
    
    # Flask 로깅 설정
    configure_logging()
    
    # 환경 설정 적용
    app.config.update(current_env)
    
    # CORS 설정 - 환경 설정에서 가져온 origins 사용
    CORS(app, resources={r"/*": {"origins": current_env['CORS_ORIGINS']}})
    
    # 이미지 업로드 경로 설정 (정적 파일 제공용)
    upload_folder = os.path.join(app.root_path, '../static/images')
    app.config['UPLOAD_FOLDER'] = upload_folder
    
    # 폴더가 없으면 생성
    os.makedirs(upload_folder, exist_ok=True)
    
    # 모델 초기화를 별도 스레드에서 실행
    logger.info("모델 초기화 시작 (백그라운드)")
    init_thread = threading.Thread(target=init_models_async, args=(app,))
    init_thread.daemon = True  # 메인 스레드가 종료되면 같이 종료
    init_thread.start()
    
    # 라우트 등록을 위한 블루프린트 import 및 등록
    app.register_blueprint(api_bp, url_prefix='/bg-remover')
    return app