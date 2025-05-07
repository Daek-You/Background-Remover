from flask import Flask
from flask_cors import CORS
from app.routes import api_bp
from config.environments import current_env
import os
from utils.logger import setup_logger

# 로거 설정
logger = setup_logger(__name__)

# PyTorch 보안 경고 패치 적용
def apply_patches():
    from app.patch_sam import apply_torch_load_patch
    if apply_torch_load_patch():
        logger.info("SAM 모델 패치가 성공적으로 적용되었습니다.")
    else:
        logger.warning("SAM 모델 패치 적용에 실패했습니다. 경고 메시지가 표시될 수 있습니다.")

def create_app():
    # 패치 적용
    apply_patches()
    
    app = Flask(__name__, static_folder='../static')
    
    # 환경 설정 적용
    app.config.update(current_env)
    
    # CORS 설정 - 환경 설정에서 가져온 origins 사용
    CORS(app, resources={r"/*": {"origins": current_env['CORS_ORIGINS']}})
    
    # 이미지 업로드 경로 설정 (정적 파일 제공용)
    upload_folder = os.path.join(app.root_path, '../static/images')
    app.config['UPLOAD_FOLDER'] = upload_folder
    
    # 폴더가 없으면 생성
    os.makedirs(upload_folder, exist_ok=True)
    
    # 모델 초기화 - 앱 생성 시 한 번만 실행
    with app.app_context():
        from app.remover import get_predictor
        get_predictor()
    
    # 라우트 등록을 위한 블루프린트 import 및 등록
    app.register_blueprint(api_bp, url_prefix='/bg-remover')
    return app