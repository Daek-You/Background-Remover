from flask import Flask
from flask_cors import CORS
from app.routes import api_bp
import os


def create_app():
    app = Flask(__name__, static_folder='../static')
    
    # CORS 설정
    CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
    
    # 이미지 업로드/결과 저장 경로 설정
    upload_folder = os.path.join(app.root_path, '../static/images')
    app.config['UPLOAD_FOLDER'] = upload_folder
    
    # 폴더가 없으면 생성
    os.makedirs(upload_folder, exist_ok=True)
    
    # 라우트 등록을 위한 블루프린트 import 및 등록
    app.register_blueprint(api_bp, url_prefix='/bg-remover')
    return app