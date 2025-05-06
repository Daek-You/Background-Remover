from flask import Blueprint, request, jsonify, send_from_directory, current_app
from app.remover import remove_background_from_image
from app.utils import save_image
import os
from PIL import Image

# 블루프린트 생성
api_bp = Blueprint('api', __name__)

@api_bp.route('/remove-background', methods=['POST'])
def remove_background():
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
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image provided"}), 400

    # 좌표 정보 받기
    x = request.form.get('x')
    y = request.form.get('y')

    if not x or not y:
        return jsonify({"error": "Coordinates not provided"}), 400

    try:
        x = int(x)
        y = int(y)
    except ValueError:
        return jsonify({"error": "Invalid coordinates"}), 400

    # 이미지 로드
    img = Image.open(file.stream)

    # 좌표 정보와 함께 배경 제거 함수 호출
    result_img = remove_background_from_image(img, x, y)

    # 결과 이미지 저장 후, 경로 반환
    result_path = save_image(result_img, current_app.config['UPLOAD_FOLDER'], filename='result.png')
    
    # 결과 이미지 파일을 응답으로 전송
    directory = os.path.dirname(result_path)
    filename = os.path.basename(result_path)
    return send_from_directory(directory, filename, as_attachment=True)