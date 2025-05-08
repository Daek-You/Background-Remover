from flask import Blueprint, request, jsonify, send_file
from app.services.background_remover import BackgroundRemover
import io
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

    # 이미지 이름 받기
    image_name = file.filename

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

    # 서비스를 직접 호출하여 배경 제거
    result_img = BackgroundRemover.remove_background(img, x, y, image_name)

    # 이미지를 메모리 스트림에 저장
    img_byte_arr = io.BytesIO()
    result_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)  # 스트림 포인터를 처음으로 이동
    
    # 메모리에서 직접 이미지 반환 (파일로 저장하지 않음)
    return send_file(
        img_byte_arr,
        mimetype='image/png',
        as_attachment=True,
        download_name='result.png'
    )