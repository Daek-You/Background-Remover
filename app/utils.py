import os
import requests
import tqdm
from pathlib import Path

def download_sam_model(model_type="vit_h"):
    """SAM 모델 파일 다운로드 함수"""
    
    # 모델 유형에 따른 URL과 파일명 설정
    model_urls = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
    
    if model_type not in model_urls:
        raise ValueError(f"지원되지 않는 모델 유형: {model_type}")
    
    # 프로젝트 루트 디렉토리를 기준으로 models 폴더 경로 설정
    base_dir = Path(__file__).parent.parent  # app 폴더의 상위 디렉토리(프로젝트 루트)
    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True)
    
    model_filename = Path(model_urls[model_type].split("/")[-1])
    model_path = model_dir / model_filename
    
    # 모델 파일이 이미 존재하는지 확인
    if model_path.exists():
        print(f"모델 파일이 이미 존재합니다: {model_path}")
        return str(model_path)
    
    # 모델 다운로드
    print(f"SAM {model_type} 모델 다운로드 중... 이 작업은 몇 분 정도 소요될 수 있습니다.")
    
    response = requests.get(model_urls[model_type], stream=True)
    response.raise_for_status()
    
    # 파일 크기 확인 및 진행률 표시 설정
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1MB
    
    with open(model_path, 'wb') as f:
        with tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc=f"다운로드 중: {model_type}") as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"모델 다운로드 완료: {model_path}")
    return str(model_path)

def save_image(img, folder, filename=None):
    """이미지를 저장하고 저장된 경로를 반환합니다."""
    if filename is None:
        filename = f"result_{os.urandom(4).hex()}.png"
    
    # 폴더가 없으면 생성
    os.makedirs(folder, exist_ok=True)
    
    # 전체 파일 경로
    file_path = os.path.join(folder, filename)
    
    # 이미지 저장
    img.save(file_path)
    return file_path