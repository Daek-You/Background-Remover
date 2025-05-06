import os
import subprocess
import sys

def install_dependencies():
    """requirements.txt에 있는 패키지들이 설치되어 있는지 확인하고 없으면 설치합니다."""
    try:
        requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
        print("의존성 설치 완료 또는 이미 설치되어 있습니다.")
    except Exception as e:
        print(f"의존성 설치 중 오류 발생: {e}")

# 앱을 임포트하기 전에 의존성 설치
install_dependencies()

# 의존성 설치 후, 앱 임포트
from app import create_app
app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
