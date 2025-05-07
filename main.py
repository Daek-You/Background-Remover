import os
import subprocess
import sys
from config.environments import current_env
from utils.logger import setup_logger

# 로거 설정
logger = setup_logger(__name__)

def install_dependencies():
    """requirements.txt에 있는 패키지들이 설치되어 있는지 확인하고 없으면 설치합니다."""
    try:
        requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
        logger.info("의존성 설치 및 확인이 완료되었습니다.")
    except Exception as e:
        logger.error(f"의존성 설치 중 오류 발생: {e}")

# 앱을 임포트하기 전에 의존성 설치
install_dependencies()

# 의존성 설치 후, 앱 임포트
from app import create_app
app = create_app()

if __name__ == "__main__":
    app.run(
        host=current_env['HOST'],
        port=current_env['PORT'],
        debug=current_env['DEBUG']
    )
    logger.info("서버가 시작되었습니다.")
    logger.info(f"서버 주소: http://{current_env['HOST']}:{current_env['PORT']}")
    logger.info(f"환경: {current_env['ENV']}")
