from app.utils.logger import setup_logger
from config.environments import current_env, ENV

# 로거 설정
logger = setup_logger(__name__)

# GPU 사용 가능 여부 확인
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# CUDA 정보 출력
if DEVICE == 'cuda':
    logger.info(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    logger.info(f"CUDA 버전: {torch.version.cuda}")
    logger.info(f"GPU 모델: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
else:
    logger.warning("CUDA를 사용할 수 없습니다. CPU 모드로 실행합니다.")

# 의존성 설치 후, 앱 임포트
from app import create_app
app = create_app()

if __name__ == "__main__":
    logger.info("서버를 시작합니다...")
    logger.info(f"서버 주소: http://{current_env['HOST']}:{current_env['PORT']}")
    logger.info(f"환경: {ENV}")
    logger.info(f"장치: {DEVICE}")

    app.run(
        host=current_env['HOST'],
        port=current_env['PORT'],
        debug=current_env['DEBUG']
    )
