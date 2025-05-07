from utils.logger import setup_logger
from config.environments import current_env, ENV

# 로거 설정
logger = setup_logger(__name__)

# GPU 사용 가능 여부 확인
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 의존성 설치 후, 앱 임포트
from app import create_app
app = create_app()

if __name__ == "__main__":
    logger.info("서버를 시작합니다...")
    logger.info(f"서버 주소: http://{current_env['HOST']}:{current_env['PORT']}")
    logger.info(f"환경: {ENV}")

    app.run(
        host=current_env['HOST'],
        port=current_env['PORT'],
        debug=current_env['DEBUG']
    )
