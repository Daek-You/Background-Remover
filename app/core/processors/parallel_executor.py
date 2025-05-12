"""
병렬 처리를 위한 스레드 풀 또는 프로세스 풀 관리 모듈
"""
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from app.utils.logger import setup_logger
from config.settings import THREAD_POOL

# 로거 설정
logger = setup_logger(__name__)

class ExecutorManager:
    """병렬 실행기 관리 클래스"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """싱글톤 패턴 구현"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ExecutorManager, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """초기화"""
        self.executor = None
        self.executor_lock = threading.Lock()
    
    def get_executor(self):
        """스레드/프로세스 풀 반환 (필요시 초기화, 스레드 안전)"""
        # 이미 초기화되었는지 빠르게 확인
        if self.executor is not None:
            return self.executor
        
        # 잠금 획득 후 다시 확인
        with self.executor_lock:
            if self.executor is None:
                if THREAD_POOL['USE_PROCESS_POOL']:
                    self.executor = ProcessPoolExecutor(max_workers=THREAD_POOL['MAX_WORKERS'])
                    logger.info(f"프로세스 풀 초기화 완료 (workers: {THREAD_POOL['MAX_WORKERS']})")
                else:
                    self.executor = ThreadPoolExecutor(max_workers=THREAD_POOL['MAX_WORKERS'])
                    logger.info(f"스레드 풀 초기화 완료 (workers: {THREAD_POOL['MAX_WORKERS']})")
        
        return self.executor
    
    def map(self, fn, *iterables):
        """풀을 사용하여 작업 병렬 처리"""
        executor = self.get_executor()
        return list(executor.map(fn, *iterables))
    
    def submit(self, fn, *args, **kwargs):
        """작업 제출"""
        executor = self.get_executor()
        return executor.submit(fn, *args, **kwargs)
    
    def shutdown(self):
        """실행기 종료"""
        if self.executor:
            self.executor.shutdown()
            self.executor = None

# 싱글톤 인스턴스 제공 함수
def get_executor_manager():
    return ExecutorManager() 