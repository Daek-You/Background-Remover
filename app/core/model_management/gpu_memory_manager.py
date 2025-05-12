""" GPU 메모리 관리 전담 모듈 """
import gc
import torch
from app.utils.logger import setup_logger
from config.settings import MODEL

logger = setup_logger(__name__)     # 로거 설정


class GPUMemoryManager:
    """GPU 메모리 관리 전담 클래스"""
    @staticmethod
    def cleanup_gpu_memory():
        """GPU 메모리 정리"""
        if MODEL['DEVICE'] == 'cuda':
            logger.debug("GPU 메모리 정리 중...")
            gc.collect()
            torch.cuda.empty_cache()
            logger.debug("GPU 메모리 정리 완료")
    
    @staticmethod
    def log_gpu_memory_status():
        """GPU 메모리 상태 로깅"""
        if MODEL['DEVICE'] == 'cuda' and torch.cuda.is_available():
            logger.info(f"GPU 모델: {torch.cuda.get_device_name(0)}")
            logger.info(f"가용 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
            logger.info(f"할당 메모리: {torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024:.2f} GB")
            logger.info(f"예약 메모리: {torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024:.2f} GB")
    
    @staticmethod
    def log_post_load_memory():
        """모델 로드 후 메모리 상태 로깅"""
        if MODEL['DEVICE'] == 'cuda':
            logger.info(f"모델 로드 후 GPU 메모리 사용량: {torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024:.2f} GB")
    
    @staticmethod
    def is_cuda_available():
        """CUDA 사용 가능 여부 확인"""
        return MODEL['DEVICE'] == 'cuda' and torch.cuda.is_available()
    
    @staticmethod
    def get_available_memory_gb():
        """사용 가능한 GPU 메모리 (GB) 반환"""
        if not GPUMemoryManager.is_cuda_available():
            return 0.0
        
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        return (total_memory - allocated_memory) / (1024 ** 3)