""" SAM 모델 관리 및 예측 수행을 담당하는 모듈 """
import threading
import torch
import asyncio
from contextlib import nullcontext
from typing import Tuple, Optional

from app.core.model_management.gpu_memory_manager import GPUMemoryManager
from app.core.point_processing.negative_point_generator import NegativePointGenerator
from app.core.model_util.sam2_model_initializer import SAM2ModelInitializer

from app.utils.logger import setup_logger
from config.settings import MODEL, SAM2_OPTIONS, MODEL_PERFORMANCE, GPU_MEMORY

logger = setup_logger(__name__)

class ModelManager:
    """SAM 모델 관리 및 예측을 담당하는 싱글톤 클래스"""
    
    _instance: Optional['ModelManager'] = None
    _lock = threading.Lock()
    _asyncio_lock: Optional[asyncio.Lock] = None
    
    def __new__(cls) -> 'ModelManager':
        """싱글톤 패턴 구현"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """인스턴스 초기화"""
        self.predictor = None
        self.predictor_lock = threading.Lock()
        self.model_initializer = SAM2ModelInitializer()
        
        # 설정값 캐싱
        self.version_name = MODEL.get('VERSION_NAME', MODEL['VERSION'])
        self.model_type = MODEL['TYPE']
        
        # 양자화 설정 (정리된 설정에서 가져오기)
        self.use_mixed_precision = (
            MODEL_PERFORMANCE['USE_MIXED_PRECISION'] and 
            MODEL['DEVICE'] == 'cuda'
        )
        self.amp_dtype = MODEL_PERFORMANCE['QUANTIZATION_DTYPE']
        
        logger.info(f"모델 관리자 초기화: {self.version_name} - {self.model_type}")
    
    @property
    def asyncio_lock(self) -> asyncio.Lock:
        """비동기 잠금 객체 반환 (지연 초기화)"""
        if self._asyncio_lock is None:
            self._asyncio_lock = asyncio.Lock()
        return self._asyncio_lock
    
    def get_predictor(self):
        """SAM 모델 predictor 객체 반환 (스레드 안전)"""
        # 빠른 체크 (Double-checked locking pattern)
        if self.predictor is not None:
            return self.predictor
        
        with self.predictor_lock:
            if self.predictor is None:
                self._load_predictor()
        
        return self.predictor
    
    def _load_predictor(self):
        """Predictor 로드 및 초기화"""
        try:
            # 메모리 정리
            GPUMemoryManager.cleanup_gpu_memory()
            
            # 로딩 상태 로깅
            logger.info(f"{self.version_name} 모델 로드 중... (장치: {MODEL['DEVICE']})")
            GPUMemoryManager.log_gpu_memory_status()
            
            # 양자화 모드 로깅
            self._log_quantization_status()
            
            # 모델 로드
            self.predictor = self.model_initializer.initialize_model(self.model_type)
            
            # 로드 완료 상태 확인
            GPUMemoryManager.log_post_load_memory()
            
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            self.predictor = None
            raise
    
    def _log_quantization_status(self):
        """양자화 모드 상태 로깅"""
        if self.use_mixed_precision:
            logger.info(f"양자화 활성화 (dtype: {self.amp_dtype})")
        else:
            logger.info("양자화 비활성화")
    
    async def get_predictor_async(self):
        """비동기 환경에서 predictor 객체 반환"""
        if self.predictor is not None:
            return self.predictor
            
        async with self.asyncio_lock:
            return await asyncio.to_thread(self.get_predictor)
    
    def predict(self, image_np, point_coords, point_labels, **kwargs) -> Tuple:
        """이미지에서 마스크 예측"""
        predictor = self.get_predictor()
        
        with self.predictor_lock:
            try:
                # 입력 준비
                prepared_points, prepared_labels = self._prepare_input_points(
                    image_np, point_coords, point_labels, kwargs
                )
                
                # 예측 수행
                masks, scores, logits = self._execute_prediction(
                    predictor, image_np, prepared_points, prepared_labels
                )
                
                return masks, scores, logits
                
            except torch.cuda.OutOfMemoryError:
                # GPU 메모리 부족 시 복구
                return self._handle_memory_error(predictor, image_np, prepared_points, prepared_labels)
    
    def _prepare_input_points(self, image_np, point_coords, point_labels, kwargs):
        """입력 포인트 준비 (positive + negative points)"""
        # Negative points 처리
        negative_points = kwargs.get('negative_points')
        if negative_points is None and SAM2_OPTIONS['AUTO_NEGATIVE_POINTS']:
            x, y = point_coords[0] if len(point_coords) > 0 else (0, 0)
            negative_points = NegativePointGenerator.generate_negative_points(image_np, x, y)
        
        # Points 유효성 검증
        if negative_points is not None:
            NegativePointGenerator.validate_points(image_np, negative_points)
        
        # Points 결합
        return NegativePointGenerator.combine_points(point_coords, point_labels, negative_points)
    
    def _execute_prediction(self, predictor, image_np, points, labels):
        """양자화 컨텍스트에서 예측 수행"""
        # 이미지 설정
        predictor.set_image(image_np)
        
        # 양자화 컨텍스트 설정
        amp_context = (
            torch.cuda.amp.autocast(dtype=self.amp_dtype) 
            if self.use_mixed_precision 
            else nullcontext()
        )
        
        with amp_context:
            return predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
                return_logits=True
            )
    
    def _handle_memory_error(self, predictor, image_np, points, labels):
        """GPU 메모리 오류 처리 및 재시도"""
        logger.warning("GPU 메모리 부족. 메모리 정리 후 재시도합니다.")
        
        # 메모리 정리
        GPUMemoryManager.cleanup_gpu_memory()
        self.predictor = None
        
        # 재시도
        predictor = self.get_predictor()
        return self._execute_prediction(predictor, image_np, points, labels)
    
    async def predict_async(self, image_np, point_coords, point_labels, **kwargs):
        """비동기 방식으로 마스크 예측"""
        return await asyncio.to_thread(
            self.predict, 
            image_np, point_coords, point_labels, **kwargs
        )
    
    def cleanup(self):
        """GPU 메모리 정리"""
        if GPU_MEMORY['ENABLE_AUTO_CLEANUP']:
            GPUMemoryManager.cleanup_gpu_memory()
    
    async def cleanup_async(self):
        """비동기 방식으로 GPU 메모리 정리"""
        await asyncio.to_thread(self.cleanup)

# 싱글톤 인스턴스 제공 함수
def get_model_manager() -> ModelManager:
    """모델 관리자 싱글톤 인스턴스 반환"""
    return ModelManager()