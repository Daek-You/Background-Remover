"""
SAM 모델 관리 메인 모듈 (각 기능별 모듈 결합)
"""
import threading
import torch
import asyncio
from contextlib import nullcontext

# 분리된 모듈들 import
from app.core.gpu_memory_manager import GPUMemoryManager
from app.core.negative_point_generator import NegativePointGenerator
from app.core.sam2_model_loader import SAM2ModelLoader

from app.utils.logger import setup_logger
from config.settings import MODEL, SAM2_OPTIONS

# 로거 설정
logger = setup_logger(__name__)

class ModelManager:
    """SAM 모델 관리 메인 클래스"""
    
    _instance = None
    _lock = threading.Lock()
    _asyncio_lock = None
    
    def __new__(cls):
        """싱글톤 패턴 구현"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
            
    def _initialize(self):
        """초기화"""
        self.predictor = None
        self.predictor_lock = threading.Lock()
        
        # 설정에서 정보 가져오기
        self.version_name = MODEL.get('VERSION_NAME', MODEL['VERSION'])
        self.model_type = MODEL['TYPE']
        
        # 양자화 설정
        self.use_mixed_precision = MODEL['USE_MIXED_PRECISION'] and MODEL['DEVICE'] == 'cuda'
        self.amp_dtype = MODEL['QUANTIZATION_DTYPE']
        
        logger.info(f"모델 관리자 초기화: {self.version_name} - {self.model_type}")
        logger.debug(f"양자화 모드: {'활성화' if self.use_mixed_precision else '비활성화'}")
    
    @property
    def asyncio_lock(self):
        """비동기 잠금 객체 반환"""
        if self._asyncio_lock is None:
            self._asyncio_lock = asyncio.Lock()
        return self._asyncio_lock
    
    def _is_predictor_loaded(self):
        """Predictor 로드 여부 확인"""
        return self.predictor is not None
    
    def _log_quantization_mode(self):
        """양자화 모드 로깅"""
        if self.use_mixed_precision:
            logger.info(f"양자화 모드: 활성화 (dtype: {self.amp_dtype})")
        else:
            logger.info("양자화 모드: 비활성화")
    
    def get_predictor(self):
        """SAM 모델 predictor 객체 반환"""
        # 빠른 체크
        if self._is_predictor_loaded():
            return self.predictor
        
        # 스레드 안전 로딩
        with self.predictor_lock:
            if self.predictor is None:
                try:
                    self._load_predictor()
                except Exception as e:
                    logger.error(f"모델 로드 중 오류 발생: {str(e)}")
                    self.predictor = None
                    raise
        
        return self.predictor
    
    def _load_predictor(self):
        """Predictor 로드 프로세스"""
        # 1. GPU 메모리 정리
        GPUMemoryManager.cleanup_gpu_memory()
        
        # 2. 로딩 상태 로깅
        logger.info(f"{self.version_name} 모델 로드 중... (장치: {MODEL['DEVICE']})")
        GPUMemoryManager.log_gpu_memory_status()
        self._log_quantization_mode()
        
        # 3. 모델 로드
        self.predictor = SAM2ModelLoader.load_model(self.model_type)
        
        # 4. 로드 후 메모리 상태 확인
        GPUMemoryManager.log_post_load_memory()
    
    async def get_predictor_async(self):
        """비동기 환경에서 predictor 객체 반환"""
        if self._is_predictor_loaded():
            return self.predictor
            
        async with self.asyncio_lock:
            return await asyncio.to_thread(self.get_predictor)
    
    def _prepare_input_points(self, image_np, point_coords, point_labels, kwargs):
        """입력 포인트 준비 (positive + negative points)"""
        # Negative points 처리
        negative_points = kwargs.get('negative_points')
        if negative_points is None and SAM2_OPTIONS['AUTO_NEGATIVE_POINTS']:
            # 클릭 좌표 기준으로 negative points 자동 생성
            x, y = point_coords[0] if len(point_coords) > 0 else (0, 0)
            negative_points = NegativePointGenerator.generate_negative_points(image_np, x, y)
        
        # Points 유효성 검증
        if negative_points is not None:
            NegativePointGenerator.validate_points(image_np, negative_points)
        
        # Points 결합
        return NegativePointGenerator.combine_points(point_coords, point_labels, negative_points)
    
    def _predict_with_context(self, predictor, all_points, all_labels):
        """양자화 컨텍스트에서 예측 수행"""
        amp_context = torch.cuda.amp.autocast(dtype=self.amp_dtype) if self.use_mixed_precision else nullcontext()
        with amp_context:
            return predictor.predict(
                point_coords=all_points,
                point_labels=all_labels,
                multimask_output=True,
                return_logits=True
            )
    
    def predict(self, image_np, point_coords, point_labels, **kwargs):
        """이미지에서 마스크 예측"""
        predictor = self.get_predictor()
        
        with self.predictor_lock:
            try:
                # 이미지 설정
                predictor.set_image(image_np)
                
                # 입력 포인트 준비
                all_points, all_labels = self._prepare_input_points(image_np, point_coords, point_labels, kwargs)
                
                # 예측 수행
                masks, scores, logits = self._predict_with_context(predictor, all_points, all_labels)
                        
            except torch.cuda.OutOfMemoryError:
                # GPU 메모리 부족 시 복구 처리
                logger.warning("GPU 메모리 부족. 메모리 정리 후 재시도합니다.")
                self._handle_gpu_memory_error()
                
                # 재시도
                predictor = self.get_predictor()
                predictor.set_image(image_np)
                masks, scores, logits = self._predict_with_context(predictor, all_points, all_labels)
        
        return masks, scores, logits
    
    def _handle_gpu_memory_error(self):
        """GPU 메모리 오류 처리"""
        GPUMemoryManager.cleanup_gpu_memory()
        self.predictor = None
    
    async def predict_async(self, image_np, point_coords, point_labels, **kwargs):
        """비동기 방식으로 이미지 마스크 예측"""
        return await asyncio.to_thread(
            self.predict, 
            image_np, point_coords, point_labels, **kwargs
        )
    
    def cleanup(self):
        """GPU 메모리 정리"""
        GPUMemoryManager.cleanup_gpu_memory()
    
    async def cleanup_async(self):
        """비동기 방식으로 GPU 메모리 정리"""
        await asyncio.to_thread(self.cleanup)

# 싱글톤 인스턴스 제공 함수
def get_model_manager():
    return ModelManager()