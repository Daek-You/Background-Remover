"""
SAM 모델 관리 및 예측 처리를 담당하는 모듈
"""
import threading
import gc
import torch
import asyncio
from contextlib import nullcontext
from segment_anything import SamPredictor, sam_model_registry
from app.utils.model_utils import download_sam_model
from app.utils.logger import setup_logger
from config.settings import MODEL

# 로거 설정
logger = setup_logger(__name__)

class ModelManager:
    """SAM 모델 관리 클래스"""
    
    _instance = None
    _lock = threading.Lock()
    _asyncio_lock = None  # 비동기 컨텍스트에서 사용할 잠금
    
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
        
        # 양자화 설정
        self.use_mixed_precision = MODEL['USE_MIXED_PRECISION'] and MODEL['DEVICE'] == 'cuda'
        self.amp_dtype = MODEL['QUANTIZATION_DTYPE']
        
    @property
    def asyncio_lock(self):
        """비동기 잠금 객체 반환"""
        if self._asyncio_lock is None:
            self._asyncio_lock = asyncio.Lock()
        return self._asyncio_lock
    
    def get_predictor(self):
        """SAM 모델 predictor 객체 반환 (필요시 초기화, 스레드 안전)"""
        # 이미 초기화되었는지 빠르게 확인
        if self.predictor is not None:
            return self.predictor
        
        # 잠금 획득 후 다시 확인
        with self.predictor_lock:
            if self.predictor is None:
                try:
                    # GPU 메모리 확보를 위한 가비지 컬렉션 실행
                    if MODEL['DEVICE'] == 'cuda':
                        logger.info("GPU 메모리 정리 완료")
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    # 모델 파일 확인 및 다운로드
                    model_path = download_sam_model(model_type=MODEL['TYPE'])
                    
                    # 기기 정보 로깅
                    logger.info(f"SAM 모델 로드 중... (장치: {MODEL['DEVICE']})")
                    if MODEL['DEVICE'] == 'cuda':
                        logger.info(f"GPU 모델: {torch.cuda.get_device_name(0)}")
                        logger.info(f"가용 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
                        logger.info(f"할당 메모리: {torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024:.2f} GB")
                        logger.info(f"예약 메모리: {torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024:.2f} GB")
                    
                    # 양자화 모드 로깅
                    if self.use_mixed_precision:
                        logger.info(f"양자화 모드 활성화 (dtype: {self.amp_dtype})")
                    else:
                        logger.info("양자화 모드 비활성화 (FP32 사용)")
                    
                    # 양자화 모드에서 모델 로드
                    amp_context = torch.cuda.amp.autocast(dtype=self.amp_dtype) if self.use_mixed_precision else nullcontext()
                    with amp_context:
                        # SAM 모델 로드
                        sam = sam_model_registry[MODEL['TYPE']](checkpoint=model_path)
                        sam.to(device=MODEL['DEVICE'])  # GPU로 모델 이동
                        self.predictor = SamPredictor(sam)
                    
                    # 모델 로드 후 메모리 사용량
                    if MODEL['DEVICE'] == 'cuda':
                        logger.info(f"모델 로드 후 GPU 메모리 사용량: {torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024:.2f} GB")
                except Exception as e:
                    logger.error(f"모델 로드 중 오류 발생: {str(e)}")
                    raise
        
        return self.predictor
    
    async def get_predictor_async(self):
        """비동기 환경에서 predictor 객체 반환"""
        # 이미 초기화되었는지 빠르게 확인
        if self.predictor is not None:
            return self.predictor
            
        # 비동기 잠금 획득 (다른 스레드에서 이미 초기화 중인지 확인)
        async with self.asyncio_lock:
            # 동기 초기화 함수를 별도 스레드에서 실행
            return await asyncio.to_thread(self.get_predictor)
    
    def predict(self, image_np, point_coords, point_labels):
        """이미지에서 마스크 예측"""
        predictor = self.get_predictor()
        
        with self.predictor_lock:
            try:
                predictor.set_image(image_np)
                
                # 양자화 컨텍스트 설정
                amp_context = torch.cuda.amp.autocast(dtype=self.amp_dtype) if self.use_mixed_precision else nullcontext()
                with amp_context:
                    masks, scores, logits = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True
                    )
            except torch.cuda.OutOfMemoryError:
                # GPU 메모리 부족 시 메모리 정리 후 재시도
                logger.warning("GPU 메모리 부족. 메모리 정리 후 재시도합니다.")
                gc.collect()
                torch.cuda.empty_cache()
                
                # predictor 초기화 및 재로드
                self.predictor = None
                predictor = self.get_predictor()
                predictor.set_image(image_np)
                
                # 예측 재시도
                amp_context = torch.cuda.amp.autocast(dtype=self.amp_dtype) if self.use_mixed_precision else nullcontext()
                with amp_context:
                    masks, scores, logits = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True
                    )
        
        return masks, scores, logits
    
    async def predict_async(self, image_np, point_coords, point_labels):
        """비동기 방식으로 이미지 마스크 예측"""
        # 예측 작업은 CPU/GPU 집약적이므로 별도 스레드에서 실행
        return await asyncio.to_thread(
            self.predict, 
            image_np, point_coords, point_labels
        )
    
    def cleanup(self):
        """GPU 메모리 정리"""
        if MODEL['DEVICE'] == 'cuda':
            gc.collect()
            torch.cuda.empty_cache()
    
    async def cleanup_async(self):
        """비동기 방식으로 GPU 메모리 정리"""
        await asyncio.to_thread(self.cleanup)

# 싱글톤 인스턴스 제공 함수
def get_model_manager():
    return ModelManager() 