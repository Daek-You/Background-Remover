#!/usr/bin/env python
"""
양자화(FP16) 모드와 일반(FP32) 모드의 성능 및 결과 비교 테스트 스크립트
"""

import sys
import os

# 모듈 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import numpy as np
import torch
import argparse
from PIL import Image, ImageChops
import importlib
from app.utils.logger import setup_logger
import glob

# 로거 설정
logger = setup_logger(__name__)

def load_app_with_setting(use_mixed_precision):
    """양자화 설정을 변경하고 모듈 다시 로드"""
    # 먼저 settings 모듈 수정
    from config import settings
    settings.MODEL['USE_MIXED_PRECISION'] = use_mixed_precision
    
    # 모듈 리로드
    import app.core.model_manager
    import app.services.background_remover
    importlib.reload(app.core.model_manager)
    importlib.reload(app.services.background_remover)
    
    # 리로드 후 서비스 가져오기
    from app.services.background_remover import BackgroundRemover
    
    # 로깅
    precision_mode = "FP16 (양자화)" if use_mixed_precision else "FP32 (일반)"
    logger.info(f"모드 변경 완료: {precision_mode}")
    
    return BackgroundRemover.remove_background

def test_quantization(image_path, x, y, save_results=True):
    """양자화 모드 켜고 끈 상태에서 성능 및 결과 비교"""
    # 이미지 로드
    img = Image.open(image_path)
    logger.info(f"이미지 크기: {img.size}")
    
    results = {}
    
    # 1. FP32 모드 테스트
    logger.info("\n===== FP32 모드 (양자화 비활성화) =====")
    remove_background_fp32 = load_app_with_setting(False)
    
    # GPU 메모리 초기화
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # 처리 시간 측정
    start_time = time.time()
    result_fp32 = remove_background_fp32(img, x, y, os.path.basename(image_path))
    elapsed_time_fp32 = time.time() - start_time
    
    # 메모리 사용량 측정
    if torch.cuda.is_available():
        gpu_mem_fp32 = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
    else:
        gpu_mem_fp32 = 0
    
    logger.info(f"처리 시간: {elapsed_time_fp32:.2f}초")
    logger.info(f"최대 GPU 메모리 사용량: {gpu_mem_fp32:.2f} GB")
    
    if save_results:
        fp32_path = f"result_fp32_{os.path.basename(image_path).split('.')[0]}.png"
        result_fp32.save(fp32_path)
        logger.info(f"FP32 결과 저장됨: {fp32_path}")
    
    results['fp32'] = {
        'time': elapsed_time_fp32,
        'memory': gpu_mem_fp32,
        'image': result_fp32
    }
    
    # 2. FP16 모드 테스트
    logger.info("\n===== FP16 모드 (양자화 활성화) =====")
    remove_background_fp16 = load_app_with_setting(True)
    
    # GPU 메모리 초기화
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # 처리 시간 측정
    start_time = time.time()
    result_fp16 = remove_background_fp16(img, x, y, os.path.basename(image_path))
    elapsed_time_fp16 = time.time() - start_time
    
    # 메모리 사용량 측정
    if torch.cuda.is_available():
        gpu_mem_fp16 = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
    else:
        gpu_mem_fp16 = 0
    
    logger.info(f"처리 시간: {elapsed_time_fp16:.2f}초")
    logger.info(f"최대 GPU 메모리 사용량: {gpu_mem_fp16:.2f} GB")
    
    if save_results:
        fp16_path = f"result_fp16_{os.path.basename(image_path).split('.')[0]}.png"
        result_fp16.save(fp16_path)
        logger.info(f"FP16 결과 저장됨: {fp16_path}")
    
    results['fp16'] = {
        'time': elapsed_time_fp16,
        'memory': gpu_mem_fp16,
        'image': result_fp16
    }
    
    # 3. 결과 비교
    logger.info("\n===== 결과 비교 =====")
    
    # 시간 및 메모리 비교
    time_diff = elapsed_time_fp32 - elapsed_time_fp16
    time_improvement = (time_diff / elapsed_time_fp32) * 100
    
    memory_diff = gpu_mem_fp32 - gpu_mem_fp16
    memory_improvement = (memory_diff / gpu_mem_fp32) * 100 if gpu_mem_fp32 > 0 else 0
    
    logger.info(f"시간 차이: {time_diff:.2f}초 ({time_improvement:.1f}% 개선)")
    logger.info(f"메모리 차이: {memory_diff:.2f} GB ({memory_improvement:.1f}% 절약)")
    
    # 이미지 차이 분석
    if save_results:
        # 알파 채널만 추출하여 비교 (마스크 차이)
        alpha_fp32 = result_fp32.getchannel('A')
        alpha_fp16 = result_fp16.getchannel('A')
        
        # 차이 계산
        diff_image = ImageChops.difference(alpha_fp32, alpha_fp16)
        diff_path = f"diff_{os.path.basename(image_path).split('.')[0]}.png"
        diff_image.save(diff_path)
        logger.info(f"차이 이미지 저장됨: {diff_path}")
        
        # 정량적 차이 계산
        diff_pixels = np.count_nonzero(np.array(diff_image))
        total_pixels = alpha_fp32.width * alpha_fp32.height
        diff_percentage = (diff_pixels / total_pixels) * 100
        
        logger.info(f"마스크 차이: {diff_pixels} 픽셀 ({diff_percentage:.2f}% 차이)")
    
    return results

def test_all_images_in_folder(folder_path='images', save_results=True):
    """폴더 내의 모든 이미지에 대해 테스트 실행"""
    # 이미지 파일 찾기
    image_files = glob.glob(os.path.join(folder_path, '*.jpg')) + glob.glob(os.path.join(folder_path, '*.png'))
    
    if not image_files:
        logger.error(f"{folder_path} 폴더에 이미지 파일이 없습니다.")
        return
    
    # 결과 저장용 변수
    all_results = {
        'fp32': {'total_time': 0, 'avg_memory': 0},
        'fp16': {'total_time': 0, 'avg_memory': 0}
    }
    
    logger.info(f"총 {len(image_files)}개 이미지에 대해 테스트를 시작합니다.")
    
    # 각 이미지별 테스트
    for i, image_path in enumerate(image_files):
        logger.info(f"\n[{i+1}/{len(image_files)}] {os.path.basename(image_path)} 테스트 중...")
        
        # 이미지 크기 확인하여 중심점 계산
        with Image.open(image_path) as img:
            width, height = img.size
            x, y = width // 2, height // 2
        
        # 테스트 실행
        results = test_quantization(image_path, x, y, save_results)
        
        # 결과 누적
        all_results['fp32']['total_time'] += results['fp32']['time']
        all_results['fp32']['avg_memory'] += results['fp32']['memory']
        all_results['fp16']['total_time'] += results['fp16']['time']
        all_results['fp16']['avg_memory'] += results['fp16']['memory']
    
    # 평균 계산
    all_results['fp32']['avg_memory'] /= len(image_files)
    all_results['fp16']['avg_memory'] /= len(image_files)
    
    # 최종 결과 요약
    logger.info("\n===== 전체 결과 요약 =====")
    logger.info(f"총 테스트 이미지: {len(image_files)}개")
    
    time_diff = all_results['fp32']['total_time'] - all_results['fp16']['total_time']
    time_improvement = (time_diff / all_results['fp32']['total_time']) * 100
    
    memory_diff = all_results['fp32']['avg_memory'] - all_results['fp16']['avg_memory']
    memory_improvement = (memory_diff / all_results['fp32']['avg_memory']) * 100 if all_results['fp32']['avg_memory'] > 0 else 0
    
    logger.info(f"FP32 총 처리 시간: {all_results['fp32']['total_time']:.2f}초")
    logger.info(f"FP16 총 처리 시간: {all_results['fp16']['total_time']:.2f}초")
    logger.info(f"시간 차이: {time_diff:.2f}초 ({time_improvement:.1f}% 개선)")
    
    logger.info(f"FP32 평균 메모리 사용량: {all_results['fp32']['avg_memory']:.2f} GB")
    logger.info(f"FP16 평균 메모리 사용량: {all_results['fp16']['avg_memory']:.2f} GB")
    logger.info(f"메모리 차이: {memory_diff:.2f} GB ({memory_improvement:.1f}% 절약)")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='양자화 모드 성능 및 결과 비교 테스트')
    parser.add_argument('--image', type=str, help='테스트할 단일 이미지 경로 (지정하지 않으면 폴더 내 모든 이미지 테스트)')
    parser.add_argument('--x', type=int, help='클릭할 x 좌표 (단일 이미지 테스트시 필요)')
    parser.add_argument('--y', type=int, help='클릭할 y 좌표 (단일 이미지 테스트시 필요)')
    parser.add_argument('--folder', type=str, default='images', help='테스트할 이미지 폴더 경로')
    parser.add_argument('--no-save', action='store_true', help='결과 이미지를 저장하지 않음')
    
    args = parser.parse_args()
    
    # 시스템 정보 출력
    logger.info(f"CPU 코어 수: {os.cpu_count()}")
    
    # GPU 정보 출력
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA 버전: {torch.version.cuda}")
    else:
        logger.info("GPU 사용 불가: CPU 모드로 실행")
        logger.warning("양자화는 GPU 모드에서만 효과가 있습니다.")
    
    # 단일 이미지 또는 전체 폴더 테스트
    if args.image:
        if args.x is None or args.y is None:
            parser.error("단일 이미지 테스트 시 --x와 --y 좌표가 필요합니다.")
        test_quantization(args.image, args.x, args.y, not args.no_save)
    else:
        test_all_images_in_folder(args.folder, not args.no_save)

if __name__ == "__main__":
    main() 