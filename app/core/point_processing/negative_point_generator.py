""" Negative Points 생성 전담 모듈 """
import numpy as np
from app.utils.logger import setup_logger
from config.settings import SAM2_OPTIONS

logger = setup_logger(__name__)     # 로거 설정


class NegativePointGenerator:
    """Negative Points 생성 전담 클래스"""
    
    @staticmethod
    def generate_negative_points(image_np, x, y):
        """
        SAM 2.1용 negative points 자동 생성
        
        Args:
            image_np: 이미지 numpy 배열
            x: 클릭한 x 좌표
            y: 클릭한 y 좌표
            
        Returns:
            numpy.ndarray: negative points 배열 또는 None
        """
        if not SAM2_OPTIONS['AUTO_NEGATIVE_POINTS']:
            return None
        
        h, w = image_np.shape[:2]
        margin = SAM2_OPTIONS['NEGATIVE_POINT_MARGIN']
        
        # 이미지 모서리에 negative points 추가
        negative_points = [
            [margin, margin],           # 좌상단
            [w - margin, margin],       # 우상단
            [margin, h - margin],       # 좌하단
            [w - margin, h - margin],   # 우하단
        ]
        
        logger.debug(f"자동 생성된 negative points: {len(negative_points)}개")
        return np.array(negative_points)
    
    @staticmethod
    def combine_points(point_coords, point_labels, negative_points):
        """
        Positive와 negative points 결합
        
        Args:
            point_coords: positive points 좌표
            point_labels: positive points 라벨
            negative_points: negative points 좌표
            
        Returns:
            tuple: (결합된_좌표, 결합된_라벨)
        """
        if negative_points is not None:
            all_points = np.vstack([point_coords, negative_points])
            all_labels = np.hstack([point_labels, np.zeros(len(negative_points))])
            logger.debug(f"Points 결합 완료: positive={len(point_coords)}, negative={len(negative_points)}")
            return all_points, all_labels
        else:
            logger.debug("Negative points 없음, positive points만 사용")
            return point_coords, point_labels
    
    @staticmethod
    def validate_points(image_np, points):
        """
        Points가 이미지 범위 내에 있는지 검증
        
        Args:
            image_np: 이미지 numpy 배열
            points: 검증할 points
            
        Returns:
            bool: 유효하면 True
        """
        if points is None or len(points) == 0:
            return True
        
        h, w = image_np.shape[:2]
        
        # 모든 점이 이미지 범위 내에 있는지 확인
        valid = np.all((points[:, 0] >= 0) & (points[:, 0] < w) & 
                      (points[:, 1] >= 0) & (points[:, 1] < h))
        
        if not valid:
            logger.warning("일부 points가 이미지 범위를 벗어났습니다.")
        
        return valid