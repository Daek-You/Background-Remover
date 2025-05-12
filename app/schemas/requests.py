"""
API 요청/응답 스키마 정의
"""

class RemoveBackgroundRequest:
    """배경 제거 요청 데이터 클래스"""
    
    def __init__(self, x: int, y: int, image_filename: str, image_data: bytes):
        self.x = x
        self.y = y
        self.image_filename = image_filename
        self.image_data = image_data
    
    def __repr__(self):
        return f"RemoveBackgroundRequest(x={self.x}, y={self.y}, filename={self.image_filename})"