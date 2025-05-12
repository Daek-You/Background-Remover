from pathlib import Path
import config

def get_project_root() -> Path:
    """프로젝트 루트 경로 반환 (settings.py 기준)"""
    return Path(config.__file__).parent.parent.resolve() 