import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""
    
    MODEL_API_URL = os.getenv('MODEL_API_URL', '')
    MODEL_API_HEALTH_URL = os.getenv('MODEL_API_HEALTH_URL', '')
    MODEL_API_KEY = os.getenv('MODEL_API_KEY', '')
    
    API_TIMEOUT_SECONDS = int(os.getenv('API_TIMEOUT_SECONDS', '10'))
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '10'))
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    
    @classmethod
    def validate(cls):
        """Check that required config is present. Returns list of missing vars."""
        missing = []
        if not cls.MODEL_API_URL:
            missing.append('MODEL_API_URL')
        return missing
