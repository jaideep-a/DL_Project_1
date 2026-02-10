# AI Attribution: This file was developed with assistance from Claude (Anthropic).
# https://claude.ai

import os
from dotenv import load_dotenv

# load_dotenv() is called at module level so that environment variables
# are available when the Config class attributes are initialized below.
load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""

    # Model API Configuration
    MODEL_API_URL = os.getenv('MODEL_API_URL', '')
    MODEL_API_HEALTH_URL = os.getenv('MODEL_API_HEALTH_URL', '')
    MODEL_API_KEY = os.getenv('MODEL_API_KEY', '')

    # Supabase Configuration
    SUPABASE_URL = os.getenv('SUPABASE_URL', '')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')

    # File Storage
    API_TIMEOUT_SECONDS = int(os.getenv('API_TIMEOUT_SECONDS', '10'))
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '10'))
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    SESSION_LIFETIME_HOURS = int(os.getenv('SESSION_LIFETIME_HOURS', '24'))

    # Allowed file types
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

    # Mapbox Configuration (for surveillance dashboard)
    MAPBOX_ACCESS_TOKEN = os.getenv('MAPBOX_ACCESS_TOKEN', '')

    @classmethod
    def validate(cls):
        """Check that required config is present. Returns list of missing vars."""
        missing = []
        if not cls.MODEL_API_URL:
            missing.append('MODEL_API_URL')
        if not cls.SUPABASE_URL:
            missing.append('SUPABASE_URL')
        if not cls.SUPABASE_KEY:
            missing.append('SUPABASE_KEY')
        return missing
