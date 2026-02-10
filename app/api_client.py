# AI Attribution: This file was developed with assistance from Claude (Anthropic).
# https://claude.ai

import requests
from .config import Config


class ModelAPIError(Exception):
    """Raised when the model API returns an error or is unreachable."""
    pass


def check_model_health():
    """
    Ping the model API health endpoint.
    Returns True if healthy, False otherwise.
    """
    if not Config.MODEL_API_HEALTH_URL:
        return False
    
    try:
        resp = requests.get(
            Config.MODEL_API_HEALTH_URL,
            timeout=5
        )
        return resp.status_code == 200
    except requests.RequestException:
        return False


def get_prediction(image_file):
    """
    Send an image to the model API and get a prediction.
    
    Args:
        image_file: A file-like object (from request.files)
    
    Returns:
        dict with keys: prediction, confidence, probabilities, 
        processing_time_ms, model_version, heatmap (optional)
    
    Raises:
        ModelAPIError: If the API request fails or returns an error
    """
    if not Config.MODEL_API_URL:
        raise ModelAPIError("Model API URL not configured")
    
    try:
        # Reset file pointer in case it was read before
        image_file.seek(0)
        
        files = {'image': (image_file.filename, image_file, image_file.content_type)}
        headers = {}
        
        if Config.MODEL_API_KEY:
            headers['Authorization'] = f'Bearer {Config.MODEL_API_KEY}'
        
        resp = requests.post(
            Config.MODEL_API_URL,
            files=files,
            headers=headers,
            timeout=Config.API_TIMEOUT_SECONDS
        )
        
        if resp.status_code != 200:
            # Try to get error message from response
            try:
                error_data = resp.json()
                msg = error_data.get('error', f'API returned status {resp.status_code}')
            except ValueError:
                msg = f'API returned status {resp.status_code}'
            raise ModelAPIError(msg)
        
        data = resp.json()
        
        # Validate response has required fields
        if 'prediction' not in data or 'confidence' not in data:
            raise ModelAPIError("Invalid response format from model API")
        
        return data
        
    except requests.Timeout:
        raise ModelAPIError("Analysis is taking longer than expected. Please try again.")
    except requests.ConnectionError:
        raise ModelAPIError("Unable to connect to analysis service. Please check your connection.")
    except requests.RequestException as e:
        raise ModelAPIError(f"Request failed: {str(e)}")
