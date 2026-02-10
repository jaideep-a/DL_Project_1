# AI Attribution: This file was developed with assistance from Claude (Anthropic).
# https://claude.ai

import pytest
from unittest.mock import Mock, patch
from io import BytesIO

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.api_client import get_prediction, check_model_health, ModelAPIError


class TestCheckModelHealth:
    """Tests for the health check function."""
    
    @patch('app.api_client.Config')
    @patch('app.api_client.requests.get')
    def test_returns_true_when_healthy(self, mock_get, mock_config):
        mock_config.MODEL_API_HEALTH_URL = 'http://api.test/health'
        mock_get.return_value = Mock(status_code=200)
        
        result = check_model_health()
        
        assert result is True
    
    @patch('app.api_client.Config')
    @patch('app.api_client.requests.get')
    def test_returns_false_on_non_200(self, mock_get, mock_config):
        mock_config.MODEL_API_HEALTH_URL = 'http://api.test/health'
        mock_get.return_value = Mock(status_code=503)
        
        result = check_model_health()
        
        assert result is False
    
    @patch('app.api_client.Config')
    def test_returns_false_when_url_not_configured(self, mock_config):
        mock_config.MODEL_API_HEALTH_URL = ''
        
        result = check_model_health()
        
        assert result is False


class TestGetPrediction:
    """Tests for the prediction API call."""
    
    def create_mock_file(self, filename='test.jpg', content=b'fake image data'):
        mock_file = Mock()
        mock_file.filename = filename
        mock_file.content_type = 'image/jpeg'
        mock_file.seek = Mock()
        mock_file.read = Mock(return_value=content)
        return mock_file
    
    @patch('app.api_client.Config')
    @patch('app.api_client.requests.post')
    def test_successful_prediction(self, mock_post, mock_config):
        mock_config.MODEL_API_URL = 'http://api.test/predict'
        mock_config.MODEL_API_KEY = ''
        mock_config.API_TIMEOUT_SECONDS = 10
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'prediction': 'PNEUMONIA',
            'confidence': 0.87,
            'probabilities': {'NORMAL': 0.13, 'PNEUMONIA': 0.87}
        }
        mock_post.return_value = mock_response
        
        result = get_prediction(self.create_mock_file())
        
        assert result['prediction'] == 'PNEUMONIA'
        assert result['confidence'] == 0.87
    
    @patch('app.api_client.Config')
    def test_raises_error_when_url_not_configured(self, mock_config):
        mock_config.MODEL_API_URL = ''
        
        with pytest.raises(ModelAPIError) as exc_info:
            get_prediction(self.create_mock_file())
        
        assert 'not configured' in str(exc_info.value)
    
    @patch('app.api_client.Config')
    @patch('app.api_client.requests.post')
    def test_handles_timeout(self, mock_post, mock_config):
        import requests
        mock_config.MODEL_API_URL = 'http://api.test/predict'
        mock_config.MODEL_API_KEY = ''
        mock_config.API_TIMEOUT_SECONDS = 10
        mock_post.side_effect = requests.Timeout()
        
        with pytest.raises(ModelAPIError) as exc_info:
            get_prediction(self.create_mock_file())
        
        assert 'taking longer than expected' in str(exc_info.value)
    
    @patch('app.api_client.Config')
    @patch('app.api_client.requests.post')
    def test_handles_connection_error(self, mock_post, mock_config):
        import requests
        mock_config.MODEL_API_URL = 'http://api.test/predict'
        mock_config.MODEL_API_KEY = ''
        mock_config.API_TIMEOUT_SECONDS = 10
        mock_post.side_effect = requests.ConnectionError()
        
        with pytest.raises(ModelAPIError) as exc_info:
            get_prediction(self.create_mock_file())
        
        assert 'Unable to connect' in str(exc_info.value)
    
    @patch('app.api_client.Config')
    @patch('app.api_client.requests.post')
    def test_handles_api_error_response(self, mock_post, mock_config):
        mock_config.MODEL_API_URL = 'http://api.test/predict'
        mock_config.MODEL_API_KEY = ''
        mock_config.API_TIMEOUT_SECONDS = 10
        
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {'error': 'Model failed to process image'}
        mock_post.return_value = mock_response
        
        with pytest.raises(ModelAPIError) as exc_info:
            get_prediction(self.create_mock_file())
        
        assert 'Model failed to process image' in str(exc_info.value)
