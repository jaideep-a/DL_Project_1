# AI Attribution: This file was developed with assistance from Claude (Anthropic).
# https://claude.ai

import pytest
from io import BytesIO
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.utils import allowed_file, validate_file_size, generate_unique_filename, get_file_extension


class TestAllowedFile:
    """Tests for file extension validation."""
    
    def test_accepts_jpg(self):
        assert allowed_file('xray.jpg') is True
    
    def test_accepts_jpeg(self):
        assert allowed_file('xray.jpeg') is True
    
    def test_accepts_png(self):
        assert allowed_file('xray.png') is True
    
    def test_accepts_uppercase_extension(self):
        assert allowed_file('xray.JPG') is True
        assert allowed_file('xray.PNG') is True
    
    def test_rejects_pdf(self):
        assert allowed_file('document.pdf') is False
    
    def test_rejects_gif(self):
        assert allowed_file('animation.gif') is False
    
    def test_rejects_no_extension(self):
        assert allowed_file('noextension') is False
    
    def test_handles_multiple_dots(self):
        assert allowed_file('my.xray.image.jpg') is True


class TestValidateFileSize:
    """Tests for file size validation."""
    
    @patch('app.utils.Config')
    def test_accepts_small_file(self, mock_config):
        mock_config.MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
        
        # Create a small fake file
        fake_file = BytesIO(b'x' * 1000)
        
        assert validate_file_size(fake_file) is True
    
    @patch('app.utils.Config')
    def test_rejects_large_file(self, mock_config):
        mock_config.MAX_FILE_SIZE_BYTES = 1000  # 1KB limit for testing
        
        # Create a file larger than limit
        fake_file = BytesIO(b'x' * 2000)
        
        assert validate_file_size(fake_file) is False
    
    @patch('app.utils.Config')
    def test_resets_file_pointer(self, mock_config):
        mock_config.MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
        
        fake_file = BytesIO(b'test content')
        validate_file_size(fake_file)
        
        # File pointer should be back at start
        assert fake_file.tell() == 0


class TestGenerateUniqueFilename:
    """Tests for filename generation."""
    
    def test_preserves_extension(self):
        result = generate_unique_filename('test.jpg')
        assert result.endswith('.jpg')
    
    def test_sanitizes_dangerous_characters(self):
        result = generate_unique_filename('../../../etc/passwd.jpg')
        assert '..' not in result
        assert '/' not in result
    
    def test_adds_unique_suffix(self):
        result1 = generate_unique_filename('test.jpg')
        result2 = generate_unique_filename('test.jpg')
        assert result1 != result2
    
    def test_handles_empty_filename(self):
        result = generate_unique_filename('')
        assert result.endswith('.jpg')
        assert len(result) > 4


class TestGetFileExtension:
    """Tests for extension extraction."""
    
    def test_extracts_jpg(self):
        assert get_file_extension('image.jpg') == 'jpg'
    
    def test_extracts_lowercase(self):
        assert get_file_extension('IMAGE.JPG') == 'jpg'
    
    def test_handles_no_extension(self):
        assert get_file_extension('noext') == ''
    
    def test_handles_multiple_dots(self):
        assert get_file_extension('my.image.file.png') == 'png'
