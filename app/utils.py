# AI Attribution: This file was developed with assistance from Claude (Anthropic).
# https://claude.ai

import os
import uuid
from werkzeug.utils import secure_filename
from .config import Config


def allowed_file(filename):
    """Check if file has an allowed extension."""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in Config.ALLOWED_EXTENSIONS


def validate_file_size(file):
    """
    Check if file is under the max size limit.
    Returns True if valid, False if too large.
    """
    # Seek to end to get size
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)  # Reset for later use
    
    return size <= Config.MAX_FILE_SIZE_BYTES


def generate_unique_filename(original_filename):
    """
    Create a safe, unique filename preserving the original extension.
    """
    filename = secure_filename(original_filename)
    if not filename:
        filename = 'upload'
    
    name, ext = os.path.splitext(filename)
    if not ext:
        ext = '.jpg'
    
    unique_id = uuid.uuid4().hex[:8]
    return f"{name}_{unique_id}{ext}"


def get_file_extension(filename):
    """Get lowercase file extension without the dot."""
    if '.' in filename:
        return filename.rsplit('.', 1)[1].lower()
    return ''
