from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.conf import settings
import os

def save_file(path: str, content: bytes) -> str:
    """Save a file to the storage backend."""
    content_file = ContentFile(content)
    return default_storage.save(path, content_file)

def file_exists(path: str) -> bool:
    """Check if a file exists in the storage backend."""
    return default_storage.exists(path)

def load_file(path: str) -> bytes:
    """Load a file's content from the storage backend."""
    if not default_storage.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with default_storage.open(path, "rb") as f:
        return f.read()

def get_media_path(*parts) -> str:
    """Get full path in 'mediafiles/' directory using given parts."""
    return os.path.join("mediafiles", *parts)
