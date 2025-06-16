import os
import shutil
import base64
from PIL import Image
import io


def save_uploaded_file(uploaded_file, upload_dir="uploads"):
    """Save uploaded file to disk and return the path."""
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def create_download_link(image, filename):
    """Create a download link for an image."""
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='JPEG', quality=95)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    
    href = f'<a href="data:image/jpeg;base64,{img_str}" download="{filename}">Download {filename}</a>'
    return href


def ensure_dir_exists(directory):
    """Ensure a directory exists, create if it doesn't."""
    os.makedirs(directory, exist_ok=True)


def get_file_size_mb(file_path):
    """Get file size in MB."""
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return round(size_mb, 2)


def validate_image(image_file):
    """Validate uploaded image file."""
    try:
        img = Image.open(image_file)
        img.verify()
        return True, "Valid image"
    except Exception as e:
        return False, f"Invalid image: {str(e)}"


def resize_image_if_needed(image, max_size=(2048, 2048)):
    """Resize image if it's too large."""
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


def clean_temp_files(directory, max_files=50):
    """Clean temporary files if directory has too many files."""
    if not os.path.exists(directory):
        return
    
    files = os.listdir(directory)
    if len(files) > max_files:
        # Sort by modification time and remove oldest files
        files_with_time = [(f, os.path.getmtime(os.path.join(directory, f))) for f in files]
        files_with_time.sort(key=lambda x: x[1])
        
        files_to_remove = files_with_time[:len(files) - max_files]
        for file_name, _ in files_to_remove:
            try:
                os.remove(os.path.join(directory, file_name))
            except OSError:
                pass
