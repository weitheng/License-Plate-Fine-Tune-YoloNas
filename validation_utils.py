import os
import logging
import coloredlogs
import torch
from typing import Optional
from download_utils import download_model_weights
from yolo_training_utils import verify_checkpoint

def setup_logging():
    """Setup logging with colored output for terminal and file output"""
    # Format for both file and terminal
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Setup file handler
    file_handler = logging.FileHandler('training.log')
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Get the root logger and clear any existing handlers
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Add the file handler
    logger.addHandler(file_handler)
    
    # Install colored logs for terminal
    coloredlogs.install(
        level='INFO',
        logger=logger,
        fmt=log_format,
        level_styles={
            'info': {'color': 'white'},
            'warning': {'color': 'yellow'},
            'error': {'color': 'red', 'bold': True},
            'success': {'color': 'green', 'bold': True}  # For checkmarks
        },
        field_styles={
            'asctime': {'color': 'cyan'},
            'levelname': {'color': 'magenta', 'bold': True}
        }
    )
    
    return logger

# Add success level for checkmarks
logging.addLevelName(25, 'SUCCESS')
def success(self, message, *args, **kwargs):
    if self.isEnabledFor(25):
        self._log(25, message, args, **kwargs)
logging.Logger.success = success

logger = setup_logging()
if not logger:
    raise RuntimeError("Failed to initialize logger")

def validate_training_prerequisites(combined_dir: str, checkpoint_dir: str, export_dir: str, 
                                 l_model_path: str, s_model_path: str):
    """Validate all prerequisites before training"""
    logger.info("Validating training prerequisites...")
    
    # Check dataset paths
    if not os.path.exists(combined_dir):
        raise RuntimeError(f"Dataset directory not found: {combined_dir}")
    
    # Validate model weights with is_model_weights=True
    if not verify_checkpoint(l_model_path, is_model_weights=True):
        logger.warning(f"YOLO-NAS-L weights not found or invalid, attempting to download...")
        if not download_model_weights('YOLO_NAS_L', l_model_path):
            raise RuntimeError("Failed to obtain valid YOLO-NAS-L weights")
            
    if not verify_checkpoint(s_model_path, is_model_weights=True):
        logger.warning(f"YOLO-NAS-S weights not found or invalid, attempting to download...")
        if not download_model_weights('YOLO_NAS_S', s_model_path):
            raise RuntimeError("Failed to obtain valid YOLO-NAS-S weights")
    
    # Check write permissions
    for dir_path in [checkpoint_dir, export_dir]:
        if not os.access(dir_path, os.W_OK):
            raise PermissionError(f"No write permission for directory: {dir_path}")
    
    logger.success("✓ All prerequisites validated")

def verify_dataset_structure(data_dir: str) -> None:
    """Verify complete dataset structure before training"""
    required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for dir_path in required_dirs:
        full_path = os.path.join(data_dir, dir_path)
        if not os.path.exists(full_path):
            raise RuntimeError(f"Missing required directory: {full_path}")
        if not os.listdir(full_path):
            raise RuntimeError(f"Empty directory: {full_path}")

def validate_image_paths(data_dir: str) -> None:
    """Validate that all image files referenced in labels exist"""
    logger.info("Validating image paths...")
    
    for split in ['train', 'val']:
        images_dir = os.path.join(data_dir, f'images/{split}')
        labels_dir = os.path.join(data_dir, f'labels/{split}')
        
        # Get all image files
        image_files = {f.lower() for f in os.listdir(images_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))}
        
        missing_images = []
        
        # Check each label file's corresponding image
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue
                
            # Get base name without extension
            base_name = os.path.splitext(label_file)[0]
            
            # Check for image with different extensions
            image_found = False
            for ext in ['.jpg', '.jpeg', '.png']:
                possible_image = f"{base_name}{ext}".lower()
                if possible_image in image_files:
                    image_found = True
                    # Verify file is actually readable
                    img_path = os.path.join(images_dir, possible_image)
                    try:
                        with open(img_path, 'rb') as f:
                            # Try to read first few bytes
                            f.read(1024)
                    except Exception as e:
                        logger.error(f"Cannot read image file {img_path}: {e}")
                        missing_images.append(possible_image)
                    break
            
            if not image_found:
                missing_images.append(f"{base_name}.*")
        
        if missing_images:
            raise RuntimeError(
                f"Missing or unreadable images in {split} split:\n" + 
                "\n".join(missing_images[:10]) +
                f"\n... and {len(missing_images) - 10} more" if len(missing_images) > 10 else ""
            )
        
        logger.success(f"✓ All {len(image_files)} images in {split} split are valid and readable")
