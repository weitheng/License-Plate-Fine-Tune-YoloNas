import os
import requests
import zipfile
import logging
import coloredlogs
from tqdm import tqdm
from typing import Optional, List, Dict

# Setup minimal logging for this module
def setup_download_logging():
    """Setup logging with colored output for download utilities"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logger = logging.getLogger('download_utils')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Add success level
    logging.addLevelName(25, 'SUCCESS')
    def success(self, message, *args, **kwargs):
        if self.isEnabledFor(25):
            self._log(25, message, args, **kwargs)
    logging.Logger.success = success
    
    # Install colored logs
    coloredlogs.install(
        level='INFO',
        logger=logger,
        fmt=log_format,
        level_styles={
            'info': {'color': 'white'},
            'warning': {'color': 'yellow'},
            'error': {'color': 'red', 'bold': True},
            'success': {'color': 'green', 'bold': True}
        },
        field_styles={
            'asctime': {'color': 'cyan'},
            'levelname': {'color': 'magenta', 'bold': True}
        }
    )
    
    return logger

logger = setup_download_logging()

def download_model_weights(model_name: str, target_path: str) -> bool:
    """Download model weights from alternative sources if primary fails"""
    urls = {
        'YOLO_NAS_L': [
            'https://sg-hub-nv.s3.amazonaws.com/models/yolo_nas_l_coco.pth',
            'https://storage.googleapis.com/super-gradients-models/yolo_nas_l_coco.pth'
        ],
        'YOLO_NAS_S': [
            'https://sg-hub-nv.s3.amazonaws.com/models/yolo_nas_s_coco.pth',
            'https://storage.googleapis.com/super-gradients-models/yolo_nas_s_coco.pth'
        ]
    }
    
    if model_name not in urls:
        logger.error(f"Unknown model name: {model_name}")
        return False
        
    for url in urls.get(model_name, []):
        try:
            logger.info(f"Attempting to download from: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                logger.success(f"Successfully downloaded {model_name}")
                return True
        except Exception as e:
            logger.error(f"Failed to download from {url}: {e}")
    return False

def download_coco_subset(target_dir, num_images=85000):
    """Download a subset of COCO dataset"""
    try:
        # Create directories for COCO structure
        os.makedirs(os.path.join(target_dir, 'coco', 'images', 'train2017'), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'coco', 'images', 'val2017'), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'coco', 'annotations'), exist_ok=True)

        # URLs for both images and annotations
        urls = {
            'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
            'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
            'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        }
        
        for name, url in urls.items():
            zip_path = os.path.join(target_dir, f'coco_{name}.zip')
            if not os.path.exists(zip_path):
                try:
                    logger.info(f"Downloading COCO {name}...")
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    total_size = int(response.headers.get('content-length', 0))
                    
                    with open(zip_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                        for data in response.iter_content(chunk_size=8192):
                            f.write(data)
                            pbar.update(len(data))
                    
                    logger.info(f"Extracting {name}...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        if name == 'annotations':
                            # Extract annotations directly to annotations directory
                            zip_ref.extractall(os.path.join(target_dir, 'coco'))
                        else:
                            # For images, we need to ensure they go into the images directory
                            for zip_info in zip_ref.filelist:
                                if zip_info.filename.endswith('.jpg'):
                                    # Extract to the correct images subdirectory
                                    zip_info.filename = os.path.join('images', zip_info.filename)
                                    zip_ref.extract(zip_info, os.path.join(target_dir, 'coco'))
                        
                except Exception as e:
                    logger.error(f"Error downloading {name}: {e}")
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
                    return False
                
                # Verify extraction
                if name == 'train_images':
                    img_dir = os.path.join(target_dir, 'coco', 'images', 'train2017')
                    if not os.path.exists(img_dir) or not os.listdir(img_dir):
                        logger.error(f"Failed to extract training images to {img_dir}")
                        return False
                elif name == 'val_images':
                    img_dir = os.path.join(target_dir, 'coco', 'images', 'val2017')
                    if not os.path.exists(img_dir) or not os.listdir(img_dir):
                        logger.error(f"Failed to extract validation images to {img_dir}")
                        return False
                
        logger.success("✓ COCO dataset downloaded and extracted successfully")
        return True
    except Exception as e:
        logger.error(f"Error in COCO dataset download: {e}")
        return False
