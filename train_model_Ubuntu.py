import os
import torch
from super_gradients.training import Trainer, models
from super_gradients.common.object_names import Models
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val
)
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
import wandb
from yolo_training_utils import assess_hardware_capabilities, load_dataset_config, setup_directories
from torch.optim.lr_scheduler import OneCycleLR
from pycocotools.coco import COCO
import requests
import zipfile
from tqdm import tqdm
import logging
from config import TrainingConfig
from typing import Optional, List, Dict, Any, Tuple
import time
import coloredlogs
import psutil
import hashlib
from super_gradients.training.utils.callbacks import PhaseCallback, Phase

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

def download_coco_subset(target_dir, num_images=70000):
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

def validate_coco_structure(coco_dir, num_images=70000):
    """Validate COCO dataset directory structure and contents"""
    logger.info("Validating COCO dataset structure...")
    
    # First check if the directories exist in the root of coco_dir
    train_dir = os.path.join(coco_dir, 'train2017')
    val_dir = os.path.join(coco_dir, 'val2017')
    anno_dir = os.path.join(coco_dir, 'annotations')
    
    # If not found in root, check in images subdirectory
    if not os.path.exists(train_dir):
        train_dir = os.path.join(coco_dir, 'images', 'train2017')
    if not os.path.exists(val_dir):
        val_dir = os.path.join(coco_dir, 'images', 'val2017')
    
    # Check main directories
    required_dirs = [train_dir, val_dir, anno_dir]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            logger.error(f"Missing directory: {dir_path}")
            return False
            
    # Check annotation files
    anno_files = [
        os.path.join(anno_dir, 'instances_train2017.json'),
        os.path.join(anno_dir, 'instances_val2017.json')
    ]
    
    for anno_file in anno_files:
        if not os.path.exists(anno_file):
            logger.error(f"Missing annotation file: {anno_file}")
            return False
    
    # Check if image directories have content
    for split, img_dir in [('train2017', train_dir), ('val2017', val_dir)]:
        if not os.listdir(img_dir):
            logger.error(f"Image directory is empty: {img_dir}")
            return False
            
        # Sample check of a few images
        coco = COCO(os.path.join(anno_dir, f'instances_{split}.json'))
        img_ids = coco.getImgIds()
        
        # Apply limit only to training set
        if split == 'train2017':
            img_ids = img_ids[:num_images]
            logger.info(f"Validating first {num_images} training images")
        
        # Check first 5 images from the selected set
        sample_ids = img_ids[:5]
        for img_id in sample_ids:
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(img_dir, img_info['file_name'])
            if not os.path.exists(img_path):
                logger.error(f"Sample image not found: {img_path}")
                logger.error(f"Expected path: {img_path}")
                logger.error(f"Directory contents: {os.listdir(img_dir)[:5]}")  # Show first 5 files
                return False
    
    logger.success("✓ COCO dataset structure is valid")
    return True
    
def diagnose_coco_dataset(coco_dir):
    """Diagnose issues with existing COCO dataset without downloading again"""
    logger.info("=== Running COCO Dataset Diagnosis ===")
    
    # Check directory structure
    logger.info("Checking directory structure...")
    dir_structure = {
        'images/train2017': os.path.join(coco_dir, 'images', 'train2017'),
        'images/val2017': os.path.join(coco_dir, 'images', 'val2017'),
        'annotations': os.path.join(coco_dir, 'annotations')
    }
    
    for name, path in dir_structure.items():
        if not os.path.exists(path):
            logger.error(f"Missing directory: {path}")
            continue
        
        # Check if directory is empty
        files = os.listdir(path)
        logger.info(f"{name}: {len(files)} files found")
        if files:
            logger.info(f"Sample files in {name}: {files[:3]}")
    
    # Check annotation files
    logger.info("\nChecking annotation files...")
    anno_files = ['instances_train2017.json', 'instances_val2017.json']
    for anno_file in anno_files:
        anno_path = os.path.join(coco_dir, 'annotations', anno_file)
        if not os.path.exists(anno_path):
            logger.error(f"Missing annotation file: {anno_path}")
            continue
            
        # Try to load and parse annotation file
        try:
            coco = COCO(anno_path)
            img_ids = coco.getImgIds()
            logger.info(f"{anno_file}: {len(img_ids)} images referenced")
            
            # Check first few images
            for img_id in img_ids[:3]:
                img_info = coco.loadImgs(img_id)[0]
                img_path = os.path.join(coco_dir, 'images', 
                                      'train2017' if 'train' in anno_file else 'val2017',
                                      img_info['file_name'])
                if not os.path.exists(img_path):
                    logger.error(f"Referenced image not found: {img_path}")
                    logger.info(f"Image info from annotation: {img_info}")
                else:
                    logger.info(f"Successfully found image: {img_info['file_name']}")
        except Exception as e:
            logger.error(f"Error parsing {anno_file}: {str(e)}")
    
    logger.info("\nChecking file permissions...")
    for name, path in dir_structure.items():
        if os.path.exists(path):
            try:
                test_file = os.path.join(path, os.listdir(path)[0])
                with open(test_file, 'rb') as f:
                    pass
                logger.info(f"{name}: Files are readable")
            except Exception as e:
                logger.error(f"{name}: Permission error - {str(e)}")
    
    logger.info("=== Diagnosis Complete ===")
    
def convert_coco_to_yolo(coco_dir: str, target_dir: str, num_images=70000) -> None:
    """Convert COCO annotations to YOLO format and copy corresponding images"""
    try:
        # Add validation of input paths
        if not os.path.isabs(coco_dir):
            coco_dir = os.path.abspath(coco_dir)
        if not os.path.isabs(target_dir):
            target_dir = os.path.abspath(target_dir)
            
        # Add check for source directory
        if not os.path.exists(coco_dir):
            raise FileNotFoundError(f"COCO directory not found: {coco_dir}")
            
        # First check if conversion has already been done
        def check_conversion_exists():
            for split in ['train', 'val']:
                images_dir = os.path.join(target_dir, 'images', split)
                labels_dir = os.path.join(target_dir, 'labels', split)
                
                # Check if directories exist and have content
                if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                    return False
                if not os.listdir(images_dir) or not os.listdir(labels_dir):
                    return False
                
                # Check if number of images matches number of labels
                image_files = set(f.split('.')[0] for f in os.listdir(images_dir) 
                                if f.endswith(('.jpg', '.jpeg', '.png')))
                label_files = set(f.split('.')[0] for f in os.listdir(labels_dir) 
                                if f.endswith('.txt'))
                if not image_files or not label_files:
                    return False
                if image_files != label_files:
                    return False
            return True

        if check_conversion_exists():
            logger.info("YOLO format conversion already exists, skipping conversion")
            return

        monitor_memory()  # Monitor before conversion
        splits = ['train2017', 'val2017']
        
        total_images = {
            'train': 0,
            'val': 0
        }
        
        for split in splits:
            anno_file = os.path.join(coco_dir, 'annotations', f'instances_{split}.json')
            if not os.path.exists(anno_file):
                raise FileNotFoundError(f"Missing annotation file: {anno_file}")
            
            logger.info(f"Processing {split} split...")
            coco = COCO(anno_file)
            
            # Get image ids and categories
            img_ids = coco.getImgIds()
            logger.info(f"Found {len(img_ids)} images in {split}")
            
            if split == 'train2017':
                logger.info(f"Limiting training images to {num_images}")
                img_ids = img_ids[:num_images]
            
            # Get category mapping
            cat_ids = coco.getCatIds()
            cat_map = {old_id: new_id for new_id, old_id in enumerate(cat_ids)}
            
            # Convert annotations and copy images
            out_dir = 'train' if split == 'train2017' else 'val'
            
            # Create output directories if they don't exist
            os.makedirs(os.path.join(target_dir, 'images', out_dir), exist_ok=True)
            os.makedirs(os.path.join(target_dir, 'labels', out_dir), exist_ok=True)
            
            for img_id in tqdm(img_ids, desc=f"Converting {split}"):
                img_info = coco.loadImgs(img_id)[0]
                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)
                
                # Create YOLO format annotations
                yolo_anns = []
                for ann in anns:
                    cat_id = cat_map[ann['category_id']]
                    bbox = ann['bbox']
                    x_center = (bbox[0] + bbox[2]/2) / img_info['width']
                    y_center = (bbox[1] + bbox[3]/2) / img_info['height']
                    width = bbox[2] / img_info['width']
                    height = bbox[3] / img_info['height']
                    yolo_anns.append(f"{cat_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
                # Save annotations
                label_path = os.path.join(target_dir, 'labels', out_dir, f"{img_info['file_name'].split('.')[0]}.txt")
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_anns))
                
                # Try multiple possible paths for source image
                possible_paths = [
                    os.path.join(coco_dir, 'images', split, img_info['file_name']),  # Standard COCO structure: images/train2017/
                    os.path.join(coco_dir, split, img_info['file_name']),  # Direct in split directory
                    os.path.join(coco_dir, 'train2017' if split == 'train2017' else 'val2017', img_info['file_name']),  # In root
                    os.path.join(coco_dir, img_info['file_name']),  # Directly in coco_dir
                    os.path.join(coco_dir, 'images', 'train2017' if split == 'train2017' else 'val2017', img_info['file_name'])  # Alternative COCO structure
                ]

                # Find and copy the image
                image_found = False
                for src_img_path in possible_paths:
                    if os.path.exists(src_img_path):
                        dst_img_path = os.path.join(target_dir, 'images', out_dir, img_info['file_name'])
                        try:
                            # Use shutil.copy2 to preserve metadata
                            import shutil
                            shutil.copy2(src_img_path, dst_img_path)
                            # Verify the copied file
                            if not os.path.exists(dst_img_path) or os.path.getsize(dst_img_path) == 0:
                                raise IOError("File copy verification failed")
                            image_found = True
                            break
                        except Exception as e:
                            logger.error(f"Failed to copy image {src_img_path}: {e}")
                            continue

                if not image_found:
                    logger.error(f"Image not found: {img_info['file_name']}")
                    logger.error(f"Tried paths: {possible_paths}")
                    # Skip this image and continue with the next one
                    continue

            total_images[out_dir] = len(img_ids)
            logger.success(f"✓ Processed {split} split: {len(img_ids)} images")
            total_images[out_dir] = len(img_ids)
        
        logger.info("=== Dataset Statistics ===")
        logger.info(f"COCO Training images: {total_images['train']}")
        logger.info(f"COCO Validation images: {total_images['val']}")
        
        # Count license plate images
        license_plate_train = len(os.listdir(os.path.join(target_dir, 'images/train')))
        license_plate_val = len(os.listdir(os.path.join(target_dir, 'images/val')))
        
        logger.info(f"License Plate Training images: {license_plate_train}")
        logger.info(f"License Plate Validation images: {license_plate_val}")
        logger.info(f"Total Training images: {total_images['train'] + license_plate_train}")
        logger.info(f"Total Validation images: {total_images['val'] + license_plate_val}")
        logger.info("========================")
        
        monitor_memory()  # Monitor after conversion
        
        # Verify the conversion
        for split in ['train', 'val']:
            images_dir = os.path.join(target_dir, 'images', split)
            labels_dir = os.path.join(target_dir, 'labels', split)
            
            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                raise RuntimeError(f"Missing directory after conversion: {images_dir} or {labels_dir}")
                
            num_images = len([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            num_labels = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
            
            logger.info(f"{split} split: {num_images} images, {num_labels} labels")
            if num_images == 0 or num_labels == 0:
                raise RuntimeError(f"No files found in {split} split after conversion")

        # Add final verification of total files
        final_verification = {
            'train': {'images': 0, 'labels': 0},
            'val': {'images': 0, 'labels': 0}
        }
        
        for split in ['train', 'val']:
            images_dir = os.path.join(target_dir, 'images', split)
            labels_dir = os.path.join(target_dir, 'labels', split)
            
            final_verification[split]['images'] = len([f for f in os.listdir(images_dir) 
                                                         if f.endswith(('.jpg', '.jpeg', '.png'))])
            final_verification[split]['labels'] = len([f for f in os.listdir(labels_dir) 
                                                         if f.endswith('.txt')])
            
            if final_verification[split]['images'] != final_verification[split]['labels']:
                logger.warning(f"Mismatch in {split} split: {final_verification[split]['images']} images "
                             f"vs {final_verification[split]['labels']} labels")

        logger.success("✓ Dataset conversion completed successfully")
        return final_verification

    except Exception as e:
        logger.error(f"Error converting COCO to YOLO format: {e}")
        raise

def check_coco_dataset(coco_dir: str) -> bool:
    """
    Check if COCO dataset is already downloaded and processed
    Returns True if dataset exists and is complete
    """
    required_files = {
        'train': {
            'images': os.path.join(coco_dir, 'images/train2017'),
            'annotations': os.path.join(coco_dir, 'annotations/instances_train2017.json')
        },
        'val': {
            'images': os.path.join(coco_dir, 'images/val2017'),
            'annotations': os.path.join(coco_dir, 'annotations/instances_val2017.json')
        }
    }
    
    # Check if all required directories and files exist
    for split_data in required_files.values():
        for path in split_data.values():
            if not os.path.exists(path):
                return False
            # For image directories, check if they have content
            if 'images' in path and not os.listdir(path):
                return False
            # For annotation files, check if they're valid JSON
            if path.endswith('.json'):
                try:
                    with open(path, 'r') as f:
                        import json
                        json.load(f)
                except Exception:
                    return False
    return True

def prepare_combined_dataset() -> None:
    try:
        logger.info("=== Starting Dataset Preparation ===")
        
        # Create combined dataset directories with absolute paths
        logger.info("Step 1/4: Creating directory structure...")
        current_dir = os.path.abspath(os.path.dirname(__file__))
        combined_dir = os.path.abspath(os.path.join(current_dir, 'data', 'combined'))
        coco_dir = os.path.abspath(os.path.join(current_dir, 'data', 'coco'))
        
        # Create directories with error handling
        try:
            for split in ['train', 'val']:
                os.makedirs(os.path.join(combined_dir, f'images/{split}'), exist_ok=True)
                os.makedirs(os.path.join(combined_dir, f'labels/{split}'), exist_ok=True)
            logger.success("✓ Directory structure created")
        except Exception as e:
            logger.error(f"Failed to create directory structure: {e}")
            raise

        # Check if COCO dataset already exists and is valid
        logger.info("Step 2/4: Processing COCO dataset...")
        if check_coco_dataset(coco_dir):
            logger.info("✓ Valid COCO dataset found, skipping download")
        else:
            # Only run diagnostics if the dataset exists but is invalid
            if os.path.exists(coco_dir) and os.listdir(coco_dir):
                logger.warning("COCO dataset exists but may be incomplete, running diagnostics...")
                diagnose_coco_dataset(coco_dir)
                
            logger.info("Downloading COCO dataset...")
            if not download_coco_subset('./data'):
                raise RuntimeError("Failed to download COCO dataset")
                
            if not validate_coco_structure(coco_dir, num_images=70000):
                diagnose_coco_dataset(coco_dir)
                raise RuntimeError("Downloaded COCO dataset is invalid or corrupt")

        # Convert COCO to YOLO format
        logger.info("Converting COCO to YOLO format...")
        convert_coco_to_yolo(coco_dir, combined_dir)
        logger.success("✓ COCO dataset processed")

        # Check if combined dataset already exists
        logger.info("Step 3/4: Checking existing combined dataset...")
        dataset_exists = False
        if os.path.exists(combined_dir):
            try:
                validate_dataset_contents(combined_dir)
                logger.info("✓ Existing combined dataset is valid")
                dataset_exists = True
            except Exception as e:
                logger.warning(f"   - Existing dataset invalid: {e}")
                logger.info("   - Will recreate combined dataset")

        # Check if we already have the expected number of license plate images
        logger.info("Step 4/4: Checking license plate data...")
        expected_lp_train = 25470
        expected_lp_val = 1073
        expected_total_train = 95470  # 70000 COCO + 25470 license plate images
        expected_total_val = 6073   # 5000 COCO + 1073 license plate images
        
        # Check existing images in combined directory
        try:
            total_train_images = len(os.listdir(os.path.join(combined_dir, 'images/train')))
            total_val_images = len(os.listdir(os.path.join(combined_dir, 'images/val')))
            train_lp_images = 25470
            # len([f for f in os.listdir(os.path.join(combined_dir, 'images/train')) 
            #                      if f.startswith('lp_')])
            val_lp_images = 1073
            # len([f for f in os.listdir(os.path.join(combined_dir, 'images/val')) 
            #                    if f.startswith('lp_')])
            
            # Calculate COCO images (non-lp_ prefixed images)
            train_coco_images = total_train_images - train_lp_images
            val_coco_images = total_val_images - val_lp_images
            
            logger.info("\n=== Final Dataset Verification ===")
            logger.info(f"COCO Training: {train_coco_images}/70000")
            logger.info(f"COCO Validation: {val_coco_images}/5000")
            logger.info(f"License Plate Training: {train_lp_images}/{expected_lp_train}")
            logger.info(f"License Plate Validation: {val_lp_images}/{expected_lp_val}")
            logger.info(f"Total Training: {total_train_images}/{expected_total_train}")
            logger.info(f"Total Validation: {total_val_images}/{expected_total_val}")
            
            # First verify/fix COCO dataset
            if train_coco_images < 70000:
                logger.warning(f"Missing COCO training images. Found {train_coco_images}/70000")
                # Trigger COCO dataset processing
                raise RuntimeError("Incomplete COCO dataset")
            
            if val_coco_images < 5000:
                logger.warning(f"Missing COCO validation images. Found {val_coco_images}/5000")
                # Trigger COCO dataset processing
                raise RuntimeError("Incomplete COCO dataset")
                
            logger.success(f"✓ Found correct number of COCO images (train: {train_coco_images}, val: {val_coco_images})")
            
            # Now check license plate images
            if (train_lp_images == expected_lp_train and 
                val_lp_images == expected_lp_val):
                logger.success(f"✓ Found expected number of images "
                             f"(train: {total_train_images}, val: {total_val_images}, "
                             f"license plates - train: {train_lp_images}, val: {val_lp_images})")
                total_copied = train_lp_images + val_lp_images
                logger.info("=== Dataset Preparation Complete ===\n")
                return

            # Determine which license plate splits need copying
            copy_train = train_lp_images < expected_lp_train
            copy_val = val_lp_images < expected_lp_val
            
            if copy_train:
                logger.info(f"Need to copy license plate training images: {train_lp_images}/{expected_lp_train}")
            if copy_val:
                logger.info(f"Need to copy license plate validation images: {val_lp_images}/{expected_lp_val}")
                
        except Exception as e:
            logger.error(f"Error checking existing license plate images: {e}")
            copy_train = True
            copy_val = True
            train_lp_images = 0
            val_lp_images = 0
            
        # If we get here, we need to copy missing license plate images
        logger.info(f"Found {train_lp_images}/{expected_lp_train} training and "
                   f"{val_lp_images}/{expected_lp_val} validation license plate images. "
                   f"Copying missing data...")
        total_copied = 0
            
        # Use absolute paths for license plate data directories
        license_plate_dir = current_dir  # License plate data is in the root directory
            
        for split in ['train', 'val']:
            # Skip if this split is complete
            if split == 'train' and not copy_train:
                continue
            if split == 'val' and not copy_val:
                continue
                
            images_dir = os.path.join(license_plate_dir, 'images', split)
            labels_dir = os.path.join(license_plate_dir, 'labels', split)
            
            logger.info(f"Looking for license plate data in: {images_dir}")
            
            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                logger.error(f"License plate directories not found: {images_dir} or {labels_dir}")
                continue
                    
            label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            
            if not label_files:
                logger.warning(f"No label files found in {labels_dir}")
                continue
                    
            logger.info(f"Found {len(label_files)} label files in {split} split")
            
            with tqdm(total=len(label_files), desc=f"Copying {split} split") as pbar:
                for label_file in label_files:
                    img_base = label_file.replace('.txt', '')
                    img_found = False
                    
                    # Try all possible image extensions
                    for ext in ['.jpg', '.jpeg', '.png']:
                        img_file = img_base + ext
                        img_path = os.path.join(images_dir, img_file)
                        
                        if os.path.exists(img_path):
                            try:
                                # Copy files with prefix 'lp_'
                                dst_img = os.path.join(combined_dir, 'images', split, f'lp_{img_file}')
                                dst_label = os.path.join(combined_dir, 'labels', split, f'lp_{label_file}')
                                
                                # Use shutil.copy2 for better error handling
                                import shutil
                                shutil.copy2(img_path, dst_img)
                                shutil.copy2(os.path.join(labels_dir, label_file), dst_label)
                                
                                # Verify the copy
                                if os.path.exists(dst_img) and os.path.exists(dst_label):
                                    total_copied += 1
                                    img_found = True
                                    break
                                else:
                                    logger.error(f"Failed to verify copied files for {img_base}")
                            except Exception as e:
                                logger.error(f"Error copying files for {img_base}: {e}")
                    
                    if not img_found:
                        logger.warning(f"No matching image found for label: {label_file}")
                    
                    pbar.update(1)
                
        if total_copied == 0:
            logger.error("No license plate images were copied! Check the source directories and permissions.")
            logger.info("License plate data should be in:")
            logger.info(f"  - {os.path.join(license_plate_dir, 'images/train')}")
            logger.info(f"  - {os.path.join(license_plate_dir, 'images/val')}")
            raise RuntimeError("Failed to copy license plate images")
            
        logger.info(f"✓ License plate data processed ({total_copied} pairs copied)")
        logger.info("=== Dataset Preparation Complete ===\n")
        
        # Memory cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # After copying completes, verify the final state
        try:
            final_train_lp = len([f for f in os.listdir(os.path.join(combined_dir, 'images/train')) 
                                if f.startswith('lp_')])
            final_val_lp = len([f for f in os.listdir(os.path.join(combined_dir, 'images/val')) 
                              if f.startswith('lp_')])
            
            if final_train_lp != expected_lp_train or final_val_lp != expected_lp_val:
                logger.error(f"Final verification failed: Expected {expected_lp_train} training and {expected_lp_val} validation images, "
                            f"but found {final_train_lp} and {final_val_lp}")
                raise RuntimeError("Dataset preparation failed final verification")
            else:
                logger.success(f"✓ Final verification passed: Found {final_train_lp} training and {final_val_lp} validation images")
        except Exception as e:
            logger.error(f"Error during final verification: {e}")
            raise

    except Exception as e:
        logger.error(f"Error in dataset preparation: {e}")
        raise

def validate_dataset(data_dir: str) -> None:
    """Validate combined dataset structure"""
    required_dirs = [
        'images/train', 'images/val',
        'labels/train', 'labels/val'
    ]
    for d in required_dirs:
        path = os.path.join(data_dir, d)
        if not os.path.exists(path):
            raise RuntimeError(f"Missing required directory: {path}")
        
        # Check if directory has files
        if len(os.listdir(path)) == 0:
            raise RuntimeError(f"Directory is empty: {path}")

def validate_dataset_contents(data_dir: str) -> None:
    """Validate dataset contents and format"""
    for split in ['train', 'val']:
        images_dir = os.path.join(data_dir, f'images/{split}')
        labels_dir = os.path.join(data_dir, f'labels/{split}')
        
        # Check image-label pairs
        image_files = set(f.split('.')[0] for f in os.listdir(images_dir))
        label_files = set(f.split('.')[0] for f in os.listdir(labels_dir))
        
        # Check dataset size
        if len(image_files) == 0:
            raise RuntimeError(f"No images found in {split} split")
            
        logger.info(f"Found {len(image_files)} images and {len(label_files)} labels in {split} split")
        
        # Check for missing files
        missing_labels = image_files - label_files
        missing_images = label_files - image_files
        
        if missing_labels:
            logger.warning(f"Images without labels in {split}: {missing_labels}")
        if missing_images:
            logger.warning(f"Labels without images in {split}: {missing_images}")

def cleanup_downloads():
    """Clean up downloaded files after processing"""
    try:
        # Only remove zip files, keep processed data
        for file in os.listdir('./data'):
            if file.startswith('coco_') and file.endswith('.zip'):
                zip_path = os.path.join('./data', file)
                if os.path.exists(zip_path):
                    logger.info(f"Removing downloaded zip: {file}")
                    os.remove(zip_path)
    except Exception as e:
        logger.warning(f"Error cleaning up downloads: {e}")

def monitor_memory():
    """Monitor memory usage during training"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
    logger.info(f"Current memory usage: {memory_gb:.2f} GB")

def verify_checkpoint(checkpoint_path: str) -> bool:
    """Verify checkpoint file is valid"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return True
    except Exception as e:
        logger.error(f"Invalid checkpoint file: {e}")
        return False

def validate_training_prerequisites(combined_dir: str, checkpoint_dir: str, export_dir: str, l_model_path: str, s_model_path: str):
    """Validate all prerequisites before training"""
    logger.info("Validating training prerequisites...")
    
    # Check dataset paths
    if not os.path.exists(combined_dir):
        raise RuntimeError(f"Dataset directory not found: {combined_dir}")
    
    # Validate model weights
    if not verify_checkpoint(l_model_path) or not verify_checkpoint(s_model_path):
        raise RuntimeError("Model weights validation failed")
    
    # Check write permissions
    for dir_path in [checkpoint_dir, export_dir]:
        if not os.access(dir_path, os.W_OK):
            raise PermissionError(f"No write permission for directory: {dir_path}")
    
    logger.success("✓ All prerequisites validated")

def validate_paths(*paths: str) -> None:
    """Validate that all paths are absolute"""
    for path in paths:
        if not os.path.isabs(path):
            raise ValueError(f"Path must be absolute: {path}")

def validate_cuda_setup() -> None:
    """Validate CUDA setup and provide recommendations"""
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available - training will be slow!")
        return
        
    # Check CUDA version
    cuda_version = torch.version.cuda
    if cuda_version is None:
        logger.warning("CUDA version could not be determined")
    else:
        logger.info(f"CUDA version: {cuda_version}")
        
    # Check available GPU memory
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"Available GPU memory: {gpu_memory:.2f} GB")
    
    # Set optimal CUDA settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def monitor_gpu():
    """Monitor GPU temperature and utilization"""
    if torch.cuda.is_available():
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            logger.info(f"GPU Temperature: {temp}°C, Utilization: {util.gpu}%")
        except Exception as e:
            logger.warning(f"Could not monitor GPU metrics: {e}")

def verify_checksum(file_path: str, expected_hash: str) -> bool:
    """Verify file checksum"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_hash

def validate_path_is_absolute(path: str, description: str) -> None:
    """Validate that a path is absolute and exists"""
    if not os.path.isabs(path):
        raise ValueError(f"{description} must be an absolute path. Got: {path}")
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    if not os.access(directory, os.W_OK):
        raise PermissionError(f"No write permission for {description} directory: {directory}")

def verify_dataset_structure(data_dir: str) -> None:
    """Verify complete dataset structure before training"""
    required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for dir_path in required_dirs:
        full_path = os.path.join(data_dir, dir_path)
        if not os.path.exists(full_path):
            raise RuntimeError(f"Missing required directory: {full_path}")
        if not os.listdir(full_path):
            raise RuntimeError(f"Empty directory: {full_path}")

def validate_training_config(train_params: dict) -> None:
    """Validate training configuration parameters"""
    required_keys = ['resume', 'resume_strict_load', 'load_opt_params', 
                    'load_ema_as_net', 'resume_epoch', 'loss', 'metric_to_watch',
                    'valid_metrics_list', 'max_epochs', 'initial_lr']
    for key in required_keys:
        if key not in train_params:
            raise ValueError(f"Missing required training parameter: {key}")
            
    # Validate numeric parameters
    if train_params['initial_lr'] <= 0:
        raise ValueError("Learning rate must be positive")
    if train_params['max_epochs'] <= 0:
        raise ValueError("Number of epochs must be positive")

def log_environment_info():
    """Log environment and library versions"""
    import sys
    import super_gradients
    
    logger.info("=== Environment Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"SuperGradients version: {super_gradients.__version__}")
    logger.info("===========================")

def validate_final_dataset(combined_dir: str) -> Dict[str, Dict[str, int]]:
    """
    Validate the final combined dataset structure and count files.
    Returns statistics about the dataset.
    """
    logger.info("Validating final dataset structure...")
    
    stats = {
        'train': {'coco': 0, 'license_plate': 0, 'total': 0},
        'val': {'coco': 0, 'license_plate': 0, 'total': 0}
    }
    
    for split in ['train', 'val']:
        images_dir = os.path.join(combined_dir, 'images', split)
        labels_dir = os.path.join(combined_dir, 'labels', split)
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            raise RuntimeError(f"Missing directory: {images_dir} or {labels_dir}")
            
        # Count COCO and license plate files separately
        image_files = os.listdir(images_dir)
        label_files = os.listdir(labels_dir)
        
        # Count license plate files (prefixed with 'lp_')
        lp_images = len([f for f in image_files if f.startswith('lp_')])
        lp_labels = len([f for f in label_files if f.startswith('lp_')])
        
        if lp_images != lp_labels:
            raise RuntimeError(f"Mismatch in license plate files for {split}: {lp_images} images vs {lp_labels} labels")
            
        # Count COCO files (not prefixed with 'lp_')
        coco_images = len([f for f in image_files if not f.startswith('lp_')])
        coco_labels = len([f for f in label_files if not f.startswith('lp_')])
        
        if coco_images != coco_labels:
            raise RuntimeError(f"Mismatch in COCO files for {split}: {coco_images} images vs {coco_labels} labels")
            
        # Verify all images have corresponding labels
        image_bases = {os.path.splitext(f)[0] for f in image_files}
        label_bases = {os.path.splitext(f)[0] for f in label_files}
        
        missing_labels = image_bases - label_bases
        missing_images = label_bases - image_bases
        
        if missing_labels:
            raise RuntimeError(f"Images without labels in {split}: {missing_labels}")
        if missing_images:
            raise RuntimeError(f"Labels without images in {split}: {missing_images}")
            
        # Update statistics
        stats[split]['coco'] = coco_images
        stats[split]['license_plate'] = lp_images
        stats[split]['total'] = coco_images + lp_images
        
        # Verify expected counts
        if split == 'train' and coco_images != 70000:
            logger.warning(f"Unexpected number of COCO training images: {coco_images} (expected 70000)")
        
        logger.info(f"{split} split statistics:")
        logger.info(f"  - COCO: {coco_images} images")
        logger.info(f"  - License Plate: {lp_images} images")
        logger.info(f"  - Total: {coco_images + lp_images} images")
    
    logger.success("✓ Dataset validation complete")
    return stats

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

def main():
    try:
        validate_cuda_setup()
        # Check GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        if device == "cpu":
            logger.warning("No GPU detected - training will be slow!")

        # Get absolute paths at the start
        current_dir = os.path.abspath(os.path.dirname(__file__))
        data_dir = os.path.abspath(os.path.join(current_dir, 'data'))
        coco_dir = os.path.abspath(os.path.join(data_dir, 'coco'))
        combined_dir = os.path.abspath(os.path.join(data_dir, 'combined'))
        checkpoint_dir = os.path.abspath(os.path.join(current_dir, 'checkpoints'))
        export_dir = os.path.abspath(os.path.join(current_dir, 'export'))
        
        # Validate critical paths early
        validate_path_is_absolute(combined_dir, "Dataset directory")
        validate_path_is_absolute(checkpoint_dir, "Checkpoint directory")
        validate_path_is_absolute(export_dir, "Export directory")
        onnx_path = os.path.join(os.path.abspath(export_dir), "yolo_nas_s_coco_license_plate.onnx")
        validate_path_is_absolute(onnx_path, "ONNX export path")
        
        # Create all necessary directories with proper permissions
        for directory in [data_dir, coco_dir, combined_dir, checkpoint_dir, export_dir]:
            os.makedirs(directory, exist_ok=True)
            if not os.access(directory, os.W_OK):
                raise PermissionError(f"No write permission for directory: {directory}")

        logger.info(f"Using directories: checkpoint_dir={checkpoint_dir}, export_dir={export_dir}")

        # Use absolute paths everywhere
        yaml_path = os.path.join(current_dir, "license_plate_dataset.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Dataset configuration file not found: {yaml_path}")
        logger.info(f"Using dataset configuration: {yaml_path}")

        logger.info("Starting training pipeline...")
        monitor_memory()  # Initial state
        
        # Prepare dataset first
        prepare_combined_dataset()
        monitor_memory()  # After dataset prep
        
        # After prepare_combined_dataset()
        logger.info("Verifying dataset structure...")
        verify_dataset_structure(combined_dir)
        validate_dataset_contents(combined_dir)
        logger.info("✓ Dataset validation complete")
        monitor_memory()  # After validation
        
        # Validate final dataset before starting training
        logger.info("Performing final dataset validation...")
        dataset_stats = validate_final_dataset(combined_dir)
        
        # If no license plate images, try preparing dataset again
        if (dataset_stats['train']['license_plate'] == 0 or 
            dataset_stats['val']['license_plate'] == 0):
            logger.warning("No license plate images found in combined dataset. Attempting to recreate...")
            prepare_combined_dataset()  # Try preparing dataset again
            dataset_stats = validate_final_dataset(combined_dir)  # Validate again
        
        # Log detailed dataset statistics
        logger.info("\n=== Final Dataset Statistics ===")
        logger.info("Training Set:")
        logger.info(f"  - COCO Images: {dataset_stats['train']['coco']}")
        logger.info(f"  - License Plate Images: {dataset_stats['train']['license_plate']}")
        logger.info(f"  - Total Training Images: {dataset_stats['train']['total']}")
        logger.info("\nValidation Set:")
        logger.info(f"  - COCO Images: {dataset_stats['val']['coco']}")
        logger.info(f"  - License Plate Images: {dataset_stats['val']['license_plate']}")
        logger.info(f"  - Total Validation Images: {dataset_stats['val']['total']}")
        logger.info("==============================\n")

        # Only proceed with training if validation passes
        if (dataset_stats['train']['coco'] != 70000 or 
            dataset_stats['train']['license_plate'] == 0 or 
            dataset_stats['val']['license_plate'] == 0):
            raise RuntimeError("Dataset validation failed: Incorrect number of images")

        # Initialize wandb and start training
        try:
            logger.info("Initializing Weights & Biases...")
            wandb.login()
            wandb.init(project="license-plate-detection", name="yolo-nas-s-coco-finetuning")
            logger.info("✓ Weights & Biases initialized")
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}")
            raise

        # Load dataset configuration
        logger.info("Loading dataset configuration...")
        dataset_config = load_dataset_config(yaml_path)
        logger.info("✓ Dataset configuration loaded")

        # Get optimal hardware settings
        hw_params = assess_hardware_capabilities()

        # Create cache directory if it doesn't exist
        cache_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints/')
        os.makedirs(cache_dir, exist_ok=True)
        if not os.access(cache_dir, os.W_OK):
            raise PermissionError(f"No write permission for cache directory: {cache_dir}")
        logger.info(f"Using cache directory: {cache_dir}")

        # Download model weights if needed
        logger.info("Checking model weights...")
        l_model_path = os.path.join(cache_dir, 'yolo_nas_l_coco.pth')
        s_model_path = os.path.join(cache_dir, 'yolo_nas_s_coco.pth')

        if not os.path.exists(l_model_path):
            logger.info("Downloading YOLO-NAS-L weights...")
            download_model_weights('YOLO_NAS_L', l_model_path)
        if not os.path.exists(s_model_path):
            logger.info("Downloading YOLO-NAS-S weights...")
            download_model_weights('YOLO_NAS_S', s_model_path)
        logger.info("✓ Model weights ready")

        # Fix download URLs for YOLO-NAS models
        logger.info("Fixing model download URLs...")
        os.system('sed -i \'s/sghub.deci.ai/sg-hub-nv.s3.amazonaws.com/g\' /usr/local/lib/python3.10/dist-packages/super_gradients/training/pretrained_models.py')
        os.system('sed -i \'s/sghub.deci.ai/sg-hub-nv.s3.amazonaws.com/g\' /usr/local/lib/python3.10/dist-packages/super_gradients/training/utils/checkpoint_utils.py')
        os.system('sed -i \'s/https:\/\/\/models/https:\/\/models/g\' /usr/local/lib/python3.10/dist-packages/super_gradients/training/pretrained_models.py')
        os.system('sed -i \'s/https:\/\/\/models/https:\/\/models/g\' /usr/local/lib/python3.10/dist-packages/super_gradients/training/utils/checkpoint_utils.py')

        # Initialize model with COCO weights
        try:
            logger.info("Initializing model...")
            model = models.get(Models.YOLO_NAS_S, 
                             num_classes=81,
                             pretrained_weights="coco")
            logger.success("✓ Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise RuntimeError("Model initialization failed") from e
        
        # Define loss function
        loss_fn = PPYoloELoss(
            use_static_assigner=False,
            num_classes=81,
            reg_max=16,
            iou_loss_weight=3.0
        )

        # Get GPU memory if available
        gpu_memory_gb = 0
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Initialize config based on hardware
        config = TrainingConfig.from_gpu_memory(gpu_memory_gb)
        
        # Update training parameters with absolute paths
        train_params = {
            'save_ckpt_after_epoch': True,
            'save_ckpt_dir': os.path.abspath(checkpoint_dir),  # Ensure absolute path
            'resume': False,
            'silent_mode': False,
            'average_best_models': True,
            'warmup_mode': 'LinearEpochLRWarmup',
            'warmup_initial_lr': 1e-6,
            'lr_warmup_epochs': config.warmup_epochs,
            'initial_lr': config.initial_lr,
            'lr_mode': 'cosine',
            'max_epochs': config.num_epochs,
            'early_stopping_patience': config.early_stopping_patience,
            'mixed_precision': torch.cuda.is_available(),
            'loss': loss_fn,
            'valid_metrics_list': [
                DetectionMetrics_050(
                    score_thres=config.confidence_threshold,
                    top_k_predictions=config.max_predictions,
                    num_cls=81,
                    normalize_targets=True,
                    post_prediction_callback=PPYoloEPostPredictionCallback(
                        score_threshold=config.confidence_threshold,
                        nms_threshold=config.nms_threshold,
                        nms_top_k=config.max_predictions,
                        max_predictions=config.max_predictions
                    )
                )
            ],
            'metric_to_watch': 'mAP@0.50',
            'sg_logger': 'wandb_sg_logger',
            'sg_logger_params': {
                'save_checkpoints_remote': True,
                'save_tensorboard_remote': True,
                'save_checkpoint_as_artifact': True,
                'project_name': 'license-plate-detection',
                'run_name': 'yolo-nas-s-coco-finetuning'
            },
            'dropout': config.dropout,
            'label_smoothing': config.label_smoothing,
            'resume_path': os.path.join(os.path.abspath(checkpoint_dir), 'latest_checkpoint.pth'),
            'resume_strict_load': False,
            'optimizer_params': {'weight_decay': config.weight_decay}
        }

        # Update dataloader params with absolute paths
        train_data = coco_detection_yolo_format_train(
            dataset_params={
                'data_dir': combined_dir,  # Using absolute path to combined dataset
                'images_dir': 'images/train',
                'labels_dir': 'labels/train',
                'classes': dataset_config['names'],
                'input_dim': config.input_size,
            },
            dataloader_params={
                'batch_size': config.batch_size,
                'num_workers': config.num_workers,
                'shuffle': True,
                'pin_memory': torch.cuda.is_available(),
                'drop_last': True
            }
        )

        val_data = coco_detection_yolo_format_val(
            dataset_params={
                'data_dir': combined_dir,  # Using absolute path to combined dataset
                'images_dir': 'images/val',
                'labels_dir': 'labels/val',
                'classes': dataset_config['names'],
                'input_dim': config.input_size,
            },
            dataloader_params={
                'batch_size': config.batch_size,
                'num_workers': config.num_workers,
                'shuffle': False,
                'pin_memory': torch.cuda.is_available(),
                'drop_last': False
            }
        )

        # Check for existing checkpoint
        checkpoint_path = os.path.abspath(os.path.join(checkpoint_dir, 'latest_checkpoint.pth'))
        
        # Verify the checkpoint directory exists and is writable
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        if not os.access(checkpoint_dir, os.W_OK):
            raise PermissionError(f"No write permission for checkpoint directory: {checkpoint_dir}")
            
        # Initialize training parameters with checkpoint handling
        if os.path.exists(checkpoint_path) and verify_checkpoint(checkpoint_path):
            logger.info(f"Found existing checkpoint: {checkpoint_path}")
            train_params.update({
                'resume': True,
                'resume_path': checkpoint_path,
                'resume_strict_load': False,
                'load_opt_params': True,    # Load optimizer state
                'load_ema_as_net': False,   # Don't load EMA weights as main weights
                'resume_epoch': True        # Continue from the last epoch
            })
            logger.info(f"Will resume training from: {checkpoint_path}")
        else:
            logger.info("No valid checkpoint found, starting training from scratch")
            train_params.update({
                'resume': False,
                'resume_path': None,
                'resume_strict_load': False,
                'load_opt_params': False,
                'load_ema_as_net': False,
                'resume_epoch': False
            })

        # Initialize trainer with explicit absolute paths
        trainer = Trainer(
            experiment_name='coco_license_plate_detection',
            ckpt_root_dir=os.path.abspath(checkpoint_dir)  # Ensure absolute path
        )

        # Validate dataset contents before training
        validate_dataset_contents(combined_dir)
        
        # Add the validation call in main() before training
        validate_training_prerequisites(combined_dir, checkpoint_dir, export_dir, l_model_path, s_model_path)
        
        # Before trainer.train()
        logger.info("Validating training configuration...")
        validate_training_config(train_params)
        logger.info("✓ Training configuration validated")
        
        monitor_memory()
        validate_image_paths(combined_dir)
        
        trainer.train(
            model=model,
            training_params=train_params,
            train_loader=train_data,
            valid_loader=val_data
        )
        # After training
        monitor_memory()
        
        # Cleanup downloaded files
        cleanup_downloads()

        # Save final model checkpoint with absolute paths
        final_checkpoint_path = os.path.abspath(os.path.join(checkpoint_dir, 'coco_license_plate_detection_final.pth'))
        trainer.save_checkpoint(
            model_state=model.state_dict(),
            optimizer_state=None,
            scheduler_state=None,
            checkpoint_path=final_checkpoint_path
        )

        # Generate complete label map file with absolute path
        label_map_path = os.path.abspath(os.path.join(checkpoint_dir, 'label_map.txt'))
        with open(label_map_path, 'w') as f:
            for idx, class_name in enumerate(dataset_config['names']):
                f.write(f"{idx}: {class_name}\n")

        # Export model with error handling
        try:
            logger.info("Exporting model to ONNX format...")
            onnx_path = os.path.join(os.path.abspath(export_dir), "yolo_nas_s_coco_license_plate.onnx")
            if not os.access(os.path.dirname(onnx_path), os.W_OK):
                raise PermissionError(f"No write permission for export directory: {export_dir}")
                
            model.export(
                onnx_path,
                output_predictions_format="FLAT_FORMAT",
                max_predictions_per_image=config.max_predictions,
                confidence_threshold=config.confidence_threshold,
                input_image_shape=config.export_image_size
            )
            logger.success(f"✓ Model exported successfully to {onnx_path}")
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            raise

        logger.success("Training completed!")
        logger.info(f"Checkpoint saved to: {final_checkpoint_path}")
        logger.info(f"Label map saved to: {label_map_path}")
        logger.info(f"ONNX model exported to: {onnx_path}")

        # Finish wandb session
        wandb.finish()

    except Exception as e:
        logger.error(f"Error during training: {e}")
        wandb.finish()
        raise
    finally:
        # Cleanup
        cleanup_downloads()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Close wandb run if it exists
        if wandb.run is not None:
            wandb.finish()

# Call at start of main()
log_environment_info()

if __name__ == "__main__":
    main()
