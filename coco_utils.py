import os
import logging
import coloredlogs
from pycocotools.coco import COCO
from tqdm import tqdm
import shutil
from typing import Dict

# Setup minimal logging for this module
def setup_coco_logging():
    """Setup logging with colored output for COCO utilities"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logger = logging.getLogger('coco_utils')
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

logger = setup_coco_logging()

def validate_coco_structure(coco_dir, num_images=85000):
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
    
def convert_coco_to_yolo(coco_dir: str, target_dir: str, num_images=85000) -> None:
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
