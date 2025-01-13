import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def verify_dataset_structure(data_dir: str) -> bool:
    """Verify dataset structure and return True if populated, False if empty/incomplete"""
    required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    
    # First ensure directories exist
    for dir_path in required_dirs:
        full_path = os.path.join(data_dir, dir_path)
        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)
            logger.info(f"Created directory: {full_path}")
            return False
            
    # Check if directories have content
    for dir_path in required_dirs:
        full_path = os.path.join(data_dir, dir_path)
        if not os.listdir(full_path):
            logger.info(f"Directory is empty: {full_path}")
            return False
            
    return True

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

def validate_dataset_size(dataset_stats: Dict, skip_lp_checks: bool) -> None:
    """Validate dataset size meets minimum requirements"""
    min_train_images = 1000
    min_val_images = 100
    
    if dataset_stats['train']['total'] < min_train_images:
        raise ValueError(f"Insufficient training images. Found {dataset_stats['train']['total']}, minimum required: {min_train_images}")
    if dataset_stats['val']['total'] < min_val_images:
        raise ValueError(f"Insufficient validation images. Found {dataset_stats['val']['total']}, minimum required: {min_val_images}")

def handle_license_plate_data(combined_dir: str, dataset_stats: Dict) -> None:
    """
    Handle license plate specific dataset operations.
    
    Args:
        combined_dir: Path to the combined dataset directory
        dataset_stats: Statistics about the dataset
    """
    try:
        # Check if we have the expected number of license plate images
        expected_lp_train = 25470
        expected_lp_val = 1073
        
        train_lp = dataset_stats['train']['license_plate']
        val_lp = dataset_stats['val']['license_plate']
        
        if train_lp != expected_lp_train:
            logger.warning(f"Unexpected number of license plate training images: {train_lp} (expected {expected_lp_train})")
            
        if val_lp != expected_lp_val:
            logger.warning(f"Unexpected number of license plate validation images: {val_lp} (expected {expected_lp_val})")
            
        # Verify label format for license plate images
        for split in ['train', 'val']:
            labels_dir = os.path.join(combined_dir, 'labels', split)
            lp_labels = [f for f in os.listdir(labels_dir) if f.startswith('lp_')]
            
            for label_file in lp_labels:
                label_path = os.path.join(labels_dir, label_file)
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        if not lines:
                            logger.warning(f"Empty label file: {label_file}")
                            continue
                        
                        for line in lines:
                            # Verify YOLO format: class x_center y_center width height
                            parts = line.strip().split()
                            if len(parts) != 5:
                                raise ValueError(f"Invalid format in {label_file}: {line}")
                            
                            # Verify values are float and in range [0,1]
                            class_id = int(parts[0])
                            coords = [float(x) for x in parts[1:]]
                            if not all(0 <= x <= 1 for x in coords):
                                raise ValueError(f"Coordinates out of range in {label_file}: {line}")
                            
                except Exception as e:
                    logger.error(f"Error validating {label_file}: {e}")
                    raise
                    
        logger.success("✓ License plate data validation complete")
        
    except Exception as e:
        logger.error(f"Error handling license plate data: {e}")
        raise