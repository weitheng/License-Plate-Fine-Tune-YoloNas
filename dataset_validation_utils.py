import os
import logging
from typing import Dict
from tqdm import tqdm
import shutil
import torch
import gc

logger = logging.getLogger(__name__)

# Constants for dataset validation
EXPECTED_LP_TRAIN = 25470
EXPECTED_LP_VAL = 1073
EXPECTED_COCO_TRAIN = 85000
EXPECTED_COCO_VAL = 5000
EXPECTED_TOTAL_TRAIN = EXPECTED_COCO_TRAIN + EXPECTED_LP_TRAIN  # 85000 + 25470 = 110470
EXPECTED_TOTAL_VAL = EXPECTED_COCO_VAL + EXPECTED_LP_VAL  # 5000 + 1073 = 6073

def validate_final_dataset(combined_dir: str, skip_lp_checks: bool = False) -> Dict[str, Dict[str, int]]:
    """
    Validate the final combined dataset structure and count files.
    Returns statistics about the dataset.
    """
    logger.info("Validating final dataset structure...")
    
    stats = {
        'train': {'total': 0},
        'val': {'total': 0}
    }
    
    for split in ['train', 'val']:
        images_dir = os.path.join(combined_dir, 'images', split)
        labels_dir = os.path.join(combined_dir, 'labels', split)
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            raise RuntimeError(f"Missing directory: {images_dir} or {labels_dir}")
            
        # Count total files
        image_files = [f for f in os.listdir(images_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        label_files = [f for f in os.listdir(labels_dir) 
                      if f.endswith('.txt')]
        
        total_images = len(image_files)
        total_labels = len(label_files)
        
        if total_images != total_labels:
            raise RuntimeError(f"Mismatch in total files for {split}: {total_images} images vs {total_labels} labels")
        
        stats[split]['total'] = total_images
        
        # Validate total counts
        expected_total = EXPECTED_TOTAL_TRAIN if split == 'train' else EXPECTED_TOTAL_VAL
        
        # Only do strict validation if not skipping checks
        if not skip_lp_checks:
            if total_images != expected_total:
                raise RuntimeError(
                    f"Incorrect number of images in {split} split. "
                    f"Found {total_images}, expected {expected_total}"
                )
        else:
            # When skipping checks, just verify we have enough images
            if total_images < expected_total:
                raise RuntimeError(
                    f"Insufficient images in {split} split. "
                    f"Found {total_images}, need at least {expected_total}"
                )
        
        logger.info(f"{split} split statistics:")
        logger.info(f"  - Total Images: {total_images}")
        if not skip_lp_checks:
            logger.info(f"  - Expected Total: {expected_total}")
    
    logger.success("✓ Dataset validation complete")
    return stats

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
                
            if not validate_coco_structure(coco_dir, num_images=85000):
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
        expected_total_train = 95470  # 85000 COCO + 25470 license plate images
        expected_total_val = 6073   # 5000 COCO + 1073 license plate images
        
        # Check existing images in combined directory
        try:
            total_train_images = len(os.listdir(os.path.join(combined_dir, 'images/train')))
            total_val_images = len(os.listdir(os.path.join(combined_dir, 'images/val')))
            train_lp_images = len([f for f in os.listdir(os.path.join(combined_dir, 'images/train')) 
                                 if f.startswith('lp_')])
            val_lp_images = len([f for f in os.listdir(os.path.join(combined_dir, 'images/val')) 
                               if f.startswith('lp_')])
            
            # Calculate COCO images (non-lp_ prefixed images)
            train_coco_images = total_train_images - train_lp_images
            val_coco_images = total_val_images - val_lp_images
            
            logger.info("\n=== Final Dataset Verification ===")
            logger.info(f"COCO Training: {train_coco_images}/85000")
            logger.info(f"COCO Validation: {val_coco_images}/5000")
            logger.info(f"License Plate Training: {train_lp_images}/{expected_lp_train}")
            logger.info(f"License Plate Validation: {val_lp_images}/{expected_lp_val}")
            logger.info(f"Total Training: {total_train_images}/{expected_total_train}")
            logger.info(f"Total Validation: {total_val_images}/{expected_total_val}")
            
            # First verify/fix COCO dataset
            if train_coco_images < 85000:
                logger.warning(f"Missing COCO training images. Found {train_coco_images}/85000")
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
