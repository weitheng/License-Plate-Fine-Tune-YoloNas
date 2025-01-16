import os
import torch
import wandb
import requests
import zipfile
import logging
import time
import coloredlogs
import psutil
import hashlib
import argparse
import textwrap
import random
import cv2

from super_gradients.training import Trainer, models
from super_gradients.common.object_names import Models
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val
)
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.utils.callbacks import PhaseCallback, Phase
from yolo_training_utils import (
    assess_hardware_capabilities, load_dataset_config, setup_directories,
    validate_cuda_setup, monitor_gpu, verify_checksum,
    validate_path_is_absolute, validate_training_config,
    log_environment_info, cleanup_downloads, monitor_memory
)
from torch.optim.lr_scheduler import OneCycleLR
from pycocotools.coco import COCO
from tqdm import tqdm
from typing import Optional, List, Dict, Any, Tuple

from config import TrainingConfig
from download_utils import download_model_weights, download_coco_subset
from coco_utils import validate_coco_structure, diagnose_coco_dataset, convert_coco_to_yolo, check_coco_dataset
from remove_prefix import remove_lp_prefix
from augmentations import get_transforms
from validation_utils import (
    validate_training_prerequisites, verify_dataset_structure, validate_image_paths
)


# Constants for dataset validation
EXPECTED_LP_TRAIN = 25470
EXPECTED_LP_VAL = 1073
EXPECTED_COCO_TRAIN = 85000
EXPECTED_COCO_VAL = 5000
EXPECTED_TOTAL_TRAIN = EXPECTED_COCO_TRAIN + EXPECTED_LP_TRAIN  # 85000 + 25470 = 110470
EXPECTED_TOTAL_VAL = EXPECTED_COCO_VAL + EXPECTED_LP_VAL  # 5000 + 1073 = 6073

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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train YOLO-NAS model on COCO and License Plate dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
            Example usage:
              %(prog)s  # Normal run with checkpoint resuming
              %(prog)s --skip-lp-checks  # Skip license plate checks if already processed
              %(prog)s --no-resume  # Start training from scratch
              %(prog)s --skip-lp-checks --no-resume  # Skip checks and start fresh
            Note: Use --skip-lp-checks only if you have already run remove_prefix.py
            '''))
    parser.add_argument('--skip-lp-checks', action='store_true',
                       help='Skip license plate dataset checks and prefix removal (use if already processed)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start training from scratch instead of resuming from checkpoint')
    return parser.parse_args()

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

def setup_checkpoint_resuming(checkpoint_dir: str, train_params: dict, force_new: bool = False) -> dict:
    """
    Setup checkpoint resuming logic with proper validation.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        train_params: Current training parameters
        force_new: If True, ignore existing checkpoints and start fresh
        
    Returns:
        Updated training parameters dict
    """
    if force_new:
        logger.info("Starting fresh training as requested (--no-resume)")
        return {**train_params, 'resume': False, 'resume_path': None}

    # First find the experiment directory (most recent RUN_* directory)
    experiment_name = 'coco_license_plate_detection'  # Must match Trainer's experiment_name
    exp_dir = os.path.join(checkpoint_dir, experiment_name)
    
    if not os.path.exists(exp_dir):
        logger.info(f"No experiment directory found at {exp_dir}")
        return train_params
        
    # Find the most recent RUN directory
    run_dirs = [d for d in os.listdir(exp_dir) if d.startswith('RUN_')]
    if not run_dirs:
        logger.info("No previous run directories found")
        return train_params
        
    # Sort by timestamp (newest first)
    run_dirs.sort(reverse=True)
    latest_run = os.path.join(exp_dir, run_dirs[0])
    
    # Look for checkpoints in the run directory
    checkpoint_path = os.path.join(latest_run, 'ckpt_latest.pth')
    best_checkpoint_path = os.path.join(latest_run, 'ckpt_best.pth')
    
    # Check for both latest and best checkpoints
    available_checkpoints = {
        'latest': checkpoint_path if os.path.exists(checkpoint_path) else None,
        'best': best_checkpoint_path if os.path.exists(best_checkpoint_path) else None
    }
    
    if not any(available_checkpoints.values()):
        logger.info(f"No checkpoints found in latest run directory: {latest_run}")
        return train_params
        
    # Verify checkpoint files
    valid_checkpoints = {}
    for name, path in available_checkpoints.items():
        if path and verify_checkpoint(path):
            valid_checkpoints[name] = path
            logger.info(f"Found valid {name} checkpoint: {path}")
    
    if not valid_checkpoints:
        logger.warning("Found checkpoint files but they are invalid - starting from scratch")
        return train_params
        
    # Prefer best checkpoint over latest if both exist
    chosen_checkpoint = valid_checkpoints.get('best', valid_checkpoints.get('latest'))
    
    # Update training parameters for resuming
    train_params.update({
        'resume': True,
        'resume_path': chosen_checkpoint,
        'resume_strict_load': False,  # Allow flexible loading
        'load_opt_params': True,      # Load optimizer state
        'load_ema_as_net': False,     # Don't load EMA weights as main weights
        'resume_epoch': True          # Continue from the last epoch
    })
    
    logger.info(f"Will resume training from checkpoint: {chosen_checkpoint}")
    
    # Verify checkpoint content
    try:
        checkpoint = torch.load(chosen_checkpoint, map_location='cpu')
        expected_keys = ['net', 'epoch', 'optimizer_state_dict']
        if not all(key in checkpoint for key in expected_keys):
            logger.warning("Checkpoint missing expected keys - may cause issues")
        else:
            logger.info(f"Resuming from epoch {checkpoint['epoch']}")
            logger.info(f"Found in run directory: {latest_run}")
    except Exception as e:
        logger.error(f"Error verifying checkpoint contents: {e}")
        logger.warning("Starting training from scratch")
        return {**train_params, 'resume': False}
    
    return train_params
    
def verify_checkpoint(checkpoint_path: str, is_model_weights: bool = False) -> bool:
    """
    Verify checkpoint file is valid and contains required data
    
    Args:
        checkpoint_path: Path to checkpoint file
        is_model_weights: If True, validates as model weights file instead of training checkpoint
    """
    try:
        if not os.path.exists(checkpoint_path):
            return False
            
        # Check file size
        if os.path.getsize(checkpoint_path) < 1000:  # Arbitrary minimum size
            logger.warning(f"Checkpoint file too small: {checkpoint_path}")
            return False
            
        # Try loading the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if is_model_weights:
            # For model weights, just verify it's a valid state dict
            if not isinstance(checkpoint, dict):
                logger.warning(f"Invalid model weights format in {checkpoint_path}")
                return False
            return True
        else:
            # For training checkpoints, check for required keys
            required_keys = ['net', 'epoch', 'optimizer_state_dict']
            if not all(key in checkpoint for key in required_keys):
                logger.warning(f"Checkpoint missing required keys: {checkpoint_path}")
                return False
                
            # Verify model state dict
            if not isinstance(checkpoint['net'], dict):
                logger.warning("Invalid model state dict in checkpoint")
                return False
                
            return True
    except Exception as e:
        logger.error(f"Error verifying checkpoint {checkpoint_path}: {e}")
        return False

def main():
    try:
        # Parse command line arguments
        args = parse_args()
        
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
        # Create required subdirectories in combined_dir
        for split in ['train', 'val']:
            for subdir in ['images', 'labels']:
                path = os.path.join(combined_dir, subdir, split)
                os.makedirs(path, exist_ok=True)
                if not os.access(path, os.W_OK):
                    raise PermissionError(f"No write permission for directory: {path}")

        # Use absolute paths everywhere
        yaml_path = os.path.join(current_dir, "license_plate_dataset.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Dataset configuration file not found: {yaml_path}")
        logger.info(f"Using dataset configuration: {yaml_path}")
        
        # Check if combined directory is empty or missing data
        is_combined_empty = any(
            len(os.listdir(os.path.join(combined_dir, subdir, split))) == 0
            for subdir in ['images', 'labels']
            for split in ['train', 'val']
        )

        if is_combined_empty:
            logger.info("Combined directory is empty or incomplete. Preparing dataset...")
            prepare_combined_dataset()
        else:
            logger.info("Combined directory exists and contains data. Validating...")
            try:
                verify_dataset_structure(combined_dir)
                validate_dataset_contents(combined_dir)
                logger.success("✓ Existing dataset validated")
            except Exception as e:
                logger.warning(f"Dataset validation failed: {e}")
                logger.info("Attempting to recreate dataset...")
                prepare_combined_dataset()

        logger.info("Starting training pipeline...")
        monitor_memory()  # Initial state
        
        # Prepare dataset first
        # prepare_combined_dataset()
        monitor_memory()  # After dataset prep
        
        # After prepare_combined_dataset()
        logger.info("Verifying dataset structure...")
        verify_dataset_structure(combined_dir)
        validate_dataset_contents(combined_dir)
        logger.info("✓ Dataset validation complete")
        monitor_memory()  # After validation
        
        # Validate final dataset before starting training
        logger.info("Performing final dataset validation...")
        dataset_stats = validate_final_dataset(combined_dir, args.skip_lp_checks)
        
        # Only check and remove LP prefix if not skipping LP checks
        if not args.skip_lp_checks:
            logger.info("Removing 'lp_' prefix from files...")
            remove_lp_prefix(combined_dir)
            
            # After prefix removal, validate with skip_lp_checks=True to only check totals
            logger.info("Validating dataset after prefix removal...")
            dataset_stats = validate_final_dataset(combined_dir, skip_lp_checks=True)
        
        # Log dataset statistics
        logger.info("\n=== Final Dataset Statistics ===")
        logger.info(f"Total Training Images: {dataset_stats['train']['total']}")
        logger.info(f"Total Validation Images: {dataset_stats['val']['total']}")
        logger.info("==============================\n")

        # Only proceed with training if validation passes
        if args.skip_lp_checks:
            if dataset_stats['train']['total'] < 85000:  # Minimum expected total
                raise RuntimeError(f"Insufficient training images: {dataset_stats['train']['total']}/85000")
        else:
            if dataset_stats['train']['total'] != EXPECTED_TOTAL_TRAIN:
                raise RuntimeError(
                    f"Incorrect number of training images: {dataset_stats['train']['total']}, "
                    f"expected {EXPECTED_TOTAL_TRAIN}"
                )
            if dataset_stats['val']['total'] != EXPECTED_TOTAL_VAL:
                raise RuntimeError(
                    f"Incorrect number of validation images: {dataset_stats['val']['total']}, "
                    f"expected {EXPECTED_TOTAL_VAL}"
                )

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
                'transforms': get_transforms(dataset_config, config.input_size, is_training=True)
            },
            dataloader_params={
                'batch_size': hw_params['batch_size'],  # Use hardware-assessed batch size config.batch_size,
                'num_workers': hw_params['num_workers'],  # Use hardware-assessed workers config.num_workers,
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
                'transforms': get_transforms(dataset_config, config.input_size, is_training=False)
            },
            dataloader_params={
                'batch_size': hw_params['batch_size'],  # Use hardware-assessed batch size config.batch_size,
                'num_workers': hw_params['num_workers'],  # Use hardware-assessed workers config.num_workers,
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
        try:
            logger.info("Checking for existing checkpoints...")
            train_params = setup_checkpoint_resuming(checkpoint_dir, train_params, force_new=args.no_resume)
        except Exception as e:
            logger.error(f"Error setting up checkpoint resuming: {e}")
            logger.warning("Starting training from scratch")
            train_params.update({
                'resume': False,
                'resume_path': None
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
        
        if args.skip_lp_checks:
            logger.warning("License plate checks are disabled. Assuming all files are properly prepared.")
        
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
