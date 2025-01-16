import os
import torch
import wandb
import logging
import time
import coloredlogs
import argparse
import textwrap
import random
import cv2
import torch.multiprocessing as mp
import numpy as np

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
    validate_cuda_setup, monitor_gpu, setup_cuda_error_handling,
    validate_path_is_absolute, validate_training_config,
    log_environment_info, cleanup_downloads, monitor_memory,
    verify_checkpoint
)
from torch.optim.lr_scheduler import OneCycleLR
from typing import Optional, List, Dict, Any, Tuple

from config import TrainingConfig
from download_utils import download_model_weights, download_coco_subset
from coco_utils import validate_coco_structure, diagnose_coco_dataset, convert_coco_to_yolo, check_coco_dataset
from remove_prefix import remove_lp_prefix
from augmentations import get_transforms
from validation_utils import (
    validate_training_prerequisites, verify_dataset_structure, validate_image_paths
)
from checkpoint_utils import setup_checkpoint_resuming
from dataset_validation_utils import (
    validate_final_dataset, prepare_combined_dataset,
    validate_dataset, validate_dataset_contents,
    EXPECTED_LP_TRAIN, EXPECTED_LP_VAL,
    EXPECTED_COCO_TRAIN, EXPECTED_COCO_VAL,
    EXPECTED_TOTAL_TRAIN, EXPECTED_TOTAL_VAL
)

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

def create_dataloader_with_memory_management(dataset_params, dataloader_params, is_training=True):
    """Create dataloader with memory management"""
    if torch.cuda.is_available():
        # Ensure spawn method for workers
        dataloader_params['multiprocessing_context'] = 'spawn'
        
        # Calculate safe batch size based on available memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = total_memory - torch.cuda.memory_allocated()
        
        # More conservative memory per sample estimate
        estimated_memory_per_sample = 750 * 1024 * 1024  # 750MB per sample
        max_safe_batch_size = max(1, int(available_memory / estimated_memory_per_sample))
        
        # Update batch size if needed
        original_batch_size = dataloader_params['batch_size']
        dataloader_params['batch_size'] = min(
            original_batch_size,
            max_safe_batch_size
        )
        if dataloader_params['batch_size'] < original_batch_size:
            logger.warning(f"Reduced batch size from {original_batch_size} to {dataloader_params['batch_size']} due to memory constraints")
        
        # Enable pinned memory but disable persistent workers
        dataloader_params['pin_memory'] = True
        dataloader_params['persistent_workers'] = False
        
        # Reduce number of workers and increase timeout
        dataloader_params['num_workers'] = min(
            dataloader_params['num_workers'],
            os.cpu_count() // 2 or 1  # Use half of available CPU cores
        )
        
        # Increase timeout and add prefetch factor
        dataloader_params['timeout'] = 120  # Increase timeout to 120 seconds
        dataloader_params['prefetch_factor'] = 2  # Reduce prefetch factor
        
        # Add worker init function to set CPU affinity
        dataloader_params['worker_init_fn'] = worker_init_fn
    
    # Extract max_targets from dataloader_params if present
    max_targets = dataloader_params.pop('max_targets', None)
    
    try:
        # Create the dataloader
        dataloader = (coco_detection_yolo_format_train if is_training else coco_detection_yolo_format_val)(
            dataset_params=dataset_params,
            dataloader_params=dataloader_params
        )
        
        # If max_targets was specified, set it on the dataset
        if max_targets is not None and hasattr(dataloader.dataset, 'max_targets'):
            dataloader.dataset.max_targets = max_targets
        
        return dataloader
    except Exception as e:
        logger.error(f"Error creating dataloader: {e}")
        raise

def worker_init_fn(worker_id):
    """Initialize worker process"""
    # Set different seed for each worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    # Try to set CPU affinity if possible
    try:
        import os
        import psutil
        
        process = psutil.Process()
        # Get the number of CPU cores
        cpu_count = os.cpu_count() or 1
        # Assign worker to specific CPU core
        worker_core = worker_id % cpu_count
        process.cpu_affinity([worker_core])
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Could not set CPU affinity: {e}")

def main():
    try:
        # Remove the log_environment_info() call from here since it's called by SuperGradients
        if torch.cuda.is_available():
            mp.set_start_method('spawn', force=True)
            logger.info("Set multiprocessing start method to 'spawn'")

        # Parse command line arguments
        args = parse_args()
        
        validate_cuda_setup()
        # Check GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        if device == "cpu":
            logger.warning("No GPU detected - training will be slow!")
            
        # Add CUDA error handling setup right after device check
        if device == "cuda":
            setup_cuda_error_handling()
            logger.info("CUDA error handling configured")

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

        # Initialize wandb first with specific settings
        try:
            logger.info("Initializing Weights & Biases...")
            wandb.init(
                project="license-plate-detection",
                name="yolo-nas-s-coco-finetuning",
                config=train_params,
                resume=True if os.path.exists(checkpoint_path) else False
            )
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
            'warmup_initial_lr': config.initial_lr / 100,
            'lr_warmup_epochs': config.warmup_epochs,
            'initial_lr': {
                'backbone': config.initial_lr * 0.1,  # Lower learning rate for backbone
                'default': config.initial_lr  # Default learning rate for other layers
            },
            'lr_mode': 'cosine',
            'max_epochs': config.num_epochs,
            'early_stopping_patience': config.early_stopping_patience,
            'mixed_precision': True if torch.cuda.is_available() else False,
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
            'optimizer': 'SGD',
            'optimizer_params': {
                'weight_decay': config.weight_decay,
                'momentum': 0.937,
                'nesterov': True
            },
            'gradient_clip_val': 1.0,
            'zero_weight_decay_on_bias_and_bn': True,
            'loss_logging_items_names': ['Loss', 'Precision', 'Recall'],
            'accelerator': 'cuda' if torch.cuda.is_available() else 'cpu',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'lr_cooldown_epochs': 10,
            'multiply_head_lr': 1.0,
            'criterion_params': {
                'alpha': 0.25,
                'gamma': 1.5
            },
        }

        # Create base dataloader first
        train_data = create_dataloader_with_memory_management(
            dataset_params={
                'data_dir': combined_dir,
                'images_dir': 'images/train',
                'labels_dir': 'labels/train',
                'classes': dataset_config['names'],
                'input_dim': config.input_size,
                'transforms': []  # Start with empty transforms
            },
            dataloader_params={
                'batch_size': hw_params['batch_size'],
                'num_workers': hw_params['num_workers'],
                'shuffle': True,
                'pin_memory': torch.cuda.is_available(),
                'drop_last': True
            }
        )

        # Create transforms with dataloader
        transforms = get_transforms(
            dataset_config, 
            config.input_size, 
            is_training=True,
            dataloader=train_data
        )

        # Update dataloader with transforms
        train_data.dataset.transforms = transforms

        val_data = create_dataloader_with_memory_management(
            dataset_params={
                'data_dir': combined_dir,  # Using absolute path to combined dataset
                'images_dir': 'images/val',
                'labels_dir': 'labels/val',
                'classes': dataset_config['names'],
                'input_dim': config.input_size,
                'transforms': get_transforms(dataset_config, config.input_size, is_training=False)
            },
            dataloader_params={
                'batch_size': max(1, hw_params['batch_size'] // 2),  # Reduce validation batch size
                'num_workers': max(1, hw_params['num_workers'] // 2),  # Reduce validation workers
                'shuffle': False,
                'pin_memory': torch.cuda.is_available(),
                'drop_last': False,
                'persistent_workers': False,  # Disable persistent workers for validation
                'max_targets': 100  # Move max_targets here in the dataloader params
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
            ckpt_root_dir=os.path.abspath(checkpoint_dir),
            training_params=train_params
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
