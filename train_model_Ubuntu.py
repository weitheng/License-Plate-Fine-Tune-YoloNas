import os
# Disable SuperGradients console logging
os.environ['CONSOLE_LOG_LEVEL'] = 'WARNING'
os.environ['DISABLE_CONSOLE_LOG'] = 'TRUE'
os.environ['DISABLE_SG_LOGGER'] = 'TRUE'

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
import super_gradients

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
from super_gradients.training.utils.callbacks.base_callbacks import PhaseContext
from yolo_training_utils import (
    assess_hardware_capabilities, load_dataset_config, setup_directories,
    validate_cuda_setup, monitor_gpu, setup_cuda_error_handling,
    validate_path_is_absolute, validate_training_config,
    log_environment_info, cleanup_downloads, monitor_memory,
    verify_checkpoint, GPUMonitorCallback, pin_memory,
    check_batch_device, GradientMonitorCallback
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

# First, move the callback classes to the top of the file (after imports)
class GradientClippingCallback(PhaseCallback):
    def __init__(self, clip_value=0.1):
        super().__init__(phase=Phase.TRAIN_BATCH_STEP)
        self.clip_value = clip_value

    def __call__(self, context: PhaseContext):
        if hasattr(context.trainer, 'net'):
            torch.nn.utils.clip_grad_norm_(
                context.trainer.net.parameters(),
                self.clip_value
            )

class LRMonitorCallback(PhaseCallback):
    def __init__(self):
        super().__init__(phase=Phase.TRAIN_BATCH_STEP)
        self.logger = logging.getLogger(__name__)
        self.last_log = 0

    def __call__(self, context: PhaseContext):
        if hasattr(context.trainer, 'optimizer'):
            current_time = time.time()
            if current_time - self.last_log >= 30:  # Log every 30 seconds
                for param_group in context.trainer.optimizer.param_groups:
                    current_lr = param_group.get('lr', 0)
                    self.logger.info(f"Current learning rate: {current_lr:.2e}")
                    if wandb.run is not None:
                        wandb.log({'learning_rate': current_lr})
                self.last_log = current_time

class LRSchedulerCallback(PhaseCallback):
    def __init__(self):
        super().__init__(phase=Phase.TRAIN_EPOCH_END)
        
    def __call__(self, context: PhaseContext):
        if context.epoch < context.training_params['lr_warmup_epochs']:
            return
            
        # Get current learning rate
        current_lr = context.optimizer.param_groups[0]['lr']
        
        # Check for NaN values in gradients
        has_nan = False
        for param in context.net.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan = True
                break
        
        # If NaN detected, reduce learning rate
        if has_nan:
            new_lr = current_lr * 0.1
            logger.warning(f"NaN detected - reducing learning rate to {new_lr}")
            for param_group in context.optimizer.param_groups:
                param_group['lr'] = new_lr

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

def worker_init_fn(worker_id):
    """Initialize worker process"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    # Set lower priority for worker processes
    try:
        os.nice(10)  # Lower priority
    except AttributeError:
        pass  # os.nice not available on Windows
        
    # Clear CUDA cache for each worker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def create_dataloader_with_memory_management(dataset_params, dataloader_params, is_training=True):
    """Create dataloader with memory management"""
    try:
        logger.info(f"Creating {'training' if is_training else 'validation'} dataloader...")
        logger.info(f"Dataset params: {dataset_params}")
        logger.info(f"Dataloader params: {dataloader_params}")
        
        if torch.cuda.is_available():
            # Use more conservative settings
            dataloader_params.update({
                'num_workers': min(8, os.cpu_count()),  # Increase workers
                'pin_memory': True,
                'persistent_workers': True,
                'prefetch_factor': 4,  # Increase prefetch
                'timeout': 300,
                'multiprocessing_context': 'spawn',
                'worker_init_fn': worker_init_fn
            })
        else:
            # CPU settings
            dataloader_params.update({
                'num_workers': 0,
                'pin_memory': False,
                'persistent_workers': False
            })

        # Create the dataloader with error handling
        dataloader = (coco_detection_yolo_format_train if is_training else coco_detection_yolo_format_val)(
            dataset_params=dataset_params,
            dataloader_params=dataloader_params
        )
        
        # Verify the dataloader
        try:
            logger.info("Testing dataloader...")
            # Test the dataloader with a single batch
            next(iter(dataloader))
            logger.info(f"{'Training' if is_training else 'Validation'} dataloader initialized successfully")
            logger.info(f"Dataloader length: {len(dataloader)}")
            return dataloader
        except Exception as e:
            logger.error(f"Failed to load first batch: {e}")
            logger.error("Dataloader verification failed", exc_info=True)
            raise
            
    except Exception as e:
        logger.error(f"Error creating dataloader: {e}")
        logger.error("Dataloader creation failed", exc_info=True)
        raise

def create_initial_transforms(dataset_config, input_size):
    """Create initial transforms without mosaic augmentation"""
    return get_transforms(
        dataset_config, 
        input_size, 
        is_training=True, 
        dataloader=None,
        skip_mosaic=True
    )

def main():
    try:
        # Set multiprocessing start method first
        if torch.cuda.is_available():
            mp.set_start_method('spawn', force=True)
        
        # Setup device using super_gradients before any other operations
        device = "cuda" if torch.cuda.is_available() else "cpu"
        super_gradients.setup_device(device=device)
        
        # Log environment info only once and only after device setup
        if not hasattr(main, '_logged'):
            log_environment_info()
            main._logged = True
        
        # Parse command line arguments
        args = parse_args()
        
        validate_cuda_setup()
        logger.info(f"Using device: {device}")
        
        if device == "cpu":
            logger.warning("No GPU detected - training will be slow!")
        else:
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

        # Initialize model with more careful initialization
        try:
            logger.info("Initializing model...")
            model = models.get(Models.YOLO_NAS_S, 
                              num_classes=81,
                              pretrained_weights="coco")
            
            # Initialize weights with more stable method
            def init_weights(m):
                if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)
            
            # Apply custom initialization only to new layers
            for name, module in model.named_children():
                if 'head' in name:  # Only initialize detection head
                    module.apply(init_weights)
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(model, 'use_gradient_checkpointing'):
                model.use_gradient_checkpointing()
            
            if torch.cuda.is_available():
                # Remove channels_last and compile for now
                model.train()
                
                # Use more conservative CUDA settings
                torch.backends.cudnn.benchmark = False  # Disable for stability
                torch.backends.cudnn.deterministic = True  # Enable for reproducibility
                torch.backends.cudnn.enabled = True
                
                # More conservative memory settings
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(0.75)  # Reduced from 0.85
                
                # Initialize weights with double precision temporarily
                model = model.double()
                model = model.float()  # Convert back to float
            
            logger.info(f"Model device: {next(model.parameters()).device}")
            logger.success("✓ Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise RuntimeError("Model initialization failed") from e
        
        # After model initialization
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure all CUDA operations are completed
            logger.info(f"Initial GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            logger.info(f"Initial GPU memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")

        # Define loss function with more conservative parameters
        loss_fn = PPYoloELoss(
            use_static_assigner=True,
            num_classes=81,
            reg_max=16,
            iou_loss_weight=1.0  # Further reduced from 2.0
        )
        if torch.cuda.is_available():
            loss_fn = loss_fn.cuda()
            logger.info("Loss function moved to GPU")

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
            'warmup_initial_lr': config.initial_lr * config.warmup_initial_lr_factor,
            'lr_warmup_epochs': config.warmup_epochs,
            'initial_lr': {
                'backbone': config.initial_lr * 0.1,
                'default': config.initial_lr
            },
            'lr_mode': 'cosine',
            'max_epochs': config.num_epochs,
            'early_stopping_patience': config.early_stopping_patience,
            'mixed_precision': False,  # Disable mixed precision temporarily
            'loss': loss_fn,
            'criterion_params': {
                'label_smoothing': 0.05,  # Reduced from 0.1 for stability
                'eps': 1e-7  # Add epsilon through criterion params
            },
            'loss_params': {
                'class_loss_weight': 1.0,
                'iou_loss_weight': 2.0,
                'dfl_loss_weight': 0.5
            },
            'train_metrics_list': [
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
            'greater_metric_to_watch_is_better': True,
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
            'optimizer': 'AdamW',
            'optimizer_params': {
                'weight_decay': config.weight_decay,
                'betas': (0.9, 0.999),  # Back to default betas
                'eps': 1e-8,
                'lr': config.initial_lr,
                'amsgrad': True,
                'foreach': False,  # Disable foreach
                'maximize': False,
                'capturable': False
            },
            'zero_weight_decay_on_bias_and_bn': True,
            'lr_mode': 'cosine',
            'lr_warmup_epochs': config.warmup_epochs,
            'warmup_initial_lr': config.initial_lr * 0.01,  # Start with even lower LR
            'initial_lr': {
                'backbone': config.initial_lr * 0.1,
                'default': config.initial_lr
            },
            'lr_cooldown_epochs': config.lr_cooldown_epochs,
            'min_lr': config.min_lr,
            'max_grad_norm': config.max_grad_norm,
            'multiply_head_lr': config.head_lr_factor,
            'ema': True,
            'ema_params': {
                'decay': config.ema_decay,
                'decay_type': 'threshold',
                'warmup_epochs': config.warmup_epochs,
            },
            'batch_accumulate': 1,  # Start with no accumulation
            'sync_bn': False,  # Disable if using single GPU
            'save_ckpt_epoch_list': [1, 2, 5, 10, 20, 50],
            'phase_callbacks': [
                GradientMonitorCallback(logging_frequency=50, max_grad_norm=config.max_grad_norm),
                GPUMonitorCallback(),
                GradientClippingCallback(clip_value=config.gradient_clip_val),
                LRMonitorCallback(),
                LRSchedulerCallback()
            ],
        }

        # Check for existing checkpoint
        checkpoint_path = os.path.abspath(os.path.join(checkpoint_dir, 'latest_checkpoint.pth'))
        
        # Verify the checkpoint directory exists and is writable
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        if not os.access(checkpoint_dir, os.W_OK):
            raise PermissionError(f"No write permission for checkpoint directory: {checkpoint_dir}")

        # Initialize trainer first
        trainer = Trainer(
            experiment_name='coco_license_plate_detection',
            ckpt_root_dir=os.path.abspath(checkpoint_dir)
        )

        # Then initialize wandb
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

        # Create initial transforms without mosaic
        initial_transforms = create_initial_transforms(dataset_config, config.input_size)
        
        # Create training dataloader with initial transforms
        train_data = create_dataloader_with_memory_management(
            dataset_params={
                'data_dir': combined_dir,
                'images_dir': 'images/train',
                'labels_dir': 'labels/train',
                'classes': dataset_config['names'],
                'input_dim': config.input_size,
                'transforms': initial_transforms
            },
            dataloader_params={
                'batch_size': 4,
                'shuffle': True,
                'drop_last': True,
                'persistent_workers': True,  # Add this
                'pin_memory': True if torch.cuda.is_available() else False
            }
        )
        
        # Now create the full transforms including mosaic
        train_transforms = get_transforms(
            dataset_config, 
            config.input_size, 
            is_training=True,
            dataloader=train_data,
            skip_mosaic=False  # Explicitly enable mosaic
        )
        
        # Update the dataset transforms
        train_data.dataset.transforms = train_transforms
        logger.info("✓ Training transforms updated with mosaic augmentation")
        
        # Create validation dataloader
        val_data = create_dataloader_with_memory_management(
            dataset_params={
                'data_dir': combined_dir,
                'images_dir': 'images/val',
                'labels_dir': 'labels/val',
                'classes': dataset_config['names'],
                'input_dim': config.input_size,
                'transforms': get_transforms(dataset_config, config.input_size, is_training=False)
            },
            dataloader_params={
                'batch_size': 2,  # Even smaller batch size for validation
                'shuffle': False,
                'drop_last': False
            }
        )

        # If dataloaders are created successfully, gradually increase batch size
        if torch.cuda.is_available():
            try:
                # Test with larger batch sizes
                for batch_size in [8, 16, hw_params['batch_size']]:
                    torch.cuda.empty_cache()
                    train_data.batch_sampler.batch_size = batch_size
                    next(iter(train_data))  # Test if it works
                    logger.info(f"Successfully increased batch size to {batch_size}")
            except Exception as e:
                logger.warning(f"Keeping smaller batch size due to: {e}")

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
        
        # Update train_params with device settings
        train_params.update({
            'accelerator': device,
            'device': device,
        })

        # Clear memory before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Before trainer.train()
        try:
            logger.info("Initializing training...")
            
            # Verify model is on correct device
            logger.info(f"Model device: {next(model.parameters()).device}")
            logger.info(f"Using device: {device}")
            
            # Verify dataloaders
            logger.info(f"Training dataloader length: {len(train_data)}")
            logger.info(f"Validation dataloader length: {len(val_data)}")
            
            # Log training parameters
            logger.info("Training parameters:")
            for key, value in train_params.items():
                if isinstance(value, (int, float, str, bool)):
                    logger.info(f"  {key}: {value}")
            
            # Clear CUDA cache before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                logger.info(f"GPU memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
            
            # Verify trainer state
            logger.info(f"Trainer device: {trainer.device}")
            logger.info("Starting training...")
            
            # Pass training_params to the train() method
            trainer.train(
                model=model,
                training_params=train_params,
                train_loader=train_data,
                valid_loader=val_data
            )
            
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            logger.error("Training initialization failed", exc_info=True)
            raise

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

        # After creating both dataloaders
        if torch.cuda.is_available():
            check_batch_device(train_data, "Training")
            check_batch_device(val_data, "Validation")

        # After creating validation dataloader
        if hasattr(val_data.dataset, 'set_processing_params'):
            val_data.dataset.set_processing_params(
                input_dim=config.input_size,
                normalize=True,
                device=device
            )

        # After creating dataloaders
        train_data = pin_memory(train_data)
        val_data = pin_memory(val_data)

        # Add weight decay exemptions for AdamW
        no_decay = ['bias', 'LayerNorm.weight', 'BatchNorm2d.weight']
        train_params['parameter_groups'] = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': config.weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

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

if __name__ == "__main__":
    main()
