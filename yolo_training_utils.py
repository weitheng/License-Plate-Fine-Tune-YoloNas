import os
import multiprocessing
import psutil
import torch
import yaml
import pynvml
from typing import Tuple, Dict, Any, Optional
import logging
import sys
import super_gradients

logger = logging.getLogger(__name__)

def assess_hardware_capabilities() -> Dict[str, int]:
    """
    Assess available hardware resources and return conservative training parameters.

    Returns:
        Dict containing recommended num_workers and batch_size based on hardware
    """
    try:
        # Get CPU cores (physical cores * 0.2 for conservative performance)
        cpu_cores = multiprocessing.cpu_count()
        recommended_workers = max(1, int(cpu_cores * 0.2))

        # Get available GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
            
            # Calculate batch size based on GPU memory
            # Conservative estimation:
            # < 6GB: batch_size = 2
            # 6-8GB: batch_size = 4
            # 8-12GB: batch_size = 8
            # 12-16GB: batch_size = 16
            # 16-24GB: batch_size = 24
            # >24GB: batch_size = 32
            if gpu_memory < 6:
                recommended_batch_size = 2
            elif gpu_memory < 8:
                recommended_batch_size = 4
            elif gpu_memory < 12:
                recommended_batch_size = 8
            elif gpu_memory < 16:
                recommended_batch_size = 16
            elif gpu_memory < 24:
                recommended_batch_size = 16 #decreased from 24 to 16
            else:
                recommended_batch_size = 32
                
            # Log GPU information
            logger.info(f"GPU Memory: {gpu_memory:.1f}GB")
            logger.info(f"Recommended batch size: {recommended_batch_size}")
        else:
            recommended_batch_size = 2  # Conservative default for CPU
            logger.warning("No GPU detected - using minimal batch size")

        # Get available RAM
        available_ram = psutil.virtual_memory().available / 1024**3  # Convert to GB
        
        # Adjust workers based on available RAM (each worker needs ~4GB)
        max_workers_ram = max(1, int(available_ram / 4))
        recommended_workers = min(recommended_workers, max_workers_ram)

        # Log final recommendations
        logger.info(f"Available RAM: {available_ram:.1f}GB")
        logger.info(f"CPU cores: {cpu_cores}")
        logger.info(f"Recommended workers: {recommended_workers}")
        
        return {
            'num_workers': recommended_workers,
            'batch_size': recommended_batch_size
        }
    except Exception as e:
        logger.error(f"Error assessing hardware: {e}")
        # Return conservative defaults
        return {
            'num_workers': 1,
            'batch_size': 4
        }

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
    
    # Set optimal CUDA settings for large datasets
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.max_split_size_mb = 512
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Log optimizations
    logger.info("CUDA optimizations enabled:")
    logger.info("✓ cuDNN benchmark mode")
    logger.info("✓ Memory split size: 512MB")
    logger.info("✓ TF32 enabled (if supported)")
    
    # Log GPU info
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")

def monitor_gpu():
    """Monitor GPU temperature and utilization"""
    if torch.cuda.is_available():
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            logger.info(f"GPU Temperature: {temp}°C, Utilization: {util.gpu}%")
        except Exception as e:
            logger.warning(f"Could not monitor GPU metrics: {e}")

def validate_path_is_absolute(path: str, description: str) -> None:
    """Validate that a path is absolute and exists"""
    if not os.path.isabs(path):
        raise ValueError(f"{description} must be an absolute path. Got: {path}")
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    if not os.access(directory, os.W_OK):
        raise PermissionError(f"No write permission for {description} directory: {directory}")

def validate_training_config(train_params: dict) -> None:
    """Validate training configuration parameters"""
    required_keys = ['resume', 'loss', 'metric_to_watch',
                    'valid_metrics_list', 'max_epochs', 'initial_lr']
    for key in required_keys:
        if key not in train_params:
            raise ValueError(f"Missing required training parameter: {key}")
            
    # Validate numeric parameters
    if train_params['initial_lr'] <= 0:
        raise ValueError("Learning rate must be positive")
    if train_params['max_epochs'] <= 0:
        raise ValueError("Number of epochs must be positive")

def load_dataset_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate dataset configuration.
    
    Args:
        config_path: Path to the YAML config file
    
    Returns:
        Dataset configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    required_keys = ['names', 'nc', 'train', 'val']
    if not all(key in config for key in required_keys):
        raise ValueError(f"Dataset config must contain these keys: {required_keys}")
    
    return config

def setup_directories(base_path: str) -> Tuple[str, str]:
    """Setup checkpoint and export directories"""
    try:
        base_path = os.path.abspath(base_path)
        checkpoint_dir = os.path.join(base_path, 'checkpoints')
        export_dir = os.path.join(base_path, 'export')
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(export_dir, exist_ok=True)
        
        # Verify write permissions
        if not os.access(checkpoint_dir, os.W_OK):
            raise PermissionError(f"No write permission for checkpoint directory: {checkpoint_dir}")
        if not os.access(export_dir, os.W_OK):
            raise PermissionError(f"No write permission for export directory: {export_dir}")
            
        return os.path.abspath(checkpoint_dir), os.path.abspath(export_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to setup directories: {str(e)}")

def validate_yaml_schema(config: Dict[str, Any]) -> None:
    """Validate YAML configuration schema"""
    required_structure = {
        'names': list,
        'nc': int,
        'train': str,
        'val': str
    }
    
    for key, expected_type in required_structure.items():
        if key not in config:
            raise ValueError(f"Missing required key: {key}")
        if not isinstance(config[key], expected_type):
            raise TypeError(f"Invalid type for {key}: expected {expected_type}, got {type(config[key])}")

def log_environment_info():
    """Log environment and library versions"""
    logger.info("=== Environment Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"SuperGradients version: {super_gradients.__version__}")
    logger.info("===========================")

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

def setup_cuda_error_handling():
    """Setup CUDA error handling and debugging"""
    if torch.cuda.is_available():
        # Enable CUDA error checking
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        # Enable device-side assertions if available
        if hasattr(torch, 'set_device_debug'):
            torch.set_device_debug(True)
        
        # Set more conservative memory settings
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of available memory
        
        # Enable anomaly detection in PyTorch
        torch.autograd.set_detect_anomaly(True)
