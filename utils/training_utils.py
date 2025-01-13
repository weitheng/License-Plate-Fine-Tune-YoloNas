import os
import torch
import logging
import pynvml
import hashlib
import sys
import super_gradients
from typing import Dict

logger = logging.getLogger(__name__)

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
    logger.info("=== Environment Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"SuperGradients version: {super_gradients.__version__}")
    logger.info("===========================")