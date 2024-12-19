import os
import multiprocessing
import psutil
import torch
import yaml
from typing import Tuple, Dict

def assess_hardware_capabilities() -> Dict[str, int]:
    """
    Assess available hardware resources and return optimal training parameters.
    
    Returns:
        Dict containing recommended num_workers and batch_size based on hardware
    """
    # Get CPU cores (physical cores * 0.75 for optimal performance)
    cpu_cores = multiprocessing.cpu_count()
    recommended_workers = max(1, int(cpu_cores * 0.75))
    
    # Get available GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
        
        # Calculate batch size based on GPU memory
        # Rough estimation: 1GB can handle batch size of 4 for YOLO-NAS S
        recommended_batch_size = max(1, int((gpu_memory - 2) * 4))  # Reserve 2GB for system
        recommended_batch_size = min(recommended_batch_size, 64)  # Cap at 64
    else:
        recommended_batch_size = 4  # Conservative default for CPU
    
    # Get available RAM
    available_ram = psutil.virtual_memory().available / 1024**3  # Convert to GB
    
    # Adjust workers based on available RAM (each worker needs ~2GB)
    max_workers_ram = max(1, int(available_ram / 2))
    recommended_workers = min(recommended_workers, max_workers_ram)
    
    return {
        'num_workers': recommended_workers,
        'batch_size': recommended_batch_size
    }

def load_dataset_config(config_path: str) -> dict:
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

def setup_directories(base_dir: str) -> Tuple[str, str]:
    """
    Create necessary directories for training.
    
    Args:
        base_dir: Base directory for the project
    
    Returns:
        Tuple of (checkpoint_dir, export_dir)
    """
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    export_dir = os.path.join(base_dir, "exported_models")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)
    
    return checkpoint_dir, export_dir