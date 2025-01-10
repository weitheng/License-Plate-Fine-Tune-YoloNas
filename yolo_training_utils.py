import os
import multiprocessing
import psutil
import torch
import yaml
from typing import Tuple, Dict, Any, Optional

def assess_hardware_capabilities() -> Dict[str, int]:
    """
    Assess available hardware resources and return conservative training parameters.

    Returns:
        Dict containing recommended num_workers and batch_size based on hardware
    """
    try:
        # Get CPU cores (physical cores * 0.5 for conservative performance)
        cpu_cores = multiprocessing.cpu_count()
        recommended_workers = max(1, int(cpu_cores * 0.25))

        # Get available GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB

            # Calculate batch size based on GPU memory
            # Rough estimation: 1GB can handle batch size of 4 for YOLO-NAS S
            recommended_batch_size = max(1, int((gpu_memory - 3) * 4))  # Reserve 3GB for system
            recommended_batch_size = min(recommended_batch_size, 32)  # Cap at 32 for safety
        else:
            recommended_batch_size = 4  # Conservative default for CPU

        # Get available RAM
        available_ram = psutil.virtual_memory().available / 1024**3  # Convert to GB

        # Adjust workers based on available RAM (each worker needs ~1.5GB)
        max_workers_ram = max(1, int(available_ram / 1.5))
        recommended_workers = min(recommended_workers, max_workers_ram)

        # Ensure minimum resources are always allocated
        recommended_workers = max(1, recommended_workers)
        recommended_batch_size = max(1, recommended_batch_size)
        
        return {
            'num_workers': recommended_workers,
            'batch_size': recommended_batch_size
        }
    except Exception as e:
        logger.error(f"Error assessing hardware capabilities: {e}")
        # Return conservative defaults
        return {
            'num_workers': 1,
            'batch_size': 4
        }

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