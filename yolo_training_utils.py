import os
import multiprocessing
import psutil
import torch
import yaml
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

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
        logger.error(f"Error assessing hardware: {e}")
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
    try:
        # Validate input path
        if not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)
            
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        if not verify_file_permissions(config_path, 'r'):
            raise PermissionError(f"No read permission for config file: {config_path}")
        
        # Load and parse YAML
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in config file: {e}")
        
        # Validate config structure
        required_keys = ['names', 'nc', 'train', 'val']
        if not all(key in config for key in required_keys):
            raise ValueError(f"Dataset config must contain these keys: {required_keys}")
            
        # Validate config values
        if not isinstance(config['names'], list) or len(config['names']) == 0:
            raise ValueError("Config 'names' must be a non-empty list")
            
        if not isinstance(config['nc'], int) or config['nc'] <= 0:
            raise ValueError("Config 'nc' must be a positive integer")
            
        if len(config['names']) != config['nc']:
            raise ValueError(f"Number of classes ({config['nc']}) doesn't match number of names ({len(config['names'])})")
            
        # Validate paths in config
        for key in ['train', 'val']:
            path = config[key]
            if not isinstance(path, str):
                raise ValueError(f"Config '{key}' must be a string path")
        
        logger.success("✓ Dataset configuration validated successfully")
        return config
        
    except Exception as e:
        logger.error(f"Error loading dataset config: {e}")
        raise

def setup_directories() -> Dict[str, str]:
    """
    Setup all necessary directories with proper permissions and return their paths.
    
    Returns:
        Dictionary containing all directory paths
    """
    try:
        # Validate system requirements first
        validate_system_requirements()
        
        # Get absolute paths
        current_dir = os.path.abspath(os.path.dirname(__file__))
        data_dir = os.path.abspath(os.path.join(current_dir, 'data'))
        coco_dir = os.path.abspath(os.path.join(data_dir, 'coco'))
        combined_dir = os.path.abspath(os.path.join(data_dir, 'combined'))
        checkpoint_dir = os.path.abspath(os.path.join(current_dir, 'checkpoints'))
        export_dir = os.path.abspath(os.path.join(current_dir, 'export'))
        
        # Create main directories with error handling
        main_dirs = [data_dir, coco_dir, combined_dir, checkpoint_dir, export_dir]
        for directory in main_dirs:
            try:
                create_directory_with_check(directory)
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                raise
            
        # Create dataset subdirectories with error handling
        dataset_subdirs = [
            os.path.join(combined_dir, 'images', 'train'),
            os.path.join(combined_dir, 'images', 'val'),
            os.path.join(combined_dir, 'labels', 'train'),
            os.path.join(combined_dir, 'labels', 'val')
        ]
        
        for subdir in dataset_subdirs:
            try:
                create_directory_with_check(subdir)
            except Exception as e:
                logger.error(f"Failed to create dataset subdirectory {subdir}: {e}")
                raise
            
        # Verify YAML configuration file exists
        yaml_path = os.path.join(current_dir, "license_plate_dataset.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Dataset configuration file not found: {yaml_path}")
            
        # Create and verify cache directory
        try:
            cache_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints/')
            os.makedirs(cache_dir, exist_ok=True)
            if not verify_file_permissions(cache_dir, 'w'):
                raise PermissionError(f"No write permission for cache directory: {cache_dir}")
        except Exception as e:
            logger.error(f"Failed to setup cache directory: {e}")
            raise
        
        # Create paths dictionary
        paths = {
            'current_dir': current_dir,
            'data_dir': data_dir,
            'coco_dir': coco_dir,
            'combined_dir': combined_dir,
            'checkpoint_dir': checkpoint_dir,
            'export_dir': export_dir,
            'cache_dir': cache_dir,
            'yaml_path': yaml_path
        }
        
        # Validate all paths
        try:
            validate_paths(paths)
        except Exception as e:
            logger.error("Path validation failed")
            raise
            
        # Verify critical directories have content where expected
        critical_dirs_with_content = {
            'combined_dir/images/train': os.path.join(combined_dir, 'images', 'train'),
            'combined_dir/images/val': os.path.join(combined_dir, 'images', 'val'),
            'combined_dir/labels/train': os.path.join(combined_dir, 'labels', 'train'),
            'combined_dir/labels/val': os.path.join(combined_dir, 'labels', 'val')
        }
        
        # Log directory information
        logger.info("Directory structure:")
        for name, path in paths.items():
            logger.info(f"  - {name}: {path}")
            if name in critical_dirs_with_content:
                if verify_directory_contents(path):
                    logger.success(f"    ✓ Contains files")
                else:
                    logger.warning(f"    ! Empty directory")
        
        # Clean up any temporary files
        logger.info("Cleaning up temporary files...")
        for directory in paths.values():
            if os.path.isdir(directory):
                try:
                    cleanup_temp_files(directory)
                except Exception as e:
                    logger.warning(f"Cleanup failed for {directory}: {e}")
        
        logger.success("✓ All directories created, verified, and cleaned")
        return paths
        
    except Exception as e:
        logger.error(f"Failed to setup directories: {e}")
        raise

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

def verify_file_permissions(file_path: str, mode: str = 'r') -> bool:
    """
    Verify file permissions.
    
    Args:
        file_path: Path to file
        mode: Permission mode to check ('r' for read, 'w' for write)
    Returns:
        bool: True if permissions are valid
    """
    try:
        if mode == 'r':
            return os.access(file_path, os.R_OK)
        elif mode == 'w':
            return os.access(file_path, os.W_OK)
        return False
    except Exception as e:
        logger.error(f"Error checking permissions for {file_path}: {e}")
        return False

def verify_directory_contents(directory: str) -> bool:
    """
    Verify directory exists and contains files.
    
    Args:
        directory: Path to directory
    Returns:
        bool: True if directory exists and contains files
    """
    try:
        if not os.path.exists(directory):
            return False
        return len(os.listdir(directory)) > 0
    except Exception as e:
        logger.error(f"Error checking directory {directory}: {e}")
        return False

def validate_paths(paths: Dict[str, str]) -> None:
    """
    Validate all paths are absolute and have correct permissions.
    
    Args:
        paths: Dictionary of path names and their values
    Raises:
        ValueError: If any path is invalid
    """
    try:
        for name, path in paths.items():
            # Check if path is absolute
            if not os.path.isabs(path):
                raise ValueError(f"{name} must be an absolute path: {path}")
            
            # Check if path exists
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} does not exist: {path}")
            
            # Check permissions based on type
            if os.path.isdir(path):
                if not verify_file_permissions(path, 'w'):
                    raise PermissionError(f"No write permission for directory {name}: {path}")
            elif os.path.isfile(path):
                if not verify_file_permissions(path, 'r'):
                    raise PermissionError(f"No read permission for file {name}: {path}")
                    
    except Exception as e:
        logger.error(f"Path validation failed: {e}")
        raise

def create_directory_with_check(directory: str, required_space_mb: int = 100) -> None:
    """
    Create directory and verify it exists and is writable.
    
    Args:
        directory: Path to directory
        required_space_mb: Required free space in MB
    """
    try:
        # Validate input
        if not directory:
            raise ValueError("Directory path cannot be empty")
            
        if not os.path.isabs(directory):
            directory = os.path.abspath(directory)
        
        # Check available disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(os.path.dirname(directory))
            free_mb = free // (1024 * 1024)  # Convert to MB
            if free_mb < required_space_mb:
                raise OSError(f"Insufficient disk space. Required: {required_space_mb}MB, Available: {free_mb}MB")
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
        
        # Create directory
        os.makedirs(directory, exist_ok=True)
        
        # Verify creation
        if not os.path.exists(directory):
            raise RuntimeError(f"Failed to create directory: {directory}")
            
        # Verify permissions
        if not os.access(directory, os.W_OK):
            raise PermissionError(f"No write permission for directory: {directory}")
            
        # Try to create a test file
        test_file = os.path.join(directory, '.test_write')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            raise PermissionError(f"Cannot write to directory {directory}: {e}")
            
        logger.info(f"Created/verified directory: {directory}")
        
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {str(e)}")
        raise

def validate_system_requirements() -> None:
    """Validate system requirements for training"""
    try:
        # Verify dependencies first
        verify_dependencies()
        
        # Check Python version
        import sys
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8 or higher is required")
        
        # Check available RAM
        import psutil
        available_ram = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
        if available_ram < 4:
            raise RuntimeError(f"Insufficient RAM. At least 4GB required, found {available_ram:.1f}GB")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            # Check CUDA version
            cuda_version = torch.version.cuda
            if cuda_version is None:
                raise RuntimeError("CUDA version could not be determined")
                
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            if gpu_memory < 4:
                raise RuntimeError(f"Insufficient GPU memory. At least 4GB required, found {gpu_memory:.1f}GB")
                
            # Check CUDA compute capability
            compute_capability = torch.cuda.get_device_capability(0)
            if compute_capability[0] < 3:
                raise RuntimeError(f"GPU compute capability too low. Required: 3.0+, Found: {compute_capability[0]}.{compute_capability[1]}")
        else:
            logger.warning("CUDA not available - training will be slow!")
            
        # Check disk space
        import shutil
        current_dir = os.path.dirname(os.path.abspath(__file__))
        total, used, free = shutil.disk_usage(current_dir)
        free_gb = free // (1024**3)
        if free_gb < 50:
            raise RuntimeError(f"Insufficient disk space. At least 50GB required, found {free_gb}GB")
            
        logger.success("✓ System requirements validated")
        
    except Exception as e:
        logger.error(f"System requirements validation failed: {e}")
        raise

def verify_dependencies() -> None:
    """Verify all required dependencies are installed with correct versions"""
    try:
        required_packages = {
            'torch': '2.0.0',
            'super_gradients': '3.1.0',
            'wandb': '0.12.0',
            'pyyaml': '5.1',
            'coloredlogs': '15.0',
            'psutil': '5.8.0'
        }
        
        import pkg_resources
        
        for package, min_version in required_packages.items():
            try:
                installed = pkg_resources.get_distribution(package)
                if pkg_resources.parse_version(installed.version) < pkg_resources.parse_version(min_version):
                    logger.warning(f"{package} version {installed.version} is older than recommended {min_version}")
            except pkg_resources.DistributionNotFound:
                raise RuntimeError(f"Required package {package} is not installed")
                
        logger.success("✓ All dependencies verified")
        
    except Exception as e:
        logger.error(f"Dependency verification failed: {e}")
        raise

def cleanup_temp_files(directory: str) -> None:
    """
    Clean up temporary files in directory.
    
    Args:
        directory: Directory to clean up
    """
    try:
        # Expanded list of temporary file patterns
        temp_patterns = [
            '.test_write',
            '*.tmp',
            '*.temp',
            '.DS_Store',
            '*~',           # Backup files
            '*.bak',        # Backup files
            '*.swp',        # Vim swap files
            '.*.swp',       # Vim swap files
            '*.pyc',        # Python compiled files
            '__pycache__',  # Python cache directories
            '.pytest_cache' # Pytest cache
        ]
        
        import glob
        import shutil
        
        files_removed = 0
        
        for pattern in temp_patterns:
            # Handle both files and directories
            for path in glob.glob(os.path.join(directory, '**', pattern), recursive=True):
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                        files_removed += 1
                        logger.debug(f"Removed temporary file: {path}")
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                        files_removed += 1
                        logger.debug(f"Removed temporary directory: {path}")
                except Exception as e:
                    logger.warning(f"Could not remove {path}: {e}")
        
        if files_removed > 0:
            logger.info(f"Cleaned up {files_removed} temporary files/directories in {directory}")
        
        # Verify cleanup
        remaining_temp_files = []
        for pattern in temp_patterns:
            remaining = glob.glob(os.path.join(directory, '**', pattern), recursive=True)
            remaining_temp_files.extend(remaining)
            
        if remaining_temp_files:
            logger.warning(f"Some temporary files could not be removed: {remaining_temp_files}")
                    
    except Exception as e:
        logger.warning(f"Error during cleanup of {directory}: {e}")