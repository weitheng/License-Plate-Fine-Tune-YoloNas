from super_gradients.training.transforms import (
    DetectionTransform,
    Resize,
    HorizontalFlip,
    Normalize,
    ComposeDetectionTransforms,
    RandomBrightnessContrast,
    RandomBlur
)
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

def validate_aug_config(config: Dict[str, Any]) -> None:
    """Validate augmentation configuration parameters."""
    aug_config = config.get('augmentation', {})
    
    for aug_name, aug_params in aug_config.items():
        if not isinstance(aug_params, dict):
            raise ValueError(f"Invalid config for {aug_name}: must be a dictionary")
        
        if 'enabled' not in aug_params:
            raise ValueError(f"Missing 'enabled' parameter for {aug_name}")
            
        if 'p' in aug_params and not (0 <= aug_params['p'] <= 1):
            raise ValueError(f"Invalid probability for {aug_name}: must be between 0 and 1")

def create_train_transforms(config: Dict[str, Any], input_size: Tuple[int, int]) -> ComposeDetectionTransforms:
    """
    Create training transforms based on config.
    
    Args:
        config: Dictionary containing augmentation configuration
        input_size: Tuple of (height, width) for input size
        
    Returns:
        ComposeDetectionTransforms object with transforms
    """
    # Validate config first
    validate_aug_config(config)
    
    transforms = []
    
    aug_config = config.get('augmentation', {})
    
    # Log which augmentations are being used
    logger.info("Setting up training augmentations:")
    
    # Add resize as first transform
    transforms.append(Resize(input_size))
    logger.info(f"  - Resize to {input_size}")
    
    # Basic augmentations based on config
    if aug_config.get('horizontal_flip', {}).get('enabled', False):
        p = aug_config['horizontal_flip'].get('p', 0.5)
        transforms.append(HorizontalFlip(p=p))
        logger.info(f"  - Horizontal Flip (p={p})")
    
    if aug_config.get('brightness_contrast', {}).get('enabled', False):
        brightness = aug_config['brightness_contrast'].get('brightness_limit', 0.2)
        contrast = aug_config['brightness_contrast'].get('contrast_limit', 0.2)
        p = aug_config['brightness_contrast'].get('p', 0.5)
        transforms.append(RandomBrightnessContrast(
            brightness_limit=brightness,
            contrast_limit=contrast,
            p=p
        ))
        logger.info(f"  - Brightness/Contrast (brightness_limit={brightness}, contrast_limit={contrast}, p={p})")
    
    if aug_config.get('blur', {}).get('enabled', False):
        blur_limit = aug_config['blur'].get('blur_limit', 3)
        p = aug_config['blur'].get('p', 0.3)
        transforms.append(RandomBlur(blur_limit=blur_limit, p=p))
        logger.info(f"  - Blur (blur_limit={blur_limit}, p={p})")
    
    # Always include normalization
    transforms.append(Normalize())
    logger.info("  - Added normalization")
    
    return ComposeDetectionTransforms(transforms)

def create_val_transforms(input_size: Tuple[int, int]) -> ComposeDetectionTransforms:
    """
    Create validation transforms.
    
    Args:
        input_size: Tuple of (height, width) for input size
        
    Returns:
        ComposeDetectionTransforms object with validation transforms
    """
    logger.info("Setting up validation transforms:")
    transforms = [
        Resize(input_size),
        Normalize()
    ]
    logger.info("  - Added resize and normalization")
    return ComposeDetectionTransforms(transforms)

def get_transforms(config: Dict[str, Any], input_size: Tuple[int, int], is_training: bool = True) -> ComposeDetectionTransforms:
    """
    Get transforms based on whether it's training or validation.
    
    Args:
        config: Dictionary containing augmentation configuration
        input_size: Tuple of (height, width) for input size
        is_training: Boolean indicating if transforms are for training
        
    Returns:
        ComposeDetectionTransforms object with appropriate transforms
    """
    if is_training:
        return create_train_transforms(config, input_size)
    return create_val_transforms(input_size)

def visualize_augmentation(transform: ComposeDetectionTransforms, image: np.ndarray, 
                         bboxes: List[List[float]], class_labels: List[int],
                         save_path: str) -> None:
    """
    Visualize augmentation results for debugging.
    
    Args:
        transform: Albumentations transform
        image: Input image
        bboxes: List of bounding boxes in YOLO format
        class_labels: List of class labels
        save_path: Path to save visualization
    """
    try:
        import cv2
        import matplotlib.pyplot as plt
        
        # Apply transform
        transformed = transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        
        # Draw original
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image with boxes
        ax1.imshow(image)
        for bbox in bboxes:
            x, y, w, h = bbox
            rect = plt.Rectangle(
                (x - w/2, y - h/2), w, h,
                fill=False, color='red'
            )
            ax1.add_patch(rect)
        ax1.set_title('Original')
        
        # Augmented image with boxes
        ax2.imshow(transformed['image'])
        for bbox in transformed['bboxes']:
            x, y, w, h = bbox
            rect = plt.Rectangle(
                (x - w/2, y - h/2), w, h,
                fill=False, color='red'
            )
            ax2.add_patch(rect)
        ax2.set_title('Augmented')
        
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Saved augmentation visualization to {save_path}")
    except Exception as e:
        logger.error(f"Failed to visualize augmentation: {e}") 

def setup_visualization_dir(base_dir: str, experiment_name: str) -> str:
    """
    Setup directory for augmentation visualizations under the specific run directory.
    
    Args:
        base_dir: Base directory of the project
        experiment_name: Name of the experiment/run
        
    Returns:
        Path to visualization directory
    """
    # Get current run directory from SuperGradients (most recent RUN_* directory)
    experiment_dir = os.path.join(base_dir, 'checkpoints', experiment_name)
    if os.path.exists(experiment_dir):
        run_dirs = [d for d in os.listdir(experiment_dir) if d.startswith('RUN_')]
        if run_dirs:
            # Sort by creation time (newest first)
            run_dirs.sort(key=lambda x: os.path.getctime(os.path.join(experiment_dir, x)), reverse=True)
            current_run = run_dirs[0]
            vis_dir = os.path.join(experiment_dir, current_run, 'visualizations', 'augmentations')
        else:
            # If no run directory exists yet, create a temporary one
            vis_dir = os.path.join(base_dir, 'visualizations', 'augmentations', 'pre_training')
    else:
        # If experiment directory doesn't exist yet, create a temporary one
        vis_dir = os.path.join(base_dir, 'visualizations', 'augmentations', 'pre_training')
    
    # Clean up old visualizations if they exist
    if os.path.exists(vis_dir):
        try:
            for file in os.listdir(vis_dir):
                if file.endswith('.png'):
                    os.remove(os.path.join(vis_dir, file))
            logger.info("Cleaned up old visualization files")
        except Exception as e:
            logger.warning(f"Failed to clean up old visualizations: {e}")
    
    os.makedirs(vis_dir, exist_ok=True)
    logger.info(f"Created visualization directory: {vis_dir}")
    return vis_dir 