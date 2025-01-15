from super_gradients.training.transforms import (
    DetectionTransform,
    DetectionRescale,
    DetectionHorizontalFlip,
    DetectionNormalize,
    DetectionHSV,
    DetectionMosaic,
    DetectionRandomAffine,
    ComposeTransforms
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

def create_train_transforms(config: Dict[str, Any], input_size: Tuple[int, int]) -> ComposeTransforms:
    """
    Create training transforms based on config.
    
    Args:
        config: Dictionary containing augmentation configuration
        input_size: Tuple of (height, width) for input size
        
    Returns:
        ComposeTransforms object with transforms
    """
    # Validate config first
    validate_aug_config(config)
    
    transforms = []
    
    aug_config = config.get('augmentation', {})
    
    # Log which augmentations are being used
    logger.info("Setting up training augmentations:")
    
    # Add rescale as first transform
    transforms.append(DetectionRescale(
        output_shape=input_size
    ))
    logger.info(f"  - Rescale to {input_size}")
    
    # Basic augmentations based on config
    if aug_config.get('horizontal_flip', {}).get('enabled', False):
        p = aug_config['horizontal_flip'].get('p', 0.5)
        transforms.append(DetectionHorizontalFlip(prob=p))
        logger.info(f"  - Horizontal Flip (p={p})")
    
    # Add HSV augmentation
    if aug_config.get('hsv', {}).get('enabled', False):
        transforms.append(DetectionHSV(
            prob=aug_config['hsv'].get('p', 0.5),
            hgain=aug_config['hsv'].get('hgain', 0.015),
            sgain=aug_config['hsv'].get('sgain', 0.7),
            vgain=aug_config['hsv'].get('vgain', 0.4)
        ))
        logger.info("  - HSV augmentation")

    # Add Mosaic augmentation
    if aug_config.get('mosaic', {}).get('enabled', False):
        transforms.append(DetectionMosaic(
            input_dim=input_size,
            prob=aug_config['mosaic'].get('p', 0.5)
        ))
        logger.info("  - Mosaic augmentation")

    # Add random affine
    if aug_config.get('affine', {}).get('enabled', False):
        transforms.append(DetectionRandomAffine(
            degrees=aug_config['affine'].get('degrees', 10.0),
            scales=aug_config['affine'].get('scales', 0.1),
            shear=aug_config['affine'].get('shear', 10.0),
            target_size=input_size
        ))
        logger.info("  - Random affine")
    
    # Always include normalization
    transforms.append(DetectionNormalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ))
    logger.info("  - Added normalization")
    
    # Create the composition
    transform = ComposeTransforms(transforms)
    
    return transform

def create_val_transforms(input_size: Tuple[int, int]) -> ComposeTransforms:
    """
    Create validation transforms.
    
    Args:
        input_size: Tuple of (height, width) for input size
        
    Returns:
        ComposeTransforms object with validation transforms
    """
    logger.info("Setting up validation transforms:")
    transform = ComposeTransforms([
        DetectionRescale(output_shape=input_size),
        DetectionNormalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    logger.info("  - Added rescale and normalization")
    return transform

def get_transforms(config: Dict[str, Any], input_size: Tuple[int, int], is_training: bool = True) -> ComposeTransforms:
    """
    Get transforms based on whether it's training or validation.
    
    Args:
        config: Dictionary containing augmentation configuration
        input_size: Tuple of (height, width) for input size
        is_training: Boolean indicating if transforms are for training
        
    Returns:
        ComposeTransforms object with appropriate transforms
    """
    if is_training:
        return create_train_transforms(config, input_size)
    return create_val_transforms(input_size)

def visualize_augmentation(transform: ComposeTransforms, image: np.ndarray, 
                         bboxes: List[List[float]], class_labels: List[int],
                         save_path: str) -> None:
    """
    Visualize augmentation results for debugging.
    
    Args:
        transform: SuperGradients ComposeTransforms
        image: Input image
        bboxes: List of bounding boxes in XYXY format
        class_labels: List of class labels
        save_path: Path to save visualization
    """
    try:
        import cv2
        import matplotlib.pyplot as plt
        from super_gradients.training.datasets.detection_datasets.detection_dataset import DetectionSample
        
        # Create DetectionSample
        sample = DetectionSample(
            image=image,
            bboxes_xyxy=np.array(bboxes),
            labels=np.array(class_labels),
            is_crowd=np.zeros(len(bboxes), dtype=bool)
        )
        
        # Apply transform
        transformed = transform.apply_to_sample(sample)
        
        # Draw original
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image with boxes
        ax1.imshow(image)
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                fill=False, color='red'
            )
            ax1.add_patch(rect)
        ax1.set_title('Original')
        
        # Augmented image with boxes
        ax2.imshow(transformed.image)
        for bbox in transformed.bboxes_xyxy:
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle(
                (x1, y1), x2-x1, y2-y1,
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