import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def create_train_transforms(config: Dict[str, Any], input_size: Tuple[int, int]) -> A.Compose:
    """
    Create training transforms based on config.
    
    Args:
        config: Dictionary containing augmentation configuration
        input_size: Tuple of (height, width) for input size
        
    Returns:
        Albumentations Compose object with transforms
    """
    transforms = []
    
    aug_config = config.get('augmentation', {})
    
    # Log which augmentations are being used
    logger.info("Setting up training augmentations:")
    
    # Basic augmentations based on config
    if aug_config.get('horizontal_flip', {}).get('enabled', False):
        p = aug_config['horizontal_flip'].get('p', 0.5)
        transforms.append(A.HorizontalFlip(p=p))
        logger.info(f"  - Horizontal Flip (p={p})")
    
    if aug_config.get('brightness_contrast', {}).get('enabled', False):
        brightness = aug_config['brightness_contrast'].get('brightness_limit', 0.2)
        contrast = aug_config['brightness_contrast'].get('contrast_limit', 0.2)
        p = aug_config['brightness_contrast'].get('p', 0.5)
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=brightness,
            contrast_limit=contrast,
            p=p
        ))
        logger.info(f"  - Brightness/Contrast (brightness_limit={brightness}, contrast_limit={contrast}, p={p})")
    
    if aug_config.get('blur', {}).get('enabled', False):
        blur_limit = aug_config['blur'].get('blur_limit', 3)
        p = aug_config['blur'].get('p', 0.3)
        transforms.append(A.Blur(blur_limit=blur_limit, p=p))
        logger.info(f"  - Blur (blur_limit={blur_limit}, p={p})")
    
    # Always include normalization and tensor conversion
    transforms.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2(p=1.0)
    ])
    
    # Create the composition with bbox handling
    transform = A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='yolo',           # Using YOLO format: [x_center, y_center, width, height]
            label_fields=['class_labels'],
            min_visibility=0.3       # Only keep boxes that are at least 30% visible after augmentation
        )
    )
    
    logger.info("  - Added bbox handling with YOLO format")
    logger.info("  - Added normalization and tensor conversion")
    
    return transform

def create_val_transforms(input_size: Tuple[int, int]) -> A.Compose:
    """
    Create validation transforms.
    
    Args:
        input_size: Tuple of (height, width) for input size
        
    Returns:
        Albumentations Compose object with validation transforms
    """
    return A.Compose(
        [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2(p=1.0)
        ],
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        )
    )

def get_transforms(config: Dict[str, Any], input_size: Tuple[int, int], is_training: bool = True) -> A.Compose:
    """
    Get transforms based on whether it's training or validation.
    
    Args:
        config: Dictionary containing augmentation configuration
        input_size: Tuple of (height, width) for input size
        is_training: Boolean indicating if transforms are for training
        
    Returns:
        Albumentations Compose object with appropriate transforms
    """
    if is_training:
        return create_train_transforms(config, input_size)
    return create_val_transforms(input_size) 