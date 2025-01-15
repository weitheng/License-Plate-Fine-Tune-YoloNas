from super_gradients.training.transforms.transforms import (
    DetectionTransform,
    DetectionPaddedRescale,
    DetectionStandardize,
    DetectionHSV,
    DetectionMosaic,
    DetectionRandomAffine,
    DetectionTargetsFormatTransform
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

def create_train_transforms(config: Dict[str, Any], input_size: Tuple[int, int]) -> List[DetectionTransform]:
    """Create training transforms based on config."""
    validate_aug_config(config)
    transforms = []
    aug_config = config.get('augmentation', {})
    
    logger.info("Setting up training augmentations:")
    
    # Add padded rescale as first transform
    transforms.append(DetectionPaddedRescale(
        input_dim=input_size,
        pad_value=114
    ))
    logger.info(f"  - Padded rescale to {input_size}")
    
    # Basic augmentations based on config
    if aug_config.get('horizontal_flip', {}).get('enabled', False):
        p = aug_config['horizontal_flip'].get('p', 0.5)
        transforms.append(DetectionTargetsFormatTransform())  # For horizontal flip
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
    
    # Always include standardization (replaces normalization)
    transforms.append(DetectionStandardize(max_value=255.0))
    logger.info("  - Added standardization")
    
    return transforms

def create_val_transforms(input_size: Tuple[int, int]) -> List[DetectionTransform]:
    """Create validation transforms."""
    logger.info("Setting up validation transforms:")
    transforms = [
        DetectionPaddedRescale(input_dim=input_size, pad_value=114),
        DetectionStandardize(max_value=255.0)
    ]
    logger.info("  - Added rescale and standardization")
    return transforms

def get_transforms(config: Dict[str, Any], input_size: Tuple[int, int], is_training: bool = True) -> List[DetectionTransform]:
    """Get transforms based on whether it's training or validation."""
    if is_training:
        return create_train_transforms(config, input_size)
    return create_val_transforms(input_size)