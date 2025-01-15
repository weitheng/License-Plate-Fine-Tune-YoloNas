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
import cv2

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

def check_image(img: np.ndarray, stage: str) -> np.ndarray:
    """Check and fix image format if needed."""
    if img is None:
        logger.error(f"{stage}: Image is None")
        raise ValueError(f"{stage}: Image is None")
    
    # Detailed logging of image properties
    logger.info(f"{stage}: Image properties:")
    logger.info(f"  - Shape: {img.shape}")
    logger.info(f"  - Type: {img.dtype}")
    logger.info(f"  - Min value: {img.min()}")
    logger.info(f"  - Max value: {img.max()}")
    
    # If image is 1D, it might be flattened
    if len(img.shape) == 1:
        logger.warning(f"{stage}: Got 1D image array, attempting to reshape")
        # Try to determine the original dimensions
        total_pixels = img.shape[0]
        if total_pixels % 3 == 0:  # If divisible by 3, might be RGB
            height = int(np.sqrt(total_pixels / 3))
            if height * height * 3 == total_pixels:
                img = img.reshape(height, height, 3)
                logger.info(f"  - Reshaped to: {img.shape}")
    
    # If image is 2D (height x width), it's grayscale
    if len(img.shape) == 2:
        logger.warning(f"{stage}: Got 2D (grayscale) image, converting to RGB")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        logger.info(f"  - Converted to RGB shape: {img.shape}")
    
    # If image is (channels, height, width), transpose to (height, width, channels)
    if len(img.shape) == 3 and img.shape[0] == 3:
        logger.warning(f"{stage}: Got channels-first format, converting to channels-last")
        img = np.transpose(img, (1, 2, 0))
        logger.info(f"  - Transposed to shape: {img.shape}")
    
    # Ensure we have 3 channels
    if len(img.shape) == 3:
        if img.shape[2] != 3:
            logger.warning(f"{stage}: Incorrect number of channels: {img.shape[2]}")
            if img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] > 3:
                img = img[:, :, :3]
            logger.info(f"  - Adjusted channels to: {img.shape}")
    
    # Ensure uint8 format with correct range
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        logger.info(f"  - Converted to uint8, new range: [{img.min()}, {img.max()}]")
    
    # Final validation
    if len(img.shape) != 3 or img.shape[2] != 3:
        logger.error(f"{stage}: Failed to process image")
        logger.error(f"  - Final shape: {img.shape}")
        logger.error(f"  - Expected shape: (height, width, 3)")
        raise ValueError(f"{stage}: Failed to convert image to correct format. Final shape: {img.shape}")
    
    return img

class ImageShapeCorrection(DetectionTransform):
    """Transform to ensure correct image shape and format"""
    def __init__(self):
        super().__init__()
    
    def apply_to_sample(self, sample):
        try:
            image = sample.image
            sample.image = check_image(image, "Shape-Correction")
            return sample
        except Exception as e:
            logger.error(f"Error in ImageShapeCorrection: {str(e)}")
            logger.error(f"Original image shape: {sample.image.shape}")
            raise

def create_train_transforms(config: Dict[str, Any], input_size: Tuple[int, int]) -> List[DetectionTransform]:
    """Create training transforms based on config."""
    validate_aug_config(config)
    transforms = []
    aug_config = config.get('augmentation', {})
    
    logger.info("Setting up training augmentations:")
    
    # Always start with shape correction
    transforms.append(ImageShapeCorrection())
    logger.info("  - Added image shape correction")
    
    # Add HSV augmentation early in the pipeline
    if aug_config.get('hsv', {}).get('enabled', False):
        transforms.append(DetectionHSV(
            prob=aug_config['hsv'].get('p', 0.5),
            hgain=aug_config['hsv'].get('hgain', 0.015),
            sgain=aug_config['hsv'].get('sgain', 0.7),
            vgain=aug_config['hsv'].get('vgain', 0.4)
        ))
        logger.info("  - HSV augmentation")
    
    # Add padded rescale after HSV
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
    
    # Add Mosaic augmentation
    if aug_config.get('mosaic', {}).get('enabled', False):
        transforms.append(DetectionMosaic(
            input_dim=input_size,
            prob=aug_config['mosaic'].get('p', 0.5)
        ))
        logger.info("  - Mosaic augmentation")

    # Add random affine - only if mosaic is disabled or after mosaic
    if aug_config.get('affine', {}).get('enabled', False):
        if not aug_config.get('mosaic', {}).get('enabled', False):
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
        ImageShapeCorrection(),
        DetectionPaddedRescale(input_dim=input_size, pad_value=114),
        DetectionStandardize(max_value=255.0)
    ]
    logger.info("  - Added shape correction, rescale and standardization")
    return transforms

def get_transforms(config: Dict[str, Any], input_size: Tuple[int, int], is_training: bool = True) -> List[DetectionTransform]:
    """Get transforms based on whether it's training or validation."""
    if is_training:
        return create_train_transforms(config, input_size)
    return create_val_transforms(input_size)