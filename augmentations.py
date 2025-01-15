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
import random
from scipy import ndimage

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
            
        # Validate motion blur parameters
        if aug_name == 'motion_blur' and aug_params.get('enabled', False):
            if aug_params.get('kernel_size', 7) % 2 == 0:
                raise ValueError("Motion blur kernel_size must be odd")
                
        # Validate noise parameters
        if aug_name == 'noise' and aug_params.get('enabled', False):
            if aug_params.get('std', 0.1) < 0:
                raise ValueError("Noise std must be non-negative")
                
        # Validate weather parameters
        if aug_name == 'weather' and aug_params.get('enabled', False):
            if not (0 <= aug_params.get('rain_intensity', 0.2) <= 1):
                raise ValueError("Rain intensity must be between 0 and 1")
            if not (0 <= aug_params.get('fog_coef', 0.1) <= 1):
                raise ValueError("Fog coefficient must be between 0 and 1")

# Add new custom transform classes
class DetectionMotionBlur(DetectionTransform):
    """Apply motion blur to simulate fast-moving vehicles"""
    def __init__(self, kernel_size=7, angle=0.0, prob=0.3):
        super().__init__()
        self.kernel_size = kernel_size
        self.angle = angle
        self.prob = prob
        self._kernel_cache = {}  # Cache for kernels

    def apply_motion_blur(self, image):
        # Add random angle variation for more realistic motion
        actual_angle = self.angle + random.uniform(-15, 15) if self.angle != 0 else 0
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        center = self.kernel_size // 2
        
        # Create motion blur kernel with intensity variation
        intensity = random.uniform(0.8, 1.0)
        kernel[center, :] = intensity / self.kernel_size
        
        if actual_angle != 0:
            kernel = cv2.warpAffine(
                kernel, 
                cv2.getRotationMatrix2D((center, center), actual_angle, 1.0), 
                (self.kernel_size, self.kernel_size)
            )
        
        return cv2.filter2D(image.astype(np.float32), -1, kernel).astype(np.uint8)

    def __call__(self, image, target):
        try:
            if random.random() < self.prob:
                image = self.apply_motion_blur(image)
            return image, target
        except Exception as e:
            logger.warning(f"Error applying motion blur: {e}")
            return image, target

    def __del__(self):
        if hasattr(self, 'kernel'):
            del self.kernel

class DetectionNoise(DetectionTransform):
    """Add random noise to simulate low-light conditions"""
    def __init__(self, mean=0.0, std=0.1, prob=0.3):
        super().__init__()
        self.mean = mean
        self.std = std
        self.prob = prob

    def add_noise(self, image):
        noise = np.random.normal(self.mean, self.std, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = self.add_noise(image)
        return image, target

    def __del__(self):
        if hasattr(self, 'noise'):
            del self.noise

class DetectionWeatherEffects(DetectionTransform):
    """Add weather effects like rain and fog"""
    def __init__(self, rain_intensity=0.2, fog_coef=0.1, prob=0.3):
        super().__init__()
        self.rain_intensity = rain_intensity
        self.fog_coef = fog_coef
        self.prob = prob

    def add_rain(self, image):
        h, w = image.shape[:2]
        # Create rain streaks
        rain_drops = np.random.random((h, w)) < self.rain_intensity
        streak_length = random.randint(10, 20)
        angle = random.uniform(-20, -10)  # Typical rain angle
        
        # Create rain streak effect
        rain_layer = np.zeros_like(rain_drops)
        for i in range(streak_length):
            shifted = np.roll(rain_drops, i)
            rain_layer = rain_layer | ndimage.rotate(shifted, angle, reshape=False)
        
        # Add brightness variation
        brightness = np.random.uniform(0.8, 1.2)
        rain_effect = image.copy()
        rain_effect[rain_layer] = np.minimum(
            rain_effect[rain_layer] * brightness, 
            255
        )
        
        # Add slight blur to simulate rain
        return cv2.GaussianBlur(rain_effect, (3, 3), 0)

    def add_fog(self, image):
        fog = np.ones_like(image) * 255
        return cv2.addWeighted(image, 1 - self.fog_coef, fog, self.fog_coef, 0)

    def __call__(self, image, target):
        if random.random() < self.prob:
            effect = random.choice(['rain', 'fog'])
            if effect == 'rain':
                image = self.add_rain(image)
            else:
                image = self.add_fog(image)
        return image, target

    def __del__(self):
        if hasattr(self, 'rain_effect'):
            del self.rain_effect

def create_train_transforms(config: Dict[str, Any], input_size: Tuple[int, int]) -> List[DetectionTransform]:
    """Create training transforms based on config."""
    validate_aug_config(config)
    transforms = []
    aug_config = config.get('augmentation', {})
    
    logger.info("Setting up training augmentations:")
    
    # Add existing transforms
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
            sgain=aug_config['hsv'].get('sgain', 0.3),
            vgain=aug_config['hsv'].get('vgain', 0.2)
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
            degrees=aug_config['affine'].get('degrees', 5.0),
            scales=aug_config['affine'].get('scales', 0.1),
            shear=aug_config['affine'].get('shear', 5.0),
            target_size=input_size
        ))
        logger.info("  - Random affine")
    # Add new CCTV-specific augmentations
    if aug_config.get('motion_blur', {}).get('enabled', False):
        transforms.append(DetectionMotionBlur(
            kernel_size=aug_config['motion_blur'].get('kernel_size', 7),
            angle=aug_config['motion_blur'].get('angle', 0.0),
            prob=aug_config['motion_blur'].get('p', 0.3)
        ))
        logger.info("  - Motion blur augmentation")

    if aug_config.get('noise', {}).get('enabled', False):
        transforms.append(DetectionNoise(
            mean=aug_config['noise'].get('mean', 0.0),
            std=aug_config['noise'].get('std', 0.1),
            prob=aug_config['noise'].get('p', 0.3)
        ))
        logger.info("  - Noise augmentation")

    if aug_config.get('weather', {}).get('enabled', False):
        transforms.append(DetectionWeatherEffects(
            rain_intensity=aug_config['weather'].get('rain_intensity', 0.2),
            fog_coef=aug_config['weather'].get('fog_coef', 0.1),
            prob=aug_config['weather'].get('p', 0.3)
        ))
        logger.info("  - Weather effects augmentation")
    
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

def validate_image(image):
    """Validate image before applying transforms"""
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")
    if image.dtype != np.uint8:
        raise ValueError("Image must be uint8")
    if len(image.shape) not in [2, 3]:
        raise ValueError("Image must be 2D or 3D array")