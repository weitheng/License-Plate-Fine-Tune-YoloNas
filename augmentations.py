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

    def apply_motion_blur(self, image):
        # Validate and ensure correct format
        image = validate_image(image, "MotionBlur")
        
        # Ensure image is uint8 RGB
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        image = ensure_rgb_format(image)
        
        # Add random angle variation
        actual_angle = self.angle + random.uniform(-15, 15) if self.angle != 0 else 0
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        center = self.kernel_size // 2
        
        intensity = random.uniform(0.8, 1.0)
        kernel[center, :] = intensity / self.kernel_size
        
        if actual_angle != 0:
            kernel = cv2.warpAffine(
                kernel, 
                cv2.getRotationMatrix2D((center, center), actual_angle, 1.0), 
                (self.kernel_size, self.kernel_size)
            )
        
        result = cv2.filter2D(image, -1, kernel)
        return result.astype(np.uint8)

    def __call__(self, sample_dict):
        try:
            if random.random() < self.prob:
                sample_dict['image'] = self.apply_motion_blur(sample_dict['image'])
            return sample_dict
        except Exception as e:
            logger.error(f"Error in MotionBlur: {e}")
            return sample_dict

class DetectionNoise(DetectionTransform):
    """Add random noise to simulate low-light conditions"""
    def __init__(self, mean=0.0, std=0.1, prob=0.3):
        super().__init__()
        self.mean = mean
        self.std = std
        self.prob = prob

    def add_noise(self, image):
        # Validate and ensure correct format
        image = validate_image(image, "Noise")
        
        # Convert to float32 for noise addition
        image = image.astype(np.float32)
        noise = np.random.normal(self.mean, self.std * 255, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    def __call__(self, sample_dict):
        try:
            if random.random() < self.prob:
                sample_dict['image'] = self.add_noise(sample_dict['image'])
            return sample_dict
        except Exception as e:
            logger.error(f"Error in Noise: {e}")
            return sample_dict

class DetectionWeatherEffects(DetectionTransform):
    """Add weather effects like rain and fog"""
    def __init__(self, rain_intensity=0.2, fog_coef=0.1, prob=0.3):
        super().__init__()
        self.rain_intensity = rain_intensity
        self.fog_coef = fog_coef
        self.prob = prob

    def add_rain(self, image):
        # Validate and ensure correct format
        image = validate_image(image, "WeatherEffects-Rain")
        image = ensure_rgb_format(image)
        
        h, w = image.shape[:2]
        rain_drops = np.random.random((h, w)) < self.rain_intensity
        rain_drops = np.stack([rain_drops] * 3, axis=-1)
        
        # Create rain streaks
        rain_layer = np.zeros_like(rain_drops, dtype=np.uint8)
        streak_length = random.randint(10, 20)
        angle = random.uniform(-20, -10)
        
        for i in range(streak_length):
            shifted = np.roll(rain_drops, i)
            rotated = ndimage.rotate(shifted, angle, reshape=False)
            rain_layer = rain_layer | rotated.astype(np.uint8)
        
        # Add rain effect
        rain_effect = image.copy()
        brightness = np.random.uniform(0.8, 1.2)
        rain_effect[rain_layer.astype(bool)] = np.minimum(
            rain_effect[rain_layer.astype(bool)] * brightness, 
            255
        ).astype(np.uint8)
        
        return cv2.GaussianBlur(rain_effect, (3, 3), 0)

    def add_fog(self, image):
        # Validate and ensure correct format
        image = validate_image(image, "WeatherEffects-Fog")
        image = ensure_rgb_format(image)
        
        fog = np.ones_like(image) * 255
        return cv2.addWeighted(image, 1 - self.fog_coef, fog, self.fog_coef, 0)

    def __call__(self, sample_dict):
        try:
            if random.random() < self.prob:
                image = sample_dict['image']
                effect = random.choice(['rain', 'fog'])
                if effect == 'rain':
                    sample_dict['image'] = self.add_rain(image)
                else:
                    sample_dict['image'] = self.add_fog(image)
            logger.debug(f"WeatherEffects output shape: {sample_dict['image'].shape}")
            return sample_dict
        except Exception as e:
            logger.error(f"Error in WeatherEffects: {e}")
            return sample_dict

class DebugTransform(DetectionTransform):
    """Debug transform to log image properties"""
    def __init__(self, name="Debug"):
        super().__init__()
        self.name = name

    def __call__(self, sample_dict):
        try:
            image = sample_dict['image']
            logger.debug(f"{self.name} - Image shape: {image.shape}, dtype: {image.dtype}")
            validate_image(image, self.name)
            return sample_dict
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            return sample_dict

class DetectionHSV(DetectionTransform):
    def __init__(self, hgain=0.015, sgain=0.7, vgain=0.4):
        super().__init__()
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, sample_dict):
        try:
            image = sample_dict['image']
            # Convert to uint8 if needed
            if image.dtype != np.uint8:
                image = (image * 255).clip(0, 255).astype(np.uint8)
            
            image = ensure_rgb_format(image)
            
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Random gains
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1
            
            # Apply gains and ensure uint8
            hsv = np.clip(hsv * r, 0, 255).astype(np.uint8)
            
            # Convert back to RGB
            sample_dict['image'] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            return sample_dict
        except Exception as e:
            logger.error(f"Error in HSV transform: {e}")
            return sample_dict

class TypeDebugTransform(DetectionTransform):
    """Debug transform to log image type and range"""
    def __init__(self, name="TypeDebug"):
        super().__init__()
        self.name = name

    def __call__(self, sample_dict):
        try:
            image = sample_dict['image']
            logger.debug(f"{self.name} - Image dtype: {image.dtype}, "
                        f"range: [{image.min():.3f}, {image.max():.3f}]")
            return sample_dict
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            return sample_dict

# Add this new class
class DetectionUint8Convert(DetectionTransform):
    """Convert image to uint8 format and ensure correct channel format"""
    def __init__(self):
        super().__init__()

    def __call__(self, sample_dict):
        image = sample_dict['image']
        
        # Check if image needs to be transposed
        if len(image.shape) == 3 and image.shape[2] > 4:
            # Image likely has channels in wrong dimension
            image = np.transpose(image, (1, 2, 0))
        
        # Ensure image is in correct format (H, W, C)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        # Convert to RGB if needed
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] > 3:
            image = image[:, :, :3]
            
        # Convert to uint8
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
            
        sample_dict['image'] = image
        return sample_dict

def create_train_transforms(config: Dict[str, Any], input_size: Tuple[int, int]) -> List[DetectionTransform]:
    """Create training transforms based on config."""
    validate_aug_config(config)
    transforms = []
    aug_config = config.get('augmentation', {})
    
    logger.info("Setting up training augmentations:")
    
    # Add debug transform at the start
    transforms.append(DebugTransform("Initial"))
    
    # Add proper format conversion before rescale
    transforms.append(DetectionUint8Convert())
    
    # Add rescale transform
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
        transforms.append(DebugTransform("Pre-HSV"))
        transforms.append(DetectionTargetsFormatTransform(
            input_format='XYXY_LABEL',
            output_format='LABEL_XYXY'
        ))
        # Use the SuperGradients DetectionHSV with required parameters
        from super_gradients.training.transforms.transforms import DetectionHSV as SGDetectionHSV
        transforms.append(SGDetectionHSV(
            prob=aug_config['hsv'].get('p', 0.5),  # Add probability parameter
            hgain=aug_config['hsv'].get('hgain', 0.015),
            sgain=aug_config['hsv'].get('sgain', 0.7),
            vgain=aug_config['hsv'].get('vgain', 0.4)
        ))
        transforms.append(DetectionTargetsFormatTransform(
            input_format='LABEL_XYXY',
            output_format='XYXY_LABEL'
        ))
        transforms.append(DebugTransform("Post-HSV"))
        logger.info("  - HSV augmentation")

    # Add Mosaic augmentation if enabled
    if aug_config.get('mosaic', {}).get('enabled', False):
        transforms.append(DetectionMosaic(
            input_dim=input_size,
            prob=aug_config['mosaic'].get('p', 0.5)
        ))
        logger.info("  - Mosaic augmentation")

    # Add affine transformation with additional checks
    transforms.append(DebugTransform("Pre-Affine"))
    if aug_config.get('affine', {}).get('enabled', False):
        transforms.append(DetectionRandomAffine(
            degrees=aug_config['affine'].get('degrees', 5.0),
            scales=aug_config['affine'].get('scales', 0.1),
            shear=aug_config['affine'].get('shear', 5.0),
            target_size=input_size,
            border_value=114  # Explicitly set border value
        ))
    transforms.append(DebugTransform("Post-Affine"))
    
    # Add custom augmentations with reduced probability
    if aug_config.get('motion_blur', {}).get('enabled', False):
        transforms.append(DetectionMotionBlur(
            kernel_size=aug_config['motion_blur'].get('kernel_size', 7),
            angle=aug_config['motion_blur'].get('angle', 0.0),
            prob=0.2  # Reduced probability
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
            prob=0.2  # Reduced probability
        ))
        logger.info("  - Weather effects augmentation")
    
    # Always include standardization at the end
    transforms.append(DetectionStandardize(max_value=255.0))
    transforms.append(DebugTransform("Final"))
    logger.info("  - Added standardization")
    
    return transforms

def create_val_transforms(input_size: Tuple[int, int]) -> List[DetectionTransform]:
    """Create validation transforms."""
    logger.info("Setting up validation transforms:")
    transforms = [
        DebugTransform("Initial-Val"),
        DetectionPaddedRescale(input_dim=input_size, pad_value=114),
        DebugTransform("Post-Rescale-Val"),
        DetectionStandardize(max_value=255.0),
        DebugTransform("Final-Val")
    ]
    logger.info("  - Added rescale and standardization")
    return transforms

def get_transforms(config: Dict[str, Any], input_size: Tuple[int, int], is_training: bool = True) -> List[DetectionTransform]:
    """Get transforms based on whether it's training or validation."""
    if is_training:
        return create_train_transforms(config, input_size)
    return create_val_transforms(input_size)

def validate_image(image, transform_name="Unknown"):
    """Validate image before applying transforms"""
    if image is None:
        raise ValueError(f"{transform_name}: Image is None")
    if not isinstance(image, np.ndarray):
        raise ValueError(f"{transform_name}: Image must be a numpy array, got {type(image)}")
    
    # Check if dimensions need to be transposed
    if len(image.shape) == 3 and image.shape[2] > 4:
        logger.warning(f"{transform_name}: Image has {image.shape[2]} channels, attempting to transpose")
        image = np.transpose(image, (1, 2, 0))
    
    # Convert float32 images to uint8 if needed
    if image.dtype == np.float32:
        if image.max() <= 1.0:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)
    
    if len(image.shape) not in [2, 3]:
        raise ValueError(f"{transform_name}: Image must be 2D or 3D array, got shape {image.shape}")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"{transform_name}: Image has invalid dimensions: {image.shape}")
    
    # Ensure proper channel format
    if len(image.shape) == 3 and image.shape[2] > 3:
        logger.warning(f"{transform_name}: Truncating to first 3 channels, original shape: {image.shape}")
        image = image[:, :, :3]
    
    return image

def verify_image_file(image_path: str) -> bool:
    """Verify if image file is valid"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to read image: {image_path}")
            return False
        if img.size == 0:
            logger.error(f"Empty image: {image_path}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error reading image {image_path}: {e}")
        return False

def ensure_rgb_format(image):
    """Ensure image is in RGB format with 3 channels"""
    if len(image.shape) == 2:  # Grayscale
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3:
        if image.shape[2] == 1:  # Single channel
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:  # Already RGB
            return image
        elif image.shape[2] == 4:  # RGBA
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image[:, :, :3]  # Take first 3 channels if more than 4 channels