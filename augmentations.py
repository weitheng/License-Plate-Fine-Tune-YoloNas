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
import torch
import torch.multiprocessing as mp

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

def check_image(img: np.ndarray, stage: str) -> np.ndarray:
    """Check and fix image format if needed."""
    if img is None:
        logger.error(f"{stage}: Image is None")
        raise ValueError(f"{stage}: Image is None")
    
    # # Detailed logging of image properties
    # logger.info(f"{stage}: Image properties:")
    # logger.info(f"  - Shape: {img.shape}")
    # logger.info(f"  - Type: {img.dtype}")
    # logger.info(f"  - Min value: {img.min()}")
    # logger.info(f"  - Max value: {img.max()}")
    # Only log errors, not regular transformations
    if len(img.shape) == 1:
        logger.warning(f"{stage}: Got 1D image array, attempting to reshape")
    
    # If image is 2D (height x width), it's grayscale
    if len(img.shape) == 2:
        logger.warning(f"{stage}: Got 2D (grayscale) image, converting to RGB")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # If image is (channels, height, width), transpose to (height, width, channels)
    if len(img.shape) == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    
    # Ensure we have 3 channels
    if len(img.shape) == 3:
        if img.shape[2] != 3:
            logger.warning(f"{stage}: Incorrect number of channels: {img.shape[2]}")
            if img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] > 3:
                img = img[:, :, :3]
    
    # Ensure uint8 format with correct range
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    # Final validation
    if len(img.shape) != 3 or img.shape[2] != 3:
        logger.error(f"{stage}: Failed to process image")
        logger.error(f"  - Final shape: {img.shape}")
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

class SafeDetectionHSV(DetectionHSV):
    """HSV transform with additional shape checking"""
    def apply_to_sample(self, sample):
        try:
            image = sample.image
            
            # Convert to channels-last if needed
            if len(image.shape) == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            
            # Ensure correct shape
            if len(image.shape) != 3 or image.shape[2] != 3:
                logger.error(f"Invalid image shape before HSV: {image.shape}")
                image = check_image(image, "Pre-HSV")
            
            # Store the corrected image
            sample.image = image
            
            # Apply HSV transform
            sample = super().apply_to_sample(sample)
            
            # Final shape check
            if len(sample.image.shape) != 3 or sample.image.shape[2] != 3:
                logger.error(f"Invalid image shape after HSV: {sample.image.shape}")
                sample.image = check_image(sample.image, "Post-HSV")
            
            return sample
        except Exception as e:
            logger.error(f"Error in HSV transform: {str(e)}")
            logger.error(f"Image shape: {sample.image.shape if hasattr(sample, 'image') else 'No image'}")
            raise

class SafeDetectionPaddedRescale(DetectionPaddedRescale):
    """Safe version of DetectionPaddedRescale that ensures correct image format"""
    def apply_to_sample(self, sample):
        try:
            image = sample.image
            
            # Convert to channels-last if needed
            if len(image.shape) == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            
            # Ensure image is in uint8 format
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Store the corrected image
            sample.image = image
            
            # Apply the rescale transform
            return super().apply_to_sample(sample)
            
        except Exception as e:
            logger.error(f"Error in SafeDetectionPaddedRescale: {str(e)}")
            logger.error(f"Image shape: {sample.image.shape if hasattr(sample, 'image') else 'No image'}")
            logger.error(f"Image dtype: {sample.image.dtype if hasattr(sample, 'image') else 'No image'}")
            raise

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
        try:
            # Convert to HWC if in CHW format
            needs_transpose = len(image.shape) == 3 and image.shape[0] == 3
            if needs_transpose:
                image = np.transpose(image, (1, 2, 0))
            
            # Ensure uint8 format
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
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
            
            result = cv2.filter2D(image, -1, kernel)
            
            # Convert back to original format if needed
            if needs_transpose:
                result = np.transpose(result, (2, 0, 1))
                
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error in motion blur application: {str(e)}")
            logger.error(f"Image shape: {image.shape if image is not None else 'None'}")
            return image

    def apply_to_sample(self, sample):
        try:
            if random.random() < self.prob:
                sample.image = self.apply_motion_blur(sample.image)
            return sample
        except Exception as e:
            logger.warning(f"Error applying motion blur: {e}")
            return sample

class DetectionNoise(DetectionTransform):
    """Add random noise to simulate low-light conditions"""
    def __init__(self, mean=0.0, std=0.1, prob=0.3):
        super().__init__()
        self.mean = mean
        self.std = std
        self.prob = prob

    def add_noise(self, image):
        try:
            # Convert to HWC if in CHW format
            needs_transpose = len(image.shape) == 3 and image.shape[0] == 3
            if needs_transpose:
                image = np.transpose(image, (1, 2, 0))
            
            # Convert to float32 for noise addition
            image_float = image.astype(np.float32)
            
            # Generate noise for each channel
            noise = np.random.normal(self.mean, self.std, image.shape) * 255.0
            noisy_image = np.clip(image_float + noise, 0, 255)
            
            # Convert back to original format if needed
            result = noisy_image.astype(np.uint8)
            if needs_transpose:
                result = np.transpose(result, (2, 0, 1))
                
            return result
            
        except Exception as e:
            logger.error(f"Error in noise application: {str(e)}")
            logger.error(f"Image shape: {image.shape if image is not None else 'None'}")
            return image

    def apply_to_sample(self, sample):
        try:
            if random.random() < self.prob:
                sample.image = self.add_noise(sample.image)
            return sample
        except Exception as e:
            logger.error(f"Error in Noise: {e}")
            return sample

class DetectionWeatherEffects(DetectionTransform):
    """Add weather effects like rain and fog"""
    def __init__(self, rain_intensity=0.2, fog_coef=0.1, prob=0.3):
        super().__init__()
        self.rain_intensity = rain_intensity
        self.fog_coef = fog_coef
        self.prob = prob

    def add_rain(self, image):
        try:
            # Ensure image is in correct format (HWC)
            if len(image.shape) == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))  # Convert from CHW to HWC
            
            # Ensure uint8 format
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            h, w = image.shape[:2]
            
            # Ensure minimum dimensions for chunking
            min_chunk_size = 32
            num_chunks = max(1, min(4, h // min_chunk_size))
            chunk_size = max(min_chunk_size, h // num_chunks)
            
            # Use smaller data type for rain drops to reduce memory
            rain_drops = (np.random.random((h, w)) < self.rain_intensity).astype(np.uint8)
            streak_length = min(15, random.randint(10, 20))
            angle = random.uniform(-20, -10)
            
            # Use uint8 instead of bool for rain_layer
            rain_layer = np.zeros((h, w), dtype=np.uint8)
            
            # Process in chunks with size validation
            for chunk_start in range(0, h, chunk_size):
                chunk_end = min(chunk_start + chunk_size, h)
                if chunk_end <= chunk_start:
                    continue
                    
                chunk = rain_drops[chunk_start:chunk_end, :]
                for i in range(streak_length):
                    shifted = np.roll(chunk, i)
                    rotated = ndimage.rotate(shifted, angle, reshape=False)
                    rain_layer[chunk_start:chunk_end] |= (rotated > 0.5).astype(np.uint8)
            
            # Apply rain effect more efficiently
            brightness = np.random.uniform(0.8, 1.2)
            rain_effect = image.copy()
            rain_mask = rain_layer > 0
            if rain_mask.any():
                # Expand rain_mask to match image channels
                rain_mask = np.stack([rain_mask] * 3, axis=-1)
                rain_effect[rain_mask] = np.minimum(
                    rain_effect[rain_mask] * brightness, 
                    255
                ).astype(np.uint8)
            
            result = cv2.GaussianBlur(rain_effect, (3, 3), 0)
            
            # Convert back to original format if needed (CHW)
            if len(image.shape) == 3 and image.shape[0] == 3:
                result = np.transpose(result, (2, 0, 1))  # Convert back to CHW
            
            return result
            
        except Exception as e:
            logger.error(f"Error in rain effect application: {str(e)}")
            logger.error(f"Image shape: {image.shape if image is not None else 'None'}")
            return image

    def add_fog(self, image):
        try:
            # Convert to HWC if in CHW format
            if len(image.shape) == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            
            # Ensure uint8 format
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            fog = np.ones_like(image) * 255
            result = cv2.addWeighted(image, 1 - self.fog_coef, fog, self.fog_coef, 0)
            
            # Convert back to original format if needed
            if len(image.shape) == 3 and image.shape[0] == 3:
                result = np.transpose(result, (2, 0, 1))
            
            return result
            
        except Exception as e:
            logger.error(f"Error in fog effect application: {str(e)}")
            logger.error(f"Image shape: {image.shape if image is not None else 'None'}")
            return image

    def apply_to_sample(self, sample):
        try:
            if random.random() < self.prob:
                effect = random.choice(['rain', 'fog'])
                if effect == 'rain':
                    sample.image = self.add_rain(sample.image)
                else:
                    sample.image = self.add_fog(sample.image)
            return sample
        except Exception as e:
            logger.error(f"Error in WeatherEffects: {e}")
            return sample

class SafeDetectionRandomAffine(DetectionRandomAffine):
    """Safe version of DetectionRandomAffine that ensures correct image format"""
    def apply_to_sample(self, sample):
        try:
            image = sample.image
            
            # Convert to HWC if in CHW format
            needs_transpose = len(image.shape) == 3 and image.shape[0] == 3
            if needs_transpose:
                image = np.transpose(image, (1, 2, 0))
            
            # Ensure uint8 format
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # Store the corrected image
            sample.image = image
            
            # Apply the affine transform
            sample = super().apply_to_sample(sample)
            
            # Convert back to original format if needed
            if needs_transpose:
                sample.image = np.transpose(sample.image, (2, 0, 1))
            
            return sample
            
        except Exception as e:
            logger.error(f"Error in affine transform: {str(e)}")
            logger.error(f"Image shape: {image.shape if hasattr(sample, 'image') else 'No image'}")
            return sample

class SafeValidationStandardize(DetectionStandardize):
    """Safe version of DetectionStandardize for validation"""
    def __init__(self, max_value=255.0):
        super().__init__(max_value=max_value)
    
    def apply_to_sample(self, sample):
        try:
            if sample.image is None:
                return sample
                
            # Ensure float32 for standardization
            sample.image = sample.image.astype(np.float32)
            sample = super().apply_to_sample(sample)
            
            # Ensure no NaN or inf values
            if np.any(np.isnan(sample.image)) or np.any(np.isinf(sample.image)):
                logger.error("Invalid values in standardized image")
                sample.image = np.clip(sample.image, 0, 1)
                
            return sample
        except Exception as e:
            logger.error(f"Error in standardization: {e}")
            return sample

def create_train_transforms(config: Dict[str, Any], input_size: Tuple[int, int], dataloader=None) -> List[DetectionTransform]:
    """Create training transforms based on config."""
    validate_aug_config(config)
    transforms = []
    aug_config = config.get('augmentation', {})
    
    logger.info("Setting up training augmentations:")
    
    # Memory management for CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Base transforms with target validation
    transforms.extend([
        ImageShapeCorrection(),
        SafeDetectionPaddedRescale(
            input_dim=input_size,
            pad_value=114,
            max_targets=100
        )
    ])
    logger.info(f"  - Added shape correction and rescale to {input_size}")
    
    # Add HSV augmentation
    if aug_config.get('hsv', {}).get('enabled', False):
        transforms.append(SafeDetectionHSV(
            prob=aug_config['hsv'].get('p', 0.5),
            hgain=aug_config['hsv'].get('hgain', 0.015),
            sgain=aug_config['hsv'].get('sgain', 0.3),
            vgain=aug_config['hsv'].get('vgain', 0.2)
        ))
        logger.info("  - Safe HSV augmentation")
    
    # Basic augmentations
    if aug_config.get('horizontal_flip', {}).get('enabled', False):
        p = aug_config['horizontal_flip'].get('p', 0.5)
        transforms.append(DetectionTargetsFormatTransform())
        logger.info(f"  - Horizontal Flip (p={p})")
    
    # Mosaic with safety checks
    if aug_config.get('mosaic', {}).get('enabled', False):
        transforms.append(SafeDetectionMosaic(
            input_dim=input_size,
            prob=aug_config['mosaic'].get('p', 0.5),
            dataloader=dataloader,  # Pass the dataloader
            enable_memory_cache=True  # Enable caching
        ))
        # Ensure consistent size after mosaic
        transforms.append(SafeDetectionPaddedRescale(
            input_dim=input_size,
            pad_value=114,
            max_targets=100
        ))
        logger.info("  - Mosaic augmentation with rescale")
    
    # Safe affine transform
    if aug_config.get('affine', {}).get('enabled', False):
        if not aug_config.get('mosaic', {}).get('enabled', False):
            transforms.append(SafeDetectionRandomAffine(
                degrees=aug_config['affine'].get('degrees', 5.0),
                scales=aug_config['affine'].get('scales', 0.1),
                shear=aug_config['affine'].get('shear', 5.0),
                target_size=input_size
            ))
            logger.info("  - Safe random affine")
    
    # Add CCTV-specific augmentations
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
    
    # Final validation and standardization
    transforms.extend([
        SafeDetectionPaddedRescale(
            input_dim=input_size,
            pad_value=114,
            max_targets=100
        ),
        SafeValidationStandardize(max_value=255.0)
    ])
    logger.info("  - Added final rescale and standardization")
    
    return transforms

# Move validation transform classes to module level (outside any function)
class SafeValidationRescale(DetectionPaddedRescale):
    """Safe version of DetectionPaddedRescale for validation"""
    def apply_to_sample(self, sample):
        try:
            # Ensure image is in correct format
            if sample.image is None:
                raise ValueError("Empty image in sample")
                
            # Convert to numpy if tensor
            if torch.is_tensor(sample.image):
                sample.image = sample.image.cpu().numpy()
            
            # Ensure HWC format for validation
            if len(sample.image.shape) == 3 and sample.image.shape[0] == 3:
                sample.image = np.transpose(sample.image, (1, 2, 0))
            
            # Ensure uint8 format
            if sample.image.dtype != np.uint8:
                if sample.image.max() <= 1.0:
                    sample.image = (sample.image * 255).astype(np.uint8)
                else:
                    sample.image = sample.image.astype(np.uint8)
            
            # Validate targets if present
            if hasattr(sample, 'target'):
                if sample.target is not None:
                    # Ensure targets are within image bounds
                    h, w = sample.image.shape[:2]
                    sample.target = sample.target[
                        (sample.target[:, 1] < w) & 
                        (sample.target[:, 2] < h) & 
                        (sample.target[:, 1] >= 0) & 
                        (sample.target[:, 2] >= 0)
                    ]
            
            # Apply parent class transform
            return super().apply_to_sample(sample)
            
        except Exception as e:
            logger.error(f"Error in validation transform: {str(e)}")
            logger.error(f"Image shape: {sample.image.shape if hasattr(sample, 'image') else 'No image'}")
            return sample

class SafeValidationStandardizeTransform(DetectionStandardize):
    """Safe version of DetectionStandardize for validation"""
    def apply_to_sample(self, sample):
        try:
            if sample.image is None:
                return sample
                
            # Ensure float32 for standardization
            sample.image = sample.image.astype(np.float32)
            sample = super().apply_to_sample(sample)
            
            # Ensure no NaN or inf values
            if np.any(np.isnan(sample.image)) or np.any(np.isinf(sample.image)):
                logger.error("Invalid values in standardized image")
                sample.image = np.clip(sample.image, 0, 1)
                
            return sample
        except Exception as e:
            logger.error(f"Error in standardization: {e}")
            return sample

def create_val_transforms(input_size: Tuple[int, int]) -> List[DetectionTransform]:
    """Create validation transforms with additional safety checks."""
    logger.info("Setting up validation transforms:")
    
    # Clear CUDA cache before validation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    transforms = [
        ImageShapeCorrection(),
        SafeValidationRescale(
            input_dim=input_size,
            pad_value=114,
            max_targets=100  # Limit maximum targets
        ),
        SafeValidationStandardizeTransform(max_value=255.0)
    ]
    
    logger.info("  - Added safe validation transforms with additional checks")
    return transforms

def get_transforms(config: Dict[str, Any], input_size: Tuple[int, int], is_training: bool = True, dataloader=None) -> List[DetectionTransform]:
    """Get transforms based on whether it's training or validation."""
    if is_training:
        return create_train_transforms(config, input_size, dataloader=dataloader)
    return create_val_transforms(input_size)

class SafeValidationBatch:
    """Wrapper to safely process validation batches"""
    @staticmethod
    def process_batch(batch):
        try:
            if torch.cuda.is_available():
                # Ensure tensors are contiguous in memory
                if isinstance(batch, (tuple, list)):
                    batch = [b.contiguous() if torch.is_tensor(b) else b for b in batch]
                elif torch.is_tensor(batch):
                    batch = batch.contiguous()
                
                # Clear cache before processing
                torch.cuda.empty_cache()
            return batch
        except Exception as e:
            logger.error(f"Error processing validation batch: {e}")
            return None

class SafeDetectionTransform(DetectionTransform):
    """Base class for safe detection transforms with target validation"""
    def validate_targets(self, sample, image_shape):
        """Validate and clean targets to ensure they're within image bounds"""
        try:
            if not hasattr(sample, 'target') or sample.target is None:
                return sample
                
            h, w = image_shape[:2]
            valid_mask = (
                (sample.target[:, 1] >= 0) &  # x1
                (sample.target[:, 2] >= 0) &  # y1
                (sample.target[:, 3] <= w) &  # x2
                (sample.target[:, 4] <= h) &  # y2
                (sample.target[:, 3] > sample.target[:, 1]) &  # width > 0
                (sample.target[:, 4] > sample.target[:, 2])    # height > 0
            )
            
            if not valid_mask.all():
                logger.warning(f"Filtered {(~valid_mask).sum()} invalid targets")
                sample.target = sample.target[valid_mask]
                
            return sample
        except Exception as e:
            logger.error(f"Error validating targets: {e}")
            return sample

    def validate_image(self, image):
        """Validate and normalize image format"""
        try:
            if image is None:
                logger.error("Empty image received")
                return None
            
            # Convert to numpy if tensor
            if torch.is_tensor(image):
                image = image.cpu().numpy()
            
            # Ensure HWC format
            if len(image.shape) == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            elif len(image.shape) == 2:
                # Handle grayscale images
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Ensure uint8 format
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Validate final shape
            if len(image.shape) != 3 or image.shape[2] != 3:
                logger.error(f"Invalid image shape after processing: {image.shape}")
                return None
            
            return image
        except Exception as e:
            logger.error(f"Error validating image: {str(e)}")
            return None

class SafeDetectionMosaic(SafeDetectionTransform):
    """Safe mosaic augmentation with target validation and memory management"""
    def __init__(self, input_dim, prob=0.5, dataloader=None, enable_memory_cache=True):
        super().__init__()
        self.input_dim = input_dim
        self.prob = prob
        self.dataloader = dataloader
        self.enable_memory_cache = enable_memory_cache
        self.cache = {} if enable_memory_cache else None
        
        # Calculate cache size based on dataset and GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            max_cache_by_gpu = min(int(gpu_mem * 100), 1000)  # Scale with GPU memory
        else:
            max_cache_by_gpu = 100
            
        self.max_cache_size = min(
            max_cache_by_gpu,
            len(dataloader.dataset) if dataloader else 0
        )
        
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Remove CUDA stream initialization from __init__
        self.pinned_memory = torch.cuda.is_available()
        
    def _setup_cuda_stream(self):
        """Create CUDA stream on demand in worker process"""
        if not hasattr(self, 'stream') and torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
            torch.cuda.empty_cache()
    
    def _get_random_image(self):
        try:
            if self.dataloader is None or not hasattr(self.dataloader, 'dataset'):
                logger.error("No dataloader provided for mosaic augmentation")
                return None
                
            dataset = self.dataloader.dataset
            if len(dataset) == 0:
                logger.error("Empty dataset in dataloader")
                return None
            
            # Add debug logging
            logger.debug(f"Dataset type: {type(dataset)}")
            logger.debug(f"Dataset length: {len(dataset)}")
            
            # Create lock if needed
            if not hasattr(self, '_lock'):
                self._lock = mp.Lock()
            
            with self._lock:
                # Check if we're in a worker process
                is_worker = mp.current_process().name != 'MainProcess'
                logger.debug(f"Process type: {'worker' if is_worker else 'main'}")
                
                if torch.cuda.is_available() and not is_worker:
                    # Setup CUDA stream when needed (only in main process)
                    self._setup_cuda_stream()
                    
                    # Try to get from cache first if enabled
                    if self.enable_memory_cache and self.cache:
                        idx = random.choice(list(self.cache.keys()))
                        self.cache_hits += 1
                        if self.cache_hits % 1000 == 0:
                            hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)
                            logger.info(f"Mosaic cache hit rate: {hit_rate:.2%}")
                        return self.cache[idx]
                    
                    # Get random sample from dataset
                    idx = random.randint(0, len(dataset) - 1)
                    sample = dataset[idx]
                    self.cache_misses += 1
                    
                    # Process image and cache if possible
                    img = self.validate_image(sample.image)
                    if img is not None and self.enable_memory_cache:
                        if len(self.cache) < self.max_cache_size:
                            self.cache[idx] = img
                        elif random.random() < 0.1:  # Occasionally replace items
                            remove_key = random.choice(list(self.cache.keys()))
                            del self.cache[remove_key]
                            self.cache[idx] = img
                    
                    return img
                else:
                    # Use CPU path for worker processes
                    return self._get_random_image_cpu()
                    
        except Exception as e:
            logger.error(f"Error getting random image: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
            
    def _get_random_image_cpu(self):
        """Fallback method for getting random images using CPU"""
        try:
            dataset = self.dataloader.dataset
            idx = random.randint(0, len(dataset) - 1)
            sample = dataset[idx]
            
            # Handle different dataset return types
            if isinstance(sample, tuple):
                # If sample is a tuple, assume first element is image
                image = sample[0]
            elif hasattr(sample, 'image'):
                # If sample is an object with image attribute
                image = sample.image
            elif isinstance(sample, dict):
                # If sample is a dictionary
                image = sample.get('image')
            else:
                # If none of the above, assume sample is the image
                image = sample
            
            if image is None:
                logger.error("Failed to extract image from sample")
                return None
            
            return self.validate_image(image)
        except Exception as e:
            logger.error(f"Error getting random image on CPU: {str(e)}")
            return None
    
    def clear_cache(self):
        """Clear cache with proper CUDA synchronization"""
        if self.cache:
            self.cache.clear()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def validate_and_clip_targets(self, targets, img_shape):
        """Validate and clip targets to image boundaries"""
        if targets is None or len(targets) == 0:
            return targets
        
        h, w = img_shape[:2]
        
        # Create a copy to avoid modifying original
        targets = targets.copy()
        
        # Clip coordinates to image boundaries
        targets[:, 1] = np.clip(targets[:, 1], 0, w)  # x1
        targets[:, 2] = np.clip(targets[:, 2], 0, h)  # y1
        targets[:, 3] = np.clip(targets[:, 3], 0, w)  # x2
        targets[:, 4] = np.clip(targets[:, 4], 0, h)  # y2
        
        # Filter invalid boxes
        valid_mask = (
            (targets[:, 3] > targets[:, 1]) &  # width > 0
            (targets[:, 4] > targets[:, 2]) &  # height > 0
            (targets[:, 1] < w) &  # x1 < width
            (targets[:, 2] < h) &  # y1 < height
            (targets[:, 3] > 0) &  # x2 > 0
            (targets[:, 4] > 0)    # y2 > 0
        )
        
        return targets[valid_mask]

    def apply_to_sample(self, sample):
        try:
            if random.random() >= self.prob:
                return sample

            # Pre-allocate output arrays
            output_h, output_w = self.input_dim
            mosaic_img = np.zeros((output_h, output_w, 3), dtype=np.uint8)
            all_targets = []

            # Generate splits with safety bounds
            center_x = int(np.clip(random.uniform(
                output_w * 0.25, output_w * 0.75), 
                output_w * 0.2, 
                output_w * 0.8
            ))
            center_y = int(np.clip(random.uniform(
                output_h * 0.25, output_h * 0.75),
                output_h * 0.2,
                output_h * 0.8
            ))

            # Process each quadrant
            for idx, (x1, y1, x2, y2) in enumerate([
                (0, 0, center_x, center_y),                     # top-left
                (center_x, 0, output_w, center_y),             # top-right
                (0, center_y, center_x, output_h),             # bottom-left
                (center_x, center_y, output_w, output_h)       # bottom-right
            ]):
                try:
                    # Get and validate image for this quadrant
                    img = self.validate_image(sample.image if idx == 0 else 
                                           self._get_random_image())
                    if img is None:
                        continue

                    # Calculate scaling factors
                    scale_x = (x2 - x1) / img.shape[1]
                    scale_y = (y2 - y1) / img.shape[0]

                    # Calculate scaling with bounds checking
                    src_h, src_w = img.shape[:2]
                    if src_w <= 0 or src_h <= 0:
                        continue

                    # Place image in mosaic
                    mosaic_img[y1:y2, x1:x2] = cv2.resize(img, (x2-x1, y2-y1))

                    # Transform targets
                    if hasattr(sample, 'target') and sample.target is not None:
                        targets = sample.target.copy()
                        if len(targets):
                            # Scale bounding box coordinates
                            targets[:, [1, 3]] = targets[:, [1, 3]] * scale_x + x1
                            targets[:, [2, 4]] = targets[:, [2, 4]] * scale_y + y1
                            
                            # Validate and clip targets
                            valid_targets = self.validate_and_clip_targets(
                                targets, 
                                (output_h, output_w)
                            )
                            
                            if len(valid_targets) > 0:
                                all_targets.append(valid_targets)

                except Exception as e:
                    logger.error(f"Error processing mosaic quadrant {idx}: {e}")
                    continue

            # Combine and validate targets
            if all_targets:
                sample.target = np.concatenate(all_targets, axis=0)
                sample = self.validate_targets(sample, mosaic_img.shape)

            sample.image = mosaic_img
            return sample

        except Exception as e:
            logger.error(f"Error in mosaic augmentation: {e}")
            return sample
