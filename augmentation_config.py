import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_training_augmentations(input_size):
    """
    Get training augmentations pipeline using Albumentations.
    
    Args:
        input_size (tuple): Target input size (height, width)
        
    Returns:
        A.Compose: Augmentation pipeline
    """
    transform = A.Compose([
        # First resize to maintain aspect ratio
        A.LongestMaxSize(max_size=max(input_size)),
        # Then pad if necessary to get exact size
        A.PadIfNeeded(
            min_height=input_size[0],
            min_width=input_size[1],
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        # Then apply other augmentations
        A.RandomResizedCrop(
            height=input_size[0],
            width=input_size[1],
            scale=(0.7, 1.0),
            ratio=(0.75, 1.25),
            p=0.7
        ),
        A.HorizontalFlip(p=0.1),  # Low probability to avoid too many flipped plates
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=12,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.4
        ),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.2),  # Allow more darkening
                contrast_limit=(-0.2, 0.2),    # Allow contrast reduction
                p=0.8
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=(-30, 25),     # Allow desaturation for night scenes
                val_shift_limit=(-30, 15),     # More aggressive value changes
                p=0.8
            ),
        ], p=0.7),
        # Add ISO noise simulation for low-light
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 80.0), mean=0, p=0.5),  # Increased variance for low-light noise
            A.MultiplicativeNoise(multiplier=(0.7, 1.3), p=0.5),  # Simulate sensor noise
        ], p=0.5),
        # Enhanced blur for CCTV footage
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=(3, 7), p=0.5),
            A.MedianBlur(blur_limit=5, p=0.3),  # Added for noise reduction simulation
        ], p=0.4),
        # Weather and lighting effects
        A.OneOf([
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=3,           # Increased for multiple light sources
                shadow_dimension=5,
                p=0.5
            ),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.3),  # Added for night conditions
        ], p=0.3),
        # Color adjustments for different lighting conditions
        A.OneOf([
            A.RGBShift(
                r_shift_limit=15,
                g_shift_limit=15,
                b_shift_limit=15,
                p=0.5
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            # Add specific night-time color cast
            A.ToGray(p=0.3),  # Simulate low-light color loss
        ], p=0.5),
        A.ToFloat(max_value=255.0),  # Normalize to [0,1]
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='yolo',
        min_visibility=0.3,
        label_fields=['class_labels']
    ))
    
    # Add debug wrapper
    def debug_transform(**kwargs):
        print("Input to transform:")
        print(f"Image shape: {kwargs['image'].shape}")
        print(f"Bboxes: {kwargs['bboxes']}")
        print(f"Labels: {kwargs['class_labels']}")
        
        result = transform(**kwargs)
        
        print("Output from transform:")
        print(f"Transformed image shape: {result['image'].shape}")
        print(f"Transformed bboxes: {result['bboxes']}")
        print(f"Transformed labels: {result['class_labels']}")
        return result
        
    return debug_transform

def get_validation_augmentations(input_size):
    """
    Get validation augmentations pipeline using Albumentations.
    Only includes necessary preprocessing.
    
    Args:
        input_size (tuple): Target input size (height, width)
        
    Returns:
        A.Compose: Augmentation pipeline
    """
    return A.Compose([
        # First resize to maintain aspect ratio
        A.LongestMaxSize(max_size=max(input_size)),
        # Then pad to get exact size
        A.PadIfNeeded(
            min_height=input_size[0],
            min_width=input_size[1],
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        A.ToFloat(max_value=255.0),  # Same normalization as training
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='yolo',
        min_visibility=0.3,
        label_fields=['class_labels']
    )) 