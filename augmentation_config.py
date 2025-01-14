import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

DEBUG_MODE = True  # Set to True to enable debug logging

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
            scale=(0.8, 1.0),  # Less aggressive scale change
            ratio=(0.9, 1.1),  # Less aggressive aspect ratio change
            p=0.5
        ),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,    # Reduced from 0.1
            scale_limit=0.05,    # Reduced from 0.1
            rotate_limit=5,      # Reduced from 12
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.3
        ),
        # Color adjustments for different lighting conditions
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),  # Reduced from (-0.3, 0.2)
                contrast_limit=(-0.1, 0.1),    # Reduced from (-0.2, 0.2)
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,            # Reduced from 15
                sat_shift_limit=15,            # Reduced from (-30, 25)
                val_shift_limit=10,            # Reduced from (-30, 15)
                p=0.7
            ),
            A.RGBShift(
                r_shift_limit=10,              # Reduced from 15
                g_shift_limit=10,
                b_shift_limit=10,
                p=0.3
            ),
        ], p=0.5),
        # Weather and lighting effects (kept but made more conservative)
        A.OneOf([
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,           # Reduced from 3
                shadow_dimension=5,
                p=0.3
            ),
            A.RandomFog(
                fog_coef_lower=0.1,
                fog_coef_upper=0.2,           # Reduced from 0.3
                p=0.2
            ),
            A.GaussNoise(
                var_limit=(10.0, 50.0),       # Reduced from (10.0, 80.0)
                mean=0,
                p=0.2
            ),
        ], p=0.3),
        # Minimal blur
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),  # Reduced blur
            A.MotionBlur(blur_limit=(3, 5), p=0.3),    # Reduced blur
        ], p=0.3),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='yolo',
        min_visibility=0.4,      # Increased from 0.3
        label_fields=['class_labels']
    ))
    
    # Add debug wrapper if needed
    if DEBUG_MODE:
        def debug_transform(**kwargs):
            result = transform(**kwargs)
            if len(kwargs['bboxes']) != len(result['bboxes']):
                print(f"Warning: Boxes changed from {len(kwargs['bboxes'])} to {len(result['bboxes'])}")
            return result
        return debug_transform
    
    return transform

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