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
        # Reduce geometric transformations
        A.OneOf([
            A.RandomResizedCrop(
                height=input_size[0],
                width=input_size[1],
                scale=(0.9, 1.0),  # Less aggressive scale
                ratio=(0.95, 1.05),  # Keep aspect ratio closer to original
                p=0.7
            ),
            A.Resize(
                height=input_size[0],
                width=input_size[1],
                p=0.3
            )
        ], p=1.0),
        
        # Reduce geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,    # Reduced from 0.05
            scale_limit=0.05,      # Reduced from 0.1
            rotate_limit=5,        # Reduced from 15
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.3
        ),
        
        # Color adjustments (kept minimal)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1),
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=10,
                val_shift_limit=10,
                p=0.3
            ),
        ], p=0.5),
        
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='yolo',
        min_visibility=0.6,  # Increased from 0.4
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
        min_visibility=0.6,  # Match training visibility threshold
        label_fields=['class_labels']
    )) 