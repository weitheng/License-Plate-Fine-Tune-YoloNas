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
        # Resize with padding
        A.LongestMaxSize(max_size=max(input_size)),
        A.PadIfNeeded(
            min_height=input_size[0],
            min_width=input_size[1],
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        
        # Basic augmentations that preserve boxes
        A.HorizontalFlip(p=0.5),
        
        # Color adjustments (no geometric transforms)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(-0.2, 0.2),
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.3
            ),
        ], p=0.5),
        
        # Normalize and convert to tensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='yolo',
        min_visibility=0.5,
        label_fields=['class_labels']
    ))
    
    # Add debug wrapper if needed
    if DEBUG_MODE:
        def debug_transform(**kwargs):
            # Handle case with no boxes
            if 'bboxes' not in kwargs:
                kwargs['bboxes'] = []
            if 'class_labels' not in kwargs:
                kwargs['class_labels'] = []
                
            result = transform(**kwargs)
            
            # Only log box changes if there were boxes to begin with
            if len(kwargs['bboxes']) > 0 and len(kwargs['bboxes']) != len(result['bboxes']):
                print(f"Warning: Boxes changed from {len(kwargs['bboxes'])} to {len(result['bboxes'])}")
            return result
        return debug_transform
    
    return transform

def get_validation_augmentations(input_size):
    """Get validation augmentations pipeline using Albumentations."""
    return A.Compose([
        A.LongestMaxSize(max_size=max(input_size)),
        A.PadIfNeeded(
            min_height=input_size[0],
            min_width=input_size[1],
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='yolo',
        min_visibility=0.5,
        label_fields=['class_labels']
    )) 