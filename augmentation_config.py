import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import yaml

DEBUG_MODE = True  # Set to True to enable debug logging
def load_augmentation_config(yaml_path):
    """Load augmentation configuration from YAML file"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('augmentation', {})

def get_training_augmentations(input_size, config):
    """
    Get training augmentations pipeline using Albumentations.
    
    Args:
        input_size (tuple): Target input size (height, width)
        config (dict): Augmentation configuration from YAML
    """
    train_config = config.get('train', {})
    
    transform = A.Compose([
        # Resize with consistent aspect ratio
        A.LongestMaxSize(
            max_size=max(input_size),
            always_apply=True
        ),
        A.PadIfNeeded(
            min_height=input_size[0],
            min_width=input_size[1],
            border_mode=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
            always_apply=True
        ),
        
        # Color augmentations
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=train_config['color']['brightness_limit'],
                contrast_limit=train_config['color']['contrast_limit'],
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=train_config['color']['hue_shift_limit'],
                sat_shift_limit=train_config['color']['sat_shift_limit'],
                val_shift_limit=train_config['color']['val_shift_limit'],
                p=0.3
            ),
        ], p=train_config['color']['p']),
        
        # Geometric augmentations
        A.HorizontalFlip(p=0.5 if train_config['geometric']['horizontal_flip'] else 0),
        A.ShiftScaleRotate(
            shift_limit=0,
            scale_limit=train_config['geometric']['scale_limit'],
            rotate_limit=train_config['geometric']['rotate_limit'],
            p=train_config['geometric']['p']
        ),
        
        # Normalize and convert to tensor
        A.Normalize(
            mean=train_config['normalize']['mean'],
            std=train_config['normalize']['std']
        ),
        ToTensorV2(),
    ], 
    bbox_params=A.BboxParams(
        format=train_config['bbox_params']['format'],
        min_visibility=train_config['bbox_params']['min_visibility'],
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

def get_validation_augmentations(input_size, config):
    """
    Get validation augmentations pipeline using Albumentations.
    
    Args:
        input_size (tuple): Target input size (height, width)
        config (dict): Augmentation configuration from YAML
    """
    val_config = config.get('val', {})
    
    return A.Compose([
        A.Resize(
            height=input_size[0],
            width=input_size[1],
            always_apply=True
        ),
        A.Normalize(
            mean=val_config['normalize']['mean'],
            std=val_config['normalize']['std']
        ),
        ToTensorV2(),
    ], 
    bbox_params=A.BboxParams(
        format=val_config['bbox_params']['format'],
        min_visibility=val_config['bbox_params']['min_visibility'],
        label_fields=['class_labels']
    )) 