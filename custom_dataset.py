from typing import List, Tuple
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

# Setup logging
logger = logging.getLogger(__name__)

DEBUG_MODE = True  # Set to True to enable debug logging

# Function to verify bbox format - this was used but not defined
def verify_bbox_format(boxes):
    """
    Verify that boxes are in the correct format and have valid values.
    """
    if not isinstance(boxes, torch.Tensor):
        raise ValueError("Boxes must be a torch.Tensor")
    if boxes.dim() != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Boxes must be a Nx4 tensor, got shape {boxes.shape}")
    if not torch.all((boxes >= 0) & (boxes <= 1)):
        raise ValueError("Box coordinates must be in range [0, 1]")

def collate_fn(batch: List[Tuple]) -> Tuple:
    """
    Custom collate function to handle variable-sized tensors and match SuperGradients YOLO format.
    
    Args:
        batch: List of tuples containing (image, target, metadata)
        
    Returns:
        Tuple of (images, targets, metadata)
    """
    if DEBUG_MODE:
        # Only log critical issues
        for batch_idx, (_, target, _) in enumerate(batch):
            boxes = target['boxes']
            if len(boxes) > 0:
                # Validate box format
                if not isinstance(boxes, torch.Tensor):
                    raise ValueError(f"Boxes must be torch.Tensor, got {type(boxes)}")
                if boxes.dim() != 2 or boxes.shape[1] != 4:
                    raise ValueError(f"Boxes must be Nx4 tensor, got shape {boxes.shape}")
                # Check for invalid values
                invalid_boxes = boxes[~((boxes >= 0) & (boxes <= 1)).all(dim=1)]
                if len(invalid_boxes) > 0:
                    print(f"WARNING: Found {len(invalid_boxes)} invalid boxes in batch {batch_idx}")
                    print(f"Invalid boxes: {invalid_boxes}")
    
    images = torch.stack([item[0] for item in batch])
    
    # Initialize empty target tensor
    max_boxes = max(len(item[1]['boxes']) for item in batch)
    if max_boxes == 0:
        targets = torch.zeros((0, 6), dtype=torch.float32)
    else:
        all_targets = []
        for batch_idx, (_, target, _) in enumerate(batch):
            boxes = target['boxes']
            labels = target['labels']
            
            if len(boxes) > 0:
                # Ensure boxes are valid
                if torch.isnan(boxes).any() or torch.isinf(boxes).any():
                    continue
                    
                # Create batch index column
                batch_col = torch.full((len(boxes), 1), batch_idx, dtype=torch.float32)
                
                # Combine batch index, labels, and boxes
                target_boxes = torch.cat([
                    batch_col,
                    labels.float().view(-1, 1),
                    boxes
                ], dim=1)
                
                all_targets.append(target_boxes)
        
        if all_targets:
            targets = torch.cat(all_targets, dim=0)
        else:
            targets = torch.zeros((0, 6), dtype=torch.float32)
    
    # Ensure targets has correct shape and no invalid values
    if len(targets) > 0:
        assert targets.shape[1] == 6, f"Invalid target shape: {targets.shape}"
        assert not torch.isnan(targets).any(), "NaN values in targets"
        assert not torch.isinf(targets).any(), "Inf values in targets"
    
    metadata = {
        'image_paths': [item[2]['image_path'] for item in batch]
    }
    
    return images, targets, metadata

def clip_bbox(bbox):
    """
    Clip bounding box coordinates to be within [0, 1] and validate dimensions.
    Returns None if the bbox becomes invalid after clipping.
    """
    x_center, y_center, width, height = bbox
    
    # If any value is NaN or infinite, reject the box
    if not all(map(lambda x: isinstance(x, (int, float)) and -float('inf') < x < float('inf'), bbox)):
        return None
        
    # Clip centers to [0, 1]
    x_center = np.clip(x_center, 0.0, 1.0)
    y_center = np.clip(y_center, 0.0, 1.0)
    
    # Clip width and height to reasonable values
    width = np.clip(width, 0.0, 1.0)
    height = np.clip(height, 0.0, 1.0)
    
    # Ensure box dimensions are valid
    if width < 0.001 or height < 0.001 or width > 0.999 or height > 0.999:
        return None
        
    # Ensure center coordinates allow box to stay within image
    if (x_center - width/2) < 0 or (x_center + width/2) > 1:
        return None
    if (y_center - height/2) < 0 or (y_center + height/2) > 1:
        return None
    
    return [x_center, y_center, width, height]

def validate_boxes(boxes, labels, image_shape):
    """
    Validate boxes after augmentation with more lenient criteria.
    Returns filtered boxes and labels.
    """
    if len(boxes) == 0:
        return boxes, labels
        
    valid_boxes = []
    valid_labels = []
    
    img_height, img_width = image_shape[:2]
    
    for box, label in zip(boxes, labels):
        # Unpack box coordinates
        x_center, y_center, width, height = box
        
        # Basic sanity checks
        if not all(isinstance(x, (int, float)) for x in box):
            continue
            
        if any(np.isnan(box)) or any(np.isinf(box)):
            continue
            
        # Ensure box dimensions are positive and within reasonable bounds
        if width <= 0 or height <= 0 or width > 1.0 or height > 1.0:
            continue
            
        # Ensure center is within image bounds
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            continue
            
        # Ensure box boundaries are within image
        x_min = x_center - width/2
        y_min = y_center - height/2
        x_max = x_center + width/2
        y_max = y_center + height/2
        
        if x_min < 0 or y_min < 0 or x_max > 1 or y_max > 1:
            continue
            
        # Add valid box and label
        valid_boxes.append([x_center, y_center, width, height])
        valid_labels.append(label)
    
    if len(valid_boxes) == 0:
        return np.array([], dtype=np.float32).reshape(0, 4), np.array([], dtype=np.int64)
    
    return np.array(valid_boxes, dtype=np.float32), np.array(valid_labels, dtype=np.int64)

class AugmentedDetectionDataset(Dataset):
    """
    Custom dataset class with Albumentations augmentations support.
    """
    def __init__(self, data_dir, images_dir, labels_dir, transforms, input_size=(640, 640)):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, images_dir)
        self.labels_dir = os.path.join(data_dir, labels_dir)
        self.transforms = transforms
        self.input_size = input_size
        
        # Get list of image files
        self.image_files = [f for f in os.listdir(self.images_dir) 
                           if f.endswith(('.jpg', '.jpeg', '.png'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Add debug logging
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(
            self.labels_dir,
            img_name.rsplit('.', 1)[0] + '.txt'
        )
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read corresponding label file
        boxes = []
        class_labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    try:
                        class_id, x_center, y_center, width, height = map(float, line.strip().split())
                        # Clip and validate bbox coordinates
                        bbox = clip_bbox([x_center, y_center, width, height])
                        if bbox is not None:  # Only add valid boxes
                            boxes.append(bbox)
                            class_labels.append(class_id)
                    except Exception as e:
                        print(f"Warning: Skipping invalid bbox in {label_path}: {line.strip()} - {str(e)}")
                        continue
        
        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        class_labels = np.array(class_labels, dtype=np.int64)
        
        # Initialize valid_boxes and valid_labels before try block
        valid_boxes = []
        valid_labels = []
        
        # Apply augmentations with additional validation
        try:
            if len(boxes) > 0:
                transformed = self.transforms(
                    image=image,
                    bboxes=boxes,
                    class_labels=class_labels
                )
                
                # Store original count for debugging
                original_box_count = len(boxes)
                
                # Validate transformed boxes
                valid_boxes, valid_labels = validate_boxes(
                    transformed['bboxes'], 
                    transformed['class_labels'],
                    transformed['image'].shape
                )
                
                if len(valid_boxes) > 0:
                    boxes = torch.tensor(valid_boxes, dtype=torch.float32)
                    class_labels = torch.tensor(valid_labels, dtype=torch.long)
                else:
                    if DEBUG_MODE:
                        logger.warning(f"All boxes were filtered out for {img_path}")
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                    class_labels = torch.zeros(0, dtype=torch.long)
                
                # Detailed debug logging
                if DEBUG_MODE:
                    if len(valid_boxes) != original_box_count:
                        logger.info(f"Filtered {original_box_count - len(valid_boxes)} boxes in {img_path}")
                    if len(valid_boxes) == 0:
                        logger.warning(f"No boxes found for {img_path}")
                    elif len(valid_boxes) > 20:
                        logger.info(f"Large number of boxes ({len(valid_boxes)}) in {img_path}")
                    if image.shape[0] > 1000 or image.shape[1] > 1000:
                        logger.info(f"Large image size {image.shape} for {img_path}")
                    if len(transformed['bboxes']) != original_box_count:
                        logger.warning(f"Boxes changed after transform for {img_path} "
                                     f"(Before: {original_box_count}, After: {len(transformed['bboxes'])})")
                
                image = transformed['image']
                
            else:
                # When there are no boxes, still pass empty lists
                transformed = self.transforms(
                    image=image,
                    bboxes=[],
                    class_labels=[]
                )
                image = transformed['image']
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                class_labels = torch.zeros(0, dtype=torch.long)
            
        except Exception as e:
            logger.error(f"Error in transformation for {img_path}: {str(e)}")
            # When there's an error, still pass empty lists
            transformed = self.transforms(
                image=image,
                bboxes=[],
                class_labels=[]
            )
            image = transformed['image']
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            class_labels = torch.zeros(0, dtype=torch.long)
        
        # Additional validation before returning
        if torch.isnan(boxes).any() or torch.isinf(boxes).any():
            logger.warning(f"Invalid box values detected in {img_path}")
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            class_labels = torch.zeros(0, dtype=torch.long)
            
        targets = {
            'boxes': boxes,
            'labels': class_labels
        }
        
        metadata = {
            'image_path': img_path
        }
        
        return image, targets, metadata 