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
                invalid_boxes = boxes[~((boxes >= 0) & (boxes <= 1)).all(dim=1)]
                if len(invalid_boxes) > 0:
                    print(f"WARNING: Invalid box coordinates in batch {batch_idx}")
    
    images = torch.stack([item[0] for item in batch])
    
    # Convert targets to SuperGradients YOLO format
    all_targets = []
    for batch_idx, (_, target, _) in enumerate(batch):
        boxes = target['boxes']
        labels = target['labels'].float()
        
        if len(boxes) > 0:
            verify_bbox_format(boxes)
            batch_col = torch.full((len(boxes), 1), batch_idx, dtype=torch.float32)
            target_boxes = torch.cat([batch_col, labels.view(-1, 1), boxes], dim=1)
            all_targets.append(target_boxes)
    
    if len(all_targets) > 0:
        targets = torch.cat(all_targets, dim=0)
        assert targets.shape[1] == 6, f"Invalid target shape: {targets.shape}"
    else:
        targets = torch.zeros((0, 6), dtype=torch.float32)
    
    metadata = {
        'image_paths': [item[2]['image_path'] for item in batch]
    }
    
    return images, targets, metadata

class AugmentedDetectionDataset(Dataset):
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        
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
        
        # Debug print
        print(f"Loading image: {img_path}")
        print(f"Loading label: {label_path}")
        
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
                
                # Additional validation after transformation
                valid_boxes = []
                valid_labels = []
                
                for box, label in zip(transformed['bboxes'], transformed['class_labels']):
                    # Strict validation of box coordinates
                    if (all(0.0 <= coord <= 1.0 for coord in box) and
                        0.001 < box[2] < 0.999 and  # width
                        0.001 < box[3] < 0.999):    # height
                        valid_boxes.append(box)
                        valid_labels.append(label)
                
                if valid_boxes:
                    boxes = torch.tensor(valid_boxes, dtype=torch.float32)
                    class_labels = torch.tensor(valid_labels, dtype=torch.long)
                else:
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                    class_labels = torch.zeros(0, dtype=torch.long)
                
                image = transformed['image']
                
                # Verify tensor values
                if torch.isnan(boxes).any() or torch.isinf(boxes).any():
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                    class_labels = torch.zeros(0, dtype=torch.long)
                
            else:
                # Handle cases with no boxes
                transformed = self.transforms(
                    image=image,
                    bboxes=np.zeros((0, 4), dtype=np.float32),
                    class_labels=np.array([], dtype=np.int64)
                )
                image = transformed['image']
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                class_labels = torch.zeros(0, dtype=torch.long)
                
        except Exception as e:
            print(f"Warning: Error in transformation for {img_path}: {str(e)}")
            # Fallback to basic transformation
            transformed = self.transforms(
                image=image,
                bboxes=np.zeros((0, 4), dtype=np.float32),
                class_labels=np.array([], dtype=np.int64)
            )
            image = transformed['image']
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            class_labels = torch.zeros(0, dtype=torch.long)
        
        # Return in SuperGradients expected format
        targets = {
            'boxes': boxes,
            'labels': class_labels
        }
        
        metadata = {
            'image_path': img_path
        }
        
        # Add counter for filtered boxes
        if len(boxes) != len(valid_boxes):
            print(f"Filtered {len(boxes) - len(valid_boxes)} invalid boxes in {img_path}")
        
        # After loading boxes and labels
        print(f"Number of boxes: {len(boxes)}")
        print(f"Box coordinates sample: {boxes[:2] if len(boxes) > 0 else 'No boxes'}")
        
        if DEBUG_MODE:
            # Only log problematic cases
            if len(boxes) == 0:
                print(f"Warning: No boxes found for {img_path}")
            elif len(boxes) > 20:
                print(f"Note: Large number of boxes ({len(boxes)}) in {img_path}")
                
            # Log unusual image sizes
            if image.shape[0] > 1000 or image.shape[1] > 1000:
                print(f"Note: Large image size {image.shape} for {img_path}")
                
            # Only log problematic transformations
            if len(transformed['bboxes']) != len(boxes):
                print(f"Warning: Number of boxes changed after transform for {img_path}")
                print(f"Before: {len(boxes)}, After: {len(transformed['bboxes'])}")
        
        return image, targets, metadata 