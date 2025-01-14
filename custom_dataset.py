import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from typing import List, Dict, Any, Tuple
import logging

DEBUG_MODE = False

def verify_bbox_format(boxes: torch.Tensor) -> None:
    """Verify bounding box format and values"""
    if len(boxes) > 0:
        if not torch.all((boxes >= 0) & (boxes <= 1)):
            invalid_boxes = boxes[~((boxes >= 0) & (boxes <= 1))]
            logging.warning(f"Invalid box coordinates found: {invalid_boxes}")
            raise ValueError("Box coordinates must be in [0,1]")

def collate_fn(batch: List[Tuple]) -> Tuple:
    """
    Custom collate function to handle variable-sized tensors and match SuperGradients YOLO format.
    
    Args:
        batch: List of tuples containing (image, target, metadata)
        
    Returns:
        Tuple of (images, targets, metadata)
    """
    # Add debug logging
    if DEBUG_MODE:
        print(f"Batch size: {len(batch)}")
        print(f"Sample targets shape: {batch[0][1]['boxes'].shape}")
        print(f"Sample labels shape: {batch[0][1]['labels'].shape}")
    
    images = torch.stack([item[0] for item in batch])
    
    # Verify image sizes
    expected_shape = images[0].shape
    for idx, img in enumerate(images):
        if img.shape != expected_shape:
            raise ValueError(f"Inconsistent image size at index {idx}: "
                           f"got {img.shape}, expected {expected_shape}")
    
    # Convert targets to SuperGradients YOLO format
    # Format: [batch_idx, class_id, x, y, w, h]
    all_targets = []
    for batch_idx, (_, target, _) in enumerate(batch):
        boxes = target['boxes']
        labels = target['labels'].float()  # Ensure float type
        
        if len(boxes) > 0:
            verify_bbox_format(boxes)
            batch_col = torch.full((len(boxes), 1), batch_idx, dtype=torch.float32)
            # Ensure boxes are in correct format
            print(f"Boxes for batch {batch_idx}: {boxes}")
            # Combine into YOLO format
            target_boxes = torch.cat([batch_col, labels.view(-1, 1), boxes], dim=1)
            all_targets.append(target_boxes)
    
    # Concatenate all targets if any exist
    if len(all_targets) > 0:
        targets = torch.cat(all_targets, dim=0)
        assert targets.shape[1] == 6, f"Invalid target shape: {targets.shape}"
    else:
        # Create empty tensor with correct shape if no targets
        targets = torch.zeros((0, 6), dtype=torch.float32)  # 6 columns: [batch_idx, class_id, x, y, w, h]
    
    # Metadata dict for any additional info
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
    
    # Clip centers to [0, 1]
    x_center = np.clip(x_center, 0.0, 1.0)
    y_center = np.clip(y_center, 0.0, 1.0)
    
    # Ensure box stays within image bounds
    width = min(width, 2 * min(x_center, 1 - x_center))
    height = min(height, 2 * min(y_center, 1 - y_center))
    
    # Validate dimensions
    if width <= 0 or height <= 0:
        return None
        
    # Ensure minimum size
    if width < 0.001 or height < 0.001:
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
                
                # Validate transformed boxes
                for box, label in zip(transformed['bboxes'], transformed['class_labels']):
                    if all(0 <= coord <= 1 for coord in box):
                        valid_boxes.append(box)
                        valid_labels.append(label)
                
                if valid_boxes:
                    boxes = torch.tensor(valid_boxes, dtype=torch.float32)
                    class_labels = torch.tensor(valid_labels, dtype=torch.long)
                else:
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
            print(f"Warning: Error applying transforms to {img_path}: {str(e)}")
            # Fallback to basic transformation
            transformed = self.transforms(
                image=image,
                bboxes=np.zeros((0, 4), dtype=np.float32),
                class_labels=np.array([], dtype=np.int64)
            )
            image = transformed['image']
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            class_labels = torch.zeros(0, dtype=torch.long)
        
        image = transformed['image']
        
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
        
        return image, targets, metadata 