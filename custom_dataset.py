import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os

def clip_bbox(bbox):
    """Clip bounding box coordinates to be within [0, 1]"""
    x_center, y_center, width, height = bbox
    x_center = np.clip(x_center, 0.0, 1.0)
    y_center = np.clip(y_center, 0.0, 1.0)
    # Ensure width and height are positive and don't exceed boundaries
    width = np.clip(width, 0.0, min(2 * x_center, 2 * (1 - x_center)))
    height = np.clip(height, 0.0, min(2 * y_center, 2 * (1 - y_center)))
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
        # Get image path
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read corresponding label file
        label_path = os.path.join(
            self.labels_dir,
            img_name.rsplit('.', 1)[0] + '.txt'
        )
        
        # Parse YOLO format labels
        boxes = []
        class_labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    try:
                        class_id, x_center, y_center, width, height = map(float, line.strip().split())
                        # Clip bbox coordinates to valid range
                        bbox = clip_bbox([x_center, y_center, width, height])
                        if all(0 <= coord <= 1 for coord in bbox):  # Additional validation
                            boxes.append(bbox)
                            class_labels.append(class_id)
                    except Exception as e:
                        print(f"Warning: Skipping invalid bbox in {label_path}: {line.strip()} - {str(e)}")
                        continue
        
        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        class_labels = np.array(class_labels, dtype=np.int64)
        
        # Apply augmentations
        try:
            if len(boxes) > 0:
                transformed = self.transforms(
                    image=image,
                    bboxes=boxes,
                    class_labels=class_labels
                )
                
                image = transformed['image']
                boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32) if transformed['bboxes'] else torch.zeros((0, 4), dtype=torch.float32)
                class_labels = torch.tensor(transformed['class_labels'], dtype=torch.long) if transformed['class_labels'] else torch.zeros(0, dtype=torch.long)
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
        
        return {
            'image': image,
            'boxes': boxes,
            'class_labels': class_labels,
            'image_path': img_path
        } 