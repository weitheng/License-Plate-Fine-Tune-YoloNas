import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os

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
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    boxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)
        
        # Convert to numpy arrays
        boxes = np.array(boxes)
        class_labels = np.array(class_labels)
        
        # Apply augmentations
        if len(boxes) > 0:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                class_labels=class_labels
            )
            
            image = transformed['image']
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            class_labels = torch.tensor(transformed['class_labels'], dtype=torch.long)
        else:
            # Handle cases with no boxes
            transformed = self.transforms(image=image)
            image = transformed['image']
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            class_labels = torch.zeros(0, dtype=torch.long)
        
        return {
            'image': image,
            'boxes': boxes,
            'class_labels': class_labels,
            'image_path': img_path
        } 