from dataclasses import dataclass
from yolo_training_utils import assess_hardware_capabilities

@dataclass
class TrainingConfig:
    """Configuration class for training parameters.
    
    Attributes:
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        input_size (tuple): Input image dimensions (height, width)
        initial_lr (float): Initial learning rate
        warmup_epochs (int): Number of warmup epochs
        num_workers (int): Number of data loading workers
        confidence_threshold (float): Confidence threshold for predictions
        nms_threshold (float): Non-maximum suppression threshold
        max_predictions (int): Maximum number of predictions per image
        early_stopping_patience (int): Epochs to wait before early stopping
        weight_decay (float): Weight decay for optimizer
        dropout (float): Dropout rate
        label_smoothing (float): Label smoothing factor
        export_image_size (tuple): Image size for model export
    """
    num_epochs: int = 100
    batch_size: int = 32
    input_size: tuple = (640, 640)
    initial_lr: float = 1e-3
    warmup_epochs: int = 5  # Increased from 3
    num_workers: int = 8
    
    # Model parameters
    confidence_threshold: float = 0.4
    nms_threshold: float = 0.5
    max_predictions: int = 200
    
    # Training parameters
    early_stopping_patience: int = 10
    weight_decay: float = 5e-4
    dropout: float = 0.1
    label_smoothing: float = 0.1
    
    # Optimizer parameters
    momentum: float = 0.937
    nesterov: bool = True
    
    # Learning rate parameters
    warmup_initial_lr_factor: float = 0.001  # Initial LR will be initial_lr * this factor
    backbone_lr_factor: float = 0.05  # Backbone LR will be initial_lr * this factor
    head_lr_factor: float = 0.1  # Head LR will be initial_lr * this factor
    lr_cooldown_epochs: int = 15
    
    # Advanced training parameters
    gradient_clip_val: float = 0.5
    clip_grad_norm: float = 1.0
    batch_accumulate: int = 2
    ema_decay: float = 0.9999
    
    # Export parameters
    export_image_size: tuple = (320, 320)
    
    def validate(self):
        """Validate configuration parameters"""
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be positive")
        if not (0 <= self.confidence_threshold <= 1):
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not (0 <= self.nms_threshold <= 1):
            raise ValueError("nms_threshold must be between 0 and 1")
        if self.max_predictions < 1:
            raise ValueError("max_predictions must be positive")
        if not all(dim > 0 and dim % 32 == 0 for dim in self.input_size):
            raise ValueError("input_size dimensions must be positive and divisible by 32")
        if not all(dim > 0 and dim % 32 == 0 for dim in self.export_image_size):
            raise ValueError("export_image_size dimensions must be positive and divisible by 32")
        if not (0 < self.initial_lr <= 1):
            raise ValueError("initial_lr must be between 0 and 1")
        if self.warmup_epochs >= self.num_epochs:
            raise ValueError("warmup_epochs must be less than num_epochs")
        if not (0 <= self.weight_decay <= 1):
            raise ValueError("weight_decay must be between 0 and 1")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be between 0 and 1")
        if not (0 <= self.label_smoothing < 1):
            raise ValueError("label_smoothing must be between 0 and 1")
        return self
    
    @classmethod
    def from_gpu_memory(cls, gpu_memory_gb: float) -> "TrainingConfig":
        """Create config based on available GPU memory and hardware capabilities"""
        # Get hardware recommendations
        hw_params = assess_hardware_capabilities()
        
        config = cls(
            batch_size=hw_params['batch_size'],
            num_workers=hw_params['num_workers']
        )
        
        # Adjust other parameters based on GPU memory
        if gpu_memory_gb > 20:  # High-end GPU
            config.initial_lr = 0.001
            config.input_size = (640, 640)
            config.warmup_epochs = 5
            config.max_predictions = 300
            config.export_image_size = (640, 640)
        elif gpu_memory_gb < 8:  # Low-end GPU
            config.initial_lr = 5e-4
            config.input_size = (416, 416)
            config.warmup_epochs = 3
            config.max_predictions = 200
            config.export_image_size = (416, 416)
        else:  # Mid-range GPU
            config.initial_lr = 7e-4
            config.input_size = (512, 512)
            config.warmup_epochs = 4
            config.max_predictions = 250
            config.export_image_size = (512, 512)
            
        return config.validate()