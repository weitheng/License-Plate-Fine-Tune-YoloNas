from dataclasses import dataclass

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
    num_epochs: int = 65
    batch_size: int = 8
    input_size: tuple = (640, 640)
    initial_lr: float = 1e-4
    warmup_epochs: int = 5
    num_workers: int = 4
    
    # Model parameters
    confidence_threshold: float = 0.4
    nms_threshold: float = 0.5
    max_predictions: int = 200
    
    # Training parameters
    early_stopping_patience: int = 5
    weight_decay: float = 0.001
    dropout: float = 0.1
    label_smoothing: float = 0.1
    
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
        config = cls()
        if gpu_memory_gb < 8:
            config.batch_size = 4
        elif gpu_memory_gb < 16:
            config.batch_size = 8
        return config.validate()