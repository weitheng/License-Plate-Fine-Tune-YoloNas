from dataclasses import dataclass

@dataclass
class TrainingConfig:
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
        return self
    
    @classmethod
    def from_gpu_memory(cls, gpu_memory_gb: float) -> "TrainingConfig":
        config = cls()
        if gpu_memory_gb < 8:
            config.batch_size = 4
        elif gpu_memory_gb < 16:
            config.batch_size = 8
        return config.validate()