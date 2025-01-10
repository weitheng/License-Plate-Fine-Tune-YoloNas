import os
import torch
from super_gradients.training import Trainer, models
from super_gradients.common.object_names import Models
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val
)
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
import wandb
from yolo_training_utils import assess_hardware_capabilities, load_dataset_config, setup_directories
from torch.optim.lr_scheduler import OneCycleLR

class KnowledgeDistillationLoss(torch.nn.Module):
    def __init__(self, student_loss_fn, temperature=4.0, alpha=0.5):
        super().__init__()
        self.student_loss_fn = student_loss_fn
        self.temperature = temperature
        self.alpha = alpha
        
    def forward(self, student_outputs, targets):
        # Get teacher outputs from targets dict
        teacher_outputs = targets.pop('teacher_outputs', None)
        
        # Standard student loss
        student_loss = self.student_loss_fn(student_outputs, targets)
        
        # If no teacher outputs, return only student loss
        if teacher_outputs is None:
            return student_loss
        
        # Distillation loss
        distillation_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_outputs / self.temperature, dim=1),
            torch.nn.functional.softmax(teacher_outputs / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = (1 - self.alpha) * student_loss + self.alpha * distillation_loss
        return total_loss

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model.to(self.device)
        
    def train_batch(self, batch_idx, batch_data, **kwargs):
        images, targets = batch_data
        images = images.to(self.device)
        
        # Get teacher predictions
        with torch.no_grad():
            teacher_outputs = self.teacher_model(images)
        
        # Create a copy of targets to avoid modifying the original
        targets_with_teacher = targets.copy()
        targets_with_teacher['teacher_outputs'] = teacher_outputs
        
        # Call parent's train_batch with modified targets
        return super().train_batch(batch_idx, (images, targets_with_teacher), **kwargs)

def download_model_weights(model_name, target_path):
    """Download model weights from alternative sources if primary fails"""
    urls = {
        'YOLO_NAS_L': [
            'https://sg-hub-nv.s3.amazonaws.com/models/yolo_nas_l_coco.pth',
            'https://storage.googleapis.com/super-gradients-models/yolo_nas_l_coco.pth'
        ],
        'YOLO_NAS_S': [
            'https://sg-hub-nv.s3.amazonaws.com/models/yolo_nas_s_coco.pth',
            'https://storage.googleapis.com/super-gradients-models/yolo_nas_s_coco.pth'
        ]
    }
    
    for url in urls.get(model_name, []):
        try:
            print(f"Attempting to download from: {url}")
            os.system(f"wget {url} -O {target_path}")
            if os.path.exists(target_path):
                return True
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
    return False

def main():
    # Initialize wandb
    wandb.login()
    wandb.init(project="license-plate-detection", name="yolo-nas-s-distillation")

    # Setup directories
    checkpoint_dir, export_dir = setup_directories("./")

    # Load dataset configuration
    yaml_path = "license_plate_dataset.yaml"
    dataset_config = load_dataset_config(yaml_path)

    # Get optimal hardware settings
    hw_params = assess_hardware_capabilities()

    # Create cache directory if it doesn't exist
    cache_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints/')
    os.makedirs(cache_dir, exist_ok=True)

    # Download model weights if needed
    l_model_path = os.path.join(cache_dir, 'yolo_nas_l_coco.pth')
    s_model_path = os.path.join(cache_dir, 'yolo_nas_s_coco.pth')

    if not os.path.exists(l_model_path):
        download_model_weights('YOLO_NAS_L', l_model_path)
    if not os.path.exists(s_model_path):
        download_model_weights('YOLO_NAS_S', s_model_path)

    # Fix download URLs for YOLO-NAS models
    print("Fixing model download URLs...")
    os.system('sed -i \'s/sghub.deci.ai/sg-hub-nv.s3.amazonaws.com/g\' /usr/local/lib/python3.10/dist-packages/super_gradients/training/pretrained_models.py')
    os.system('sed -i \'s/sghub.deci.ai/sg-hub-nv.s3.amazonaws.com/g\' /usr/local/lib/python3.10/dist-packages/super_gradients/training/utils/checkpoint_utils.py')
    os.system('sed -i \'s/https:\/\/\/models/https:\/\/models/g\' /usr/local/lib/python3.10/dist-packages/super_gradients/training/pretrained_models.py')
    os.system('sed -i \'s/https:\/\/\/models/https:\/\/models/g\' /usr/local/lib/python3.10/dist-packages/super_gradients/training/utils/checkpoint_utils.py')

    # Setup teacher model with COCO weights
    teacher_model = models.get(Models.YOLO_NAS_L, pretrained_weights="coco")
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Initialize student model with COCO weights and modify last layer for 81 classes
    model = models.get(Models.YOLO_NAS_S, 
                      num_classes=81,  # 80 COCO classes + 1 license plate class
                      pretrained_weights="coco")  # Start with COCO weights
    
    # Modify the model's classification head for 81 classes
    # The exact modification depends on the model architecture
    # The pretrained weights for the first 80 classes are preserved
    
    # Update the loss function for 81 classes
    base_loss = PPYoloELoss(
        use_static_assigner=False,
        num_classes=81,  # Updated for total number of classes
        reg_max=16,
        iou_loss_weight=3.0
    )

    # Create knowledge distillation loss
    distillation_loss = KnowledgeDistillationLoss(
        student_loss_fn=base_loss,
        temperature=4.0,
        alpha=0.5
    )

    # Prepare dataloaders with data augmentation
    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': './',
            'images_dir': 'images/train',
            'labels_dir': 'labels/train',
            'classes': dataset_config['names'],
            'input_dim': (640, 640),
        },
        dataloader_params={
            'batch_size': hw_params['batch_size'],
            'num_workers': 4,  # Hardcoded num_workers
            'shuffle': True,
            'pin_memory': torch.cuda.is_available(),
            'drop_last': True  # Helps with batch normalization
        }
    )

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': './',
            'images_dir': 'images/val',
            'labels_dir': 'labels/val',
            'classes': dataset_config['names'],
            'input_dim': (640, 640)
        },
        dataloader_params={
            'batch_size': hw_params['batch_size'],
            'num_workers': 4,  # Hardcoded num_workers
            'shuffle': False,
            'pin_memory': torch.cuda.is_available(),
            'drop_last': False
        }
    )

    # Update metrics for 81 classes
    train_params = {
        'save_ckpt_after_epoch': True,
        'save_ckpt_dir': checkpoint_dir,
        'resume': False,
        'silent_mode': False,
        'average_best_models': True,
        'warmup_mode': 'LinearEpochLRWarmup',
        'warmup_initial_lr': 1e-6,
        'lr_warmup_epochs': 5,
        'initial_lr': 1e-4,
        'lr_mode': 'cosine',  # Changed back to cosine
        'cosine_final_lr_ratio': 0.1,
        'optimizer': 'AdamW',
        'optimizer_params': {'weight_decay': 0.001},
        'zero_weight_decay_on_bias_and_bn': True,
        'ema': True,
        'ema_params': {
            'decay': 0.9995,
            'decay_type': 'exp',
            'beta': 10
        },
        'max_epochs': 65,
        'early_stopping_patience': 5,
        'mixed_precision': torch.cuda.is_available(),
        'loss': distillation_loss,
        'valid_metrics_list': [
            DetectionMetrics_050(
                score_thres=0.3,
                top_k_predictions=200,
                num_cls=81,  # Updated for total number of classes
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.3,
                    nms_threshold=0.5,
                    nms_top_k=500,
                    max_predictions=200
                )
            )
        ],
        'metric_to_watch': 'mAP@0.50',
        'sg_logger': 'wandb_sg_logger',
        'sg_logger_params': {
            'save_checkpoints_remote': True,
            'save_tensorboard_remote': True,
            'save_checkpoint_as_artifact': True,
            'project_name': 'license-plate-detection',
            'run_name': 'yolo-nas-s-distillation'
        },
        'dropout': 0.1,
        'label_smoothing': 0.1
    }

    # Initialize trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        experiment_name='coco_license_plate_detection',
        ckpt_root_dir=checkpoint_dir
    )

    # Train the model
    trainer.train(
        model=model,
        training_params=train_params,
        train_loader=train_data,
        valid_loader=val_data
    )

    # Save final model checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, 'coco_license_plate_detection_final.pth')
    trainer.save_checkpoint(model_state=model.state_dict(), optimizer_state=None, scheduler_state=None, checkpoint_path=final_checkpoint_path)

    # Generate complete label map file
    label_map_path = os.path.join(checkpoint_dir, 'label_map.txt')
    with open(label_map_path, 'w') as f:
        for idx, class_name in enumerate(dataset_config['names']):
            f.write(f"{idx}: {class_name}\n")

    # Export model to ONNX format
    onnx_path = os.path.join(export_dir, "yolo_nas_s_coco_license_plate.onnx")
    model.export(
        onnx_path,
        output_predictions_format="FLAT_FORMAT",
        max_predictions_per_image=20,
        confidence_threshold=0.4,
        input_image_shape=(320, 320)
    )

    print(f"\nTraining completed!")
    print(f"Checkpoint saved to: {final_checkpoint_path}")
    print(f"Label map saved to: {label_map_path}")
    print(f"ONNX model exported to: {onnx_path}")

    # Finish wandb session
    wandb.finish()

if __name__ == "__main__":
    main()
