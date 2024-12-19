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


def main():
    # Initialize wandb
    wandb.login()
    wandb.init(project="license-plate-detection", name="yolo-nas-s-finetuning")

    # Setup directories
    checkpoint_dir, export_dir = setup_directories("./")

    # Load dataset configuration
    yaml_path = "license_plate_dataset.yaml"
    dataset_config = load_dataset_config(yaml_path)

    # Get optimal hardware settings
    hw_params = assess_hardware_capabilities()

    # Prepare dataloaders with data augmentation
    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': './',
            'images_dir': 'images/train',
            'labels_dir': 'labels/train',
            'classes': dataset_config['names'],
            'input_dim': (640, 640),
            'augmentations': {
                'flip': 0.5,  # Random horizontal flip
                'scale': [0.8, 1.2],  # Random scaling
                'crop': [0.8, 1.0],  # Random cropping
            }
        },
        dataloader_params={
            'batch_size': hw_params['batch_size'],
            'num_workers': 4,
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
            'num_workers': 4,
            'shuffle': False,
            'pin_memory': torch.cuda.is_available(),
            'drop_last': False
        }
    )

    # Define the YOLO-NAS-S model
    model = models.get(Models.YOLO_NAS_S, num_classes=len(dataset_config['names']))

    # Training parameters
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
        'lr_mode': 'cosine',
        'cosine_final_lr_ratio': 0.01,
        'optimizer': 'AdamW',
        'optimizer_params': {
            'weight_decay': 0.001
        },
        'zero_weight_decay_on_bias_and_bn': True,
        'ema': True,
        'ema_params': {
            'decay': 0.9995,
            'decay_type': 'exp',
            'beta': 10
        },
        'max_epochs': 40,
        'early_stopping_patience': 5,
        'mixed_precision': torch.cuda.is_available(),
        'loss': PPYoloELoss(
            use_static_assigner=False,
            num_classes=len(dataset_config['names']),
            reg_max=16,
            iou_loss_weight=3.0
        ),
        'valid_metrics_list': [
            DetectionMetrics_050(
                score_thres=0.3,
                top_k_predictions=200,
                num_cls=len(dataset_config['names']),
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
            'run_name': 'yolo-nas-s-finetuning'
        },
        'dropout': 0.1,
        'label_smoothing': 0.1
    }

    # Scheduler integration
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_params['initial_lr'], weight_decay=0.001)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=train_params['initial_lr'],
        steps_per_epoch=len(train_data),
        epochs=train_params['max_epochs'],
        pct_start=0.3
    )

    # Initialize trainer and start training
    trainer = Trainer(experiment_name='license_plate_detection', ckpt_root_dir=checkpoint_dir)

    trainer.train(
        model=model,
        training_params=train_params,
        train_loader=train_data,
        valid_loader=val_data,
        optimizer=optimizer,
        scheduler=scheduler
    )

    # Save final model checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, 'license_plate_detection_final.pth')
    trainer.save_checkpoint(model_state=model.state_dict(), optimizer_state=None, scheduler_state=None, checkpoint_path=final_checkpoint_path)

    # Generate label map file
    label_map_path = os.path.join(checkpoint_dir, 'label_map.txt')
    with open(label_map_path, 'w') as f:
        for idx, class_name in enumerate(dataset_config['names']):
            f.write(f"{idx}: {class_name}\n")

    # Export model to ONNX format
    onnx_path = os.path.join(export_dir, "yolo_nas_s_fine_tuned.onnx")
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
