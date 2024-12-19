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

    # Prepare dataloaders with optimized parameters
    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': './',
            'images_dir': 'images/train',
            'labels_dir': 'labels/train',
            'classes': dataset_config['names'],
            'input_dim': (640, 640),
            'transforms': [
                {'DetectionRandomAffine': {
                    'degrees': 15
                }}
            ]
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
            'input_dim': (640, 640)  # Add this line
        },
        dataloader_params={
            'batch_size': hw_params['batch_size'],
            'num_workers': 4,
            'shuffle': False,
            'pin_memory': torch.cuda.is_available(),
            'drop_last': False
        }
    )

    # Fix download URLs for YOLO-NAS S model
    print("Fixing model download URLs...")
    os.system('sed -i \'s/sghub.deci.ai/sg-hub-nv.s3.amazonaws.com/\' /usr/local/lib/python3.8/dist-packages/super_gradients/training/pretrained_models.py')
    os.system('sed -i \'s/sghub.deci.ai/sg-hub-nv.s3.amazonaws.com/\' /usr/local/lib/python3.8/dist-packages/super_gradients/training/utils/checkpoint_utils.py')

    # Define the YOLO-NAS-S model
    model = models.get(Models.YOLO_NAS_S, num_classes=len(dataset_config['names']))

    # Define checkpoint directory
    checkpoint_dir = os.path.abspath("./checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure the directory exists

    # Define checkpoint directory
    export_dir = os.path.abspath("./export")
    os.makedirs(export_dir, exist_ok=True)  # Ensure the directory exists# Training parameters

    train_params = {
        'save_ckpt_after_epoch': True,
        'save_ckpt_dir': checkpoint_dir,
        'resume': False,
        'silent_mode': False,
        'average_best_models': True,
        'warmup_mode': 'LinearEpochLRWarmup',
        'warmup_initial_lr': 1e-6,
        'lr_warmup_epochs': 5,  #Increase from 3
        'initial_lr': 1e-4,  # Slightly lower learning rate
        'lr_mode': 'cosine',
        'cosine_final_lr_ratio': 0.01,
        'optimizer': 'AdamW',  # Switch to AdamW from Adam
        'optimizer_params': {
            'weight_decay': 0.001,  # Increase weight decay
            'momentum': 0.937,
        },
        'zero_weight_decay_on_bias_and_bn': True,
        'ema': True,
        'ema_params': {
            'decay': 0.9995,
            'decay_type': 'exp',
            'beta': 10
        },
        'max_epochs': 50,
        'mixed_precision': torch.cuda.is_available(),
        'loss': PPYoloELoss(
            use_static_assigner=False,
            num_classes=len(dataset_config['names']),
            reg_max=16,
            iou_loss_weight=3.0  # Increase IOU loss weight
        ),
        'valid_metrics_list': [
        DetectionMetrics_050(
            score_thres=0.3,  # Increase confidence threshold
            top_k_predictions=200,  # Reduce top k predictions
            num_cls=len(dataset_config['names']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.3,  # Increase score threshold
                nms_threshold=0.5,  # Stricter NMS
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
        }
    }
    train_params['callback_kwargs'] = {
        'early_stopping_patience': 5,
        'monitor': 'val_precision_50',  # Monitor precision
        'mode': 'max'
    }

    train_params['input_size_range'] = {
        'min': 320,
        'max': 640,
        'step': 32
    }

    train_params.update({
        'dropout': 0.1,
        'mixed_precision': True,
        'label_smoothing': 0.1
    })
    # Initialize trainer and start training
    trainer = Trainer(experiment_name='license_plate_detection',
                     ckpt_root_dir=checkpoint_dir)

    trainer.train(
        model=model,
        training_params=train_params,
        train_loader=train_data,
        valid_loader=val_data
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
