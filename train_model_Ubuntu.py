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
from pycocotools.coco import COCO
import requests
import zipfile
from tqdm import tqdm
import logging
from config import TrainingConfig
from typing import Optional, List, Dict, Any, Tuple
import time
import coloredlogs

def setup_logging():
    """Setup logging with colored output for terminal and file output"""
    # Format for both file and terminal
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Setup file handler
    file_handler = logging.FileHandler('training.log')
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Get the root logger and clear any existing handlers
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Add the file handler
    logger.addHandler(file_handler)
    
    # Install colored logs for terminal
    coloredlogs.install(
        level='INFO',
        logger=logger,
        fmt=log_format,
        level_styles={
            'info': {'color': 'white'},
            'warning': {'color': 'yellow'},
            'error': {'color': 'red', 'bold': True},
            'success': {'color': 'green', 'bold': True}  # For checkmarks
        },
        field_styles={
            'asctime': {'color': 'cyan'},
            'levelname': {'color': 'magenta', 'bold': True}
        }
    )
    
    return logger

# Add success level for checkmarks
logging.addLevelName(25, 'SUCCESS')
def success(self, message, *args, **kwargs):
    if self.isEnabledFor(25):
        self._log(25, message, args, **kwargs)
logging.Logger.success = success

logger = setup_logging()

def download_model_weights(model_name: str, target_path: str) -> bool:
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

def download_coco_subset(target_dir, num_images=70000):
    """Download a subset of COCO dataset"""
    # Create directories
    os.makedirs(os.path.join(target_dir, 'coco/images/train'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'coco/images/val'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'coco/labels/train'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'coco/labels/val'), exist_ok=True)

    # URLs for both images and annotations
    urls = {
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }
    
    for name, url in urls.items():
        zip_path = os.path.join(target_dir, f'coco_{name}.zip')
        if not os.path.exists(zip_path):
            try:
                print(f"Downloading COCO {name}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                with open(zip_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for data in response.iter_content(chunk_size=8192):
                        f.write(data)
                        pbar.update(len(data))
                
                print(f"Extracting {name}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.join(target_dir, 'coco'))
                    
            except Exception as e:
                print(f"Error downloading {name}: {e}")
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                return False
    return True

def convert_coco_to_yolo(coco_dir, target_dir, num_images=70000):
    """Convert COCO annotations to YOLO format"""
    try:
        splits = ['train2017', 'val2017']
        
        for split in splits:
            anno_file = os.path.join(coco_dir, 'annotations', f'instances_{split}.json')
            if not os.path.exists(anno_file):
                raise FileNotFoundError(f"Missing annotation file: {anno_file}")
            
            print(f"Processing {split} split...")
            coco = COCO(anno_file)
            
            # Get image ids and categories
            img_ids = coco.getImgIds()
            if split == 'train2017':
                img_ids = img_ids[:num_images]  # Limit only training images
            
            # Create category mapping
            cat_ids = coco.getCatIds()
            cat_map = {old_id: new_id for new_id, old_id in enumerate(cat_ids)}
            
            # Convert annotations
            for img_id in tqdm(img_ids, desc=f"Converting {split}"):
                img_info = coco.loadImgs(img_id)[0]
                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)
                
                # Create YOLO format annotations
                yolo_anns = []
                for ann in anns:
                    cat_id = cat_map[ann['category_id']]  # Map to new category ID
                    bbox = ann['bbox']
                    x_center = (bbox[0] + bbox[2]/2) / img_info['width']
                    y_center = (bbox[1] + bbox[3]/2) / img_info['height']
                    width = bbox[2] / img_info['width']
                    height = bbox[3] / img_info['height']
                    yolo_anns.append(f"{cat_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
                # Save annotations
                out_dir = 'train' if split == 'train2017' else 'val'
                with open(os.path.join(target_dir, 'labels', out_dir, f"{img_info['file_name'].split('.')[0]}.txt"), 'w') as f:
                    f.write('\n'.join(yolo_anns))

    except Exception as e:
        print(f"Error converting COCO to YOLO format: {e}")
        raise

def check_coco_dataset(coco_dir: str) -> bool:
    """
    Check if COCO dataset is already downloaded and processed
    Returns True if dataset exists and is complete
    """
    required_files = {
        'train': {
            'images': os.path.join(coco_dir, 'images/train'),
            'labels': os.path.join(coco_dir, 'labels/train'),
            'annotations': os.path.join(coco_dir, 'annotations/instances_train2017.json')
        },
        'val': {
            'images': os.path.join(coco_dir, 'images/val'),
            'labels': os.path.join(coco_dir, 'labels/val'),
            'annotations': os.path.join(coco_dir, 'annotations/instances_val2017.json')
        }
    }
    
    # Check if all required directories and files exist
    for split_data in required_files.values():
        for path in split_data.values():
            if not os.path.exists(path):
                return False
            # Check if directories have content
            if os.path.isdir(path) and not os.listdir(path):
                return False
    return True

def prepare_combined_dataset() -> None:
    """Prepare combined COCO and license plate dataset"""
    logger.info("=== Starting Dataset Preparation ===")
    
    # Create combined dataset directories
    logger.info("Step 1/4: Creating directory structure...")
    combined_dir = './data/combined'
    coco_dir = './data/coco'
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(combined_dir, f'images/{split}'), exist_ok=True)
        os.makedirs(os.path.join(combined_dir, f'labels/{split}'), exist_ok=True)
    logger.success("✓ Directory structure created")

    # Check if COCO dataset already exists
    logger.info("Step 2/4: Processing COCO dataset...")
    if not check_coco_dataset(coco_dir):
        logger.info("   - COCO dataset not found, downloading...")
        if download_coco_subset('./data'):
            logger.info("   - Converting COCO to YOLO format...")
            convert_coco_to_yolo(coco_dir, combined_dir)
        else:
            raise RuntimeError("Failed to download COCO dataset")
    else:
        logger.info("   - COCO dataset found, converting to YOLO format...")
        convert_coco_to_yolo(coco_dir, combined_dir)
    logger.info("✓ COCO dataset processed")

    # Check if combined dataset already exists
    logger.info("Step 3/4: Checking existing combined dataset...")
    if os.path.exists(combined_dir):
        try:
            validate_dataset_contents(combined_dir)
            logger.info("✓ Existing combined dataset is valid")
            return
        except Exception as e:
            logger.warning(f"   - Existing dataset invalid: {e}")
            logger.info("   - Will recreate combined dataset")
    
    # Copy license plate data with prefix to avoid conflicts
    logger.info("Step 4/4: Processing license plate data...")
    total_copied = 0
    for split in ['train', 'val']:
        images_dir = f'images/{split}'
        labels_dir = f'labels/{split}'
        if os.path.exists(images_dir) and os.path.exists(labels_dir):
            label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            
            with tqdm(total=len(label_files), desc=f"Copying {split} split") as pbar:
                for label_file in label_files:
                    img_base = label_file.replace('.txt', '')
                    img_extensions = ['.jpg', '.jpeg', '.png']
                    img_file = None
                    for ext in img_extensions:
                        if os.path.exists(os.path.join(images_dir, img_base + ext)):
                            img_file = img_base + ext
                            break
                    
                    if img_file:
                        os.system(f'cp "{os.path.join(images_dir, img_file)}" "{os.path.join(combined_dir, f"images/{split}/lp_{img_file}")}"')
                        os.system(f'cp "{os.path.join(labels_dir, label_file)}" "{os.path.join(combined_dir, f"labels/{split}/lp_{label_file}")}"')
                        total_copied += 1
                    pbar.update(1)
    
    logger.info(f"✓ License plate data processed ({total_copied} pairs copied)")
    logger.info("=== Dataset Preparation Complete ===\n")

def validate_dataset(data_dir: str) -> None:
    """Validate combined dataset structure"""
    required_dirs = [
        'images/train', 'images/val',
        'labels/train', 'labels/val'
    ]
    for d in required_dirs:
        path = os.path.join(data_dir, d)
        if not os.path.exists(path):
            raise RuntimeError(f"Missing required directory: {path}")
        
        # Check if directory has files
        if len(os.listdir(path)) == 0:
            raise RuntimeError(f"Directory is empty: {path}")

def validate_dataset_contents(data_dir: str) -> None:
    """Validate dataset contents and format"""
    for split in ['train', 'val']:
        images_dir = os.path.join(data_dir, f'images/{split}')
        labels_dir = os.path.join(data_dir, f'labels/{split}')
        
        # Check image-label pairs
        image_files = set(f.split('.')[0] for f in os.listdir(images_dir))
        label_files = set(f.split('.')[0] for f in os.listdir(labels_dir))
        
        # Check for missing files
        missing_labels = image_files - label_files
        missing_images = label_files - image_files
        
        if missing_labels:
            logger.warning(f"Images without labels in {split}: {missing_labels}")
        if missing_images:
            logger.warning(f"Labels without images in {split}: {missing_images}")

def cleanup_downloads():
    """Clean up downloaded files after processing"""
    try:
        # Only remove zip files, keep processed data
        for file in os.listdir('./data'):
            if file.startswith('coco_') and file.endswith('.zip'):
                zip_path = os.path.join('./data', file)
                if os.path.exists(zip_path):
                    logger.info(f"Removing downloaded zip: {file}")
                    os.remove(zip_path)
    except Exception as e:
        logger.warning(f"Error cleaning up downloads: {e}")

class TrainingProgressCallback:
    def __init__(self):
        self.best_map = 0
        self.best_epoch = 0
        self.start_time = None
        
    def __call__(self, epoch, metrics, context):
        if self.start_time is None:
            self.start_time = time.time()
            
        current_map = metrics.get('mAP@0.50', 0)
        if current_map > self.best_map:
            self.best_map = current_map
            self.best_epoch = epoch
            logger.info(f"New best mAP: {self.best_map:.4f} at epoch {epoch}")
            
        # Log training progress
        elapsed_time = time.time() - self.start_time
        logger.info(f"Epoch {epoch}: mAP={current_map:.4f}, Best={self.best_map:.4f} (epoch {self.best_epoch}), Time={elapsed_time/3600:.1f}h")

def main():
    try:
        logger.info("Starting training pipeline...")
        
        # Prepare dataset first
        prepare_combined_dataset()
        
        # Validate dataset structure
        logger.info("Validating final dataset structure...")
        validate_dataset('./data/combined')
        validate_dataset_contents('./data/combined')
        logger.info("✓ Dataset validation complete")
        
        # Initialize wandb
        logger.info("Initializing Weights & Biases...")
        wandb.login()
        wandb.init(project="license-plate-detection", name="yolo-nas-s-coco-finetuning")
        logger.info("✓ Weights & Biases initialized")

        # Setup directories
        logger.info("Setting up directories...")
        checkpoint_dir, export_dir = setup_directories("./")
        logger.info(f"✓ Directories set up: {checkpoint_dir}, {export_dir}")

        # Load dataset configuration
        logger.info("Loading dataset configuration...")
        yaml_path = "license_plate_dataset.yaml"
        dataset_config = load_dataset_config(yaml_path)
        logger.info("✓ Dataset configuration loaded")

        # Get optimal hardware settings
        hw_params = assess_hardware_capabilities()

        # Create cache directory if it doesn't exist
        cache_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints/')
        os.makedirs(cache_dir, exist_ok=True)

        # Download model weights if needed
        logger.info("Checking model weights...")
        l_model_path = os.path.join(cache_dir, 'yolo_nas_l_coco.pth')
        s_model_path = os.path.join(cache_dir, 'yolo_nas_s_coco.pth')

        if not os.path.exists(l_model_path):
            logger.info("Downloading YOLO-NAS-L weights...")
            download_model_weights('YOLO_NAS_L', l_model_path)
        if not os.path.exists(s_model_path):
            logger.info("Downloading YOLO-NAS-S weights...")
            download_model_weights('YOLO_NAS_S', s_model_path)
        logger.info("✓ Model weights ready")

        # Fix download URLs for YOLO-NAS models
        print("Fixing model download URLs...")
        os.system('sed -i \'s/sghub.deci.ai/sg-hub-nv.s3.amazonaws.com/g\' /usr/local/lib/python3.10/dist-packages/super_gradients/training/pretrained_models.py')
        os.system('sed -i \'s/sghub.deci.ai/sg-hub-nv.s3.amazonaws.com/g\' /usr/local/lib/python3.10/dist-packages/super_gradients/training/utils/checkpoint_utils.py')
        os.system('sed -i \'s/https:\/\/\/models/https:\/\/models/g\' /usr/local/lib/python3.10/dist-packages/super_gradients/training/pretrained_models.py')
        os.system('sed -i \'s/https:\/\/\/models/https:\/\/models/g\' /usr/local/lib/python3.10/dist-packages/super_gradients/training/utils/checkpoint_utils.py')

        # Initialize model with COCO weights
        model = models.get(Models.YOLO_NAS_S, 
                          num_classes=81,  # 80 COCO classes + 1 license plate class
                          pretrained_weights="coco")
        
        # Define loss function
        loss_fn = PPYoloELoss(
            use_static_assigner=False,
            num_classes=81,
            reg_max=16,
            iou_loss_weight=3.0
        )

        # Get GPU memory if available
        gpu_memory_gb = 0
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Initialize config based on hardware
        config = TrainingConfig.from_gpu_memory(gpu_memory_gb)
        
        # Update training parameters with config values
        train_params = {
            'save_ckpt_after_epoch': True,
            'save_ckpt_dir': checkpoint_dir,
            'resume': False,
            'silent_mode': False,
            'average_best_models': True,
            'warmup_mode': 'LinearEpochLRWarmup',
            'warmup_initial_lr': 1e-6,
            'lr_warmup_epochs': config.warmup_epochs,
            'initial_lr': config.initial_lr,
            'lr_mode': 'cosine',
            'max_epochs': config.num_epochs,
            'early_stopping_patience': config.early_stopping_patience,
            'mixed_precision': torch.cuda.is_available(),
            'loss': loss_fn,
            'valid_metrics_list': [
                DetectionMetrics_050(
                    score_thres=config.confidence_threshold,
                    top_k_predictions=config.max_predictions,
                    num_cls=81,
                    normalize_targets=True,
                    post_prediction_callback=PPYoloEPostPredictionCallback(
                        score_threshold=config.confidence_threshold,
                        nms_threshold=config.nms_threshold,
                        nms_top_k=config.max_predictions,
                        max_predictions=config.max_predictions
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
                'run_name': 'yolo-nas-s-coco-finetuning'
            },
            'dropout': config.dropout,
            'label_smoothing': config.label_smoothing,
            'resume_path': os.path.join(checkpoint_dir, 'latest_checkpoint.pth'),
            'resume_strict_load': False,
            'optimizer_params': {'weight_decay': config.weight_decay}
        }

        # Update dataloader params
        train_data = coco_detection_yolo_format_train(
            dataset_params={
                'data_dir': './data/combined',
                'images_dir': 'images/train',
                'labels_dir': 'labels/train',
                'classes': dataset_config['names'],
                'input_dim': config.input_size,
            },
            dataloader_params={
                'batch_size': config.batch_size,
                'num_workers': config.num_workers,
                'shuffle': True,
                'pin_memory': torch.cuda.is_available(),
                'drop_last': True
            }
        )

        val_data = coco_detection_yolo_format_val(
            dataset_params={
                'data_dir': './data/combined',
                'images_dir': 'images/val',
                'labels_dir': 'labels/val',
                'classes': dataset_config['names'],
                'input_dim': config.input_size,
            },
            dataloader_params={
                'batch_size': config.batch_size,
                'num_workers': config.num_workers,
                'shuffle': False,
                'pin_memory': torch.cuda.is_available(),
                'drop_last': False
            }
        )

        # Initialize standard trainer
        trainer = Trainer(
            experiment_name='coco_license_plate_detection',
            ckpt_root_dir=checkpoint_dir
        )

        # Initialize progress callback
        progress_callback = TrainingProgressCallback()
        
        # Add callback to training params
        train_params['phase_callbacks'] = [progress_callback]
        
        # Validate dataset contents before training
        validate_dataset_contents('./data/combined')
        
        # Train model
        trainer.train(
            model=model,
            training_params=train_params,
            train_loader=train_data,
            valid_loader=val_data
        )
        
        # Cleanup downloaded files
        cleanup_downloads()
        
        # Save final model checkpoint
        final_checkpoint_path = os.path.join(checkpoint_dir, 'coco_license_plate_detection_final.pth')
        trainer.save_checkpoint(model_state=model.state_dict(), optimizer_state=None, scheduler_state=None, checkpoint_path=final_checkpoint_path)

        # Generate complete label map file
        label_map_path = os.path.join(checkpoint_dir, 'label_map.txt')
        with open(label_map_path, 'w') as f:
            for idx, class_name in enumerate(dataset_config['names']):
                f.write(f"{idx}: {class_name}\n")

        # Export model to ONNX format with reduced size for efficient inference
        onnx_path = os.path.join(export_dir, "yolo_nas_s_coco_license_plate.onnx")
        model.export(
            onnx_path,
            output_predictions_format="FLAT_FORMAT",
            max_predictions_per_image=config.max_predictions,
            confidence_threshold=config.confidence_threshold,
            input_image_shape=config.export_image_size  # Using smaller size for inference
        )

        print(f"\nTraining completed!")
        print(f"Checkpoint saved to: {final_checkpoint_path}")
        print(f"Label map saved to: {label_map_path}")
        print(f"ONNX model exported to: {onnx_path}")

        # Finish wandb session
        wandb.finish()

    except Exception as e:
        logger.error(f"Error during training: {e}")
        wandb.finish()
        raise
    finally:
        # Always try to cleanup
        try:
            cleanup_downloads()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

if __name__ == "__main__":
    main()
