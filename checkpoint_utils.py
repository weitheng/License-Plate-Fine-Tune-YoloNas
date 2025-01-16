import os
import torch
import logging
from typing import Dict
from yolo_training_utils import verify_checkpoint

logger = logging.getLogger(__name__)

def setup_checkpoint_resuming(checkpoint_dir: str, train_params: dict, force_new: bool = False) -> dict:
    """
    Setup checkpoint resuming logic with proper validation.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        train_params: Current training parameters
        force_new: If True, ignore existing checkpoints and start fresh
        
    Returns:
        Updated training parameters dict
    """
    if force_new:
        logger.info("Starting fresh training as requested (--no-resume)")
        return {**train_params, 'resume': False, 'resume_path': None}

    # First find the experiment directory (most recent RUN_* directory)
    experiment_name = 'coco_license_plate_detection'  # Must match Trainer's experiment_name
    exp_dir = os.path.join(checkpoint_dir, experiment_name)
    
    if not os.path.exists(exp_dir):
        logger.info(f"No experiment directory found at {exp_dir}")
        return train_params
        
    # Find the most recent RUN directory
    run_dirs = [d for d in os.listdir(exp_dir) if d.startswith('RUN_')]
    if not run_dirs:
        logger.info("No previous run directories found")
        return train_params
        
    # Sort by timestamp (newest first)
    run_dirs.sort(reverse=True)
    latest_run = os.path.join(exp_dir, run_dirs[0])
    
    # Look for checkpoints in the run directory
    checkpoint_path = os.path.join(latest_run, 'ckpt_latest.pth')
    best_checkpoint_path = os.path.join(latest_run, 'ckpt_best.pth')
    
    # Check for both latest and best checkpoints
    available_checkpoints = {
        'latest': checkpoint_path if os.path.exists(checkpoint_path) else None,
        'best': best_checkpoint_path if os.path.exists(best_checkpoint_path) else None
    }
    
    if not any(available_checkpoints.values()):
        logger.info(f"No checkpoints found in latest run directory: {latest_run}")
        return train_params
        
    # Verify checkpoint files
    valid_checkpoints = {}
    for name, path in available_checkpoints.items():
        if path and verify_checkpoint(path):
            valid_checkpoints[name] = path
            logger.info(f"Found valid {name} checkpoint: {path}")
    
    if not valid_checkpoints:
        logger.warning("Found checkpoint files but they are invalid - starting from scratch")
        return train_params
        
    # Prefer best checkpoint over latest if both exist
    chosen_checkpoint = valid_checkpoints.get('best', valid_checkpoints.get('latest'))
    
    # Update training parameters for resuming
    train_params.update({
        'resume': True,
        'resume_path': chosen_checkpoint,
        'resume_strict_load': False,  # Allow flexible loading
        'load_opt_params': True,      # Load optimizer state
        'load_ema_as_net': False,     # Don't load EMA weights as main weights
        'resume_epoch': True          # Continue from the last epoch
    })
    
    logger.info(f"Will resume training from checkpoint: {chosen_checkpoint}")
    
    # Verify checkpoint content
    try:
        checkpoint = torch.load(chosen_checkpoint, map_location='cpu')
        expected_keys = ['net', 'epoch', 'optimizer_state_dict']
        if not all(key in checkpoint for key in expected_keys):
            logger.warning("Checkpoint missing expected keys - may cause issues")
        else:
            logger.info(f"Resuming from epoch {checkpoint['epoch']}")
            logger.info(f"Found in run directory: {latest_run}")
    except Exception as e:
        logger.error(f"Error verifying checkpoint contents: {e}")
        logger.warning("Starting training from scratch")
        return {**train_params, 'resume': False}
    
    return train_params
