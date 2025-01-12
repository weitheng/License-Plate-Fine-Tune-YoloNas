import os
import shutil
from pathlib import Path
import logging
import coloredlogs

# Setup custom success level
SUCCESS_LEVEL = 25  # Between INFO and WARNING
logging.addLevelName(SUCCESS_LEVEL, 'SUCCESS')

def success(self, message, *args, **kwargs):
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)

# Add success method to Logger class
logging.Logger.success = success

# Setup logging
logger = logging.getLogger(__name__)
coloredlogs.install(
    level='INFO',
    logger=logger,
    fmt='%(asctime)s - %(levelname)s - %(message)s',
    level_styles={
        'info': {'color': 'white'},
        'success': {'color': 'green', 'bold': True},
        'warning': {'color': 'yellow'},
        'error': {'color': 'red', 'bold': True}
    }
)

def remove_lp_prefix(combined_dir: str) -> None:
    """
    Remove 'lp_' prefix from all files in the combined dataset directory
    for both images and labels in train and val splits.
    
    Args:
        combined_dir (str): Path to the combined dataset directory
    """
    try:
        combined_dir = os.path.abspath(combined_dir)
        if not os.path.exists(combined_dir):
            raise FileNotFoundError(f"Combined directory not found: {combined_dir}")

        # Process both train and val splits
        for split in ['train', 'val']:
            # Process both images and labels
            for data_type in ['images', 'labels']:
                directory = os.path.join(combined_dir, data_type, split)
                if not os.path.exists(directory):
                    logger.warning(f"Directory not found: {directory}")
                    continue

                logger.info(f"Processing {data_type} in {split} split...")
                files_processed = 0
                
                # Get all files with 'lp_' prefix
                files = [f for f in os.listdir(directory) if f.startswith('lp_')]
                
                for filename in files:
                    old_path = os.path.join(directory, filename)
                    new_filename = filename[3:]  # Remove 'lp_' prefix
                    new_path = os.path.join(directory, new_filename)
                    
                    try:
                        # If the destination file already exists, remove it
                        if os.path.exists(new_path):
                            os.remove(new_path)
                            
                        # Rename the file
                        os.rename(old_path, new_path)
                        files_processed += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing {filename}: {str(e)}")
                        continue

                logger.success(f"✓ Processed {files_processed} files in {data_type}/{split}")

        logger.success("✓ Successfully removed 'lp_' prefix from all files")

    except Exception as e:
        logger.error(f"Error removing 'lp_' prefix: {str(e)}")
        raise

if __name__ == "__main__":
    # Get the combined directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    combined_dir = os.path.join(current_dir, 'data', 'combined')
    
    # Remove 'lp_' prefix from files
    remove_lp_prefix(combined_dir) 