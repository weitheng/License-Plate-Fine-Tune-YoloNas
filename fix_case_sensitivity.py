import os
import logging
import coloredlogs
from pathlib import Path
import shutil

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

def fix_case_sensitivity(combined_dir: str) -> None:
    """
    Fix case sensitivity issues in the combined dataset directory
    by converting all filenames to lowercase.
    
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
                
                # Get all files in directory
                files = os.listdir(directory)
                
                for filename in files:
                    old_path = os.path.join(directory, filename)
                    new_filename = filename.lower()  # Convert to lowercase
                    new_path = os.path.join(directory, new_filename)
                    
                    # Skip if filename is already lowercase
                    if filename == new_filename:
                        continue
                        
                    try:
                        # If the destination file already exists, handle it
                        if os.path.exists(new_path) and new_path != old_path:
                            # Compare file sizes to see if they're the same file
                            if os.path.getsize(old_path) == os.path.getsize(new_path):
                                os.remove(old_path)  # Remove the uppercase version
                                logger.info(f"Removed duplicate file: {filename}")
                                continue
                            else:
                                # If sizes differ, keep both but rename the new one
                                base, ext = os.path.splitext(new_filename)
                                new_path = os.path.join(directory, f"{base}_2{ext}")
                        
                        # Rename the file
                        os.rename(old_path, new_path)
                        files_processed += 1
                        logger.info(f"Renamed: {filename} -> {os.path.basename(new_path)}")
                        
                    except Exception as e:
                        logger.error(f"Error processing {filename}: {str(e)}")
                        continue

                logger.info(f"Processed {files_processed} files in {data_type}/{split}")

        # Verify the changes
        for split in ['train', 'val']:
            for data_type in ['images', 'labels']:
                directory = os.path.join(combined_dir, data_type, split)
                if os.path.exists(directory):
                    # Check if any uppercase letters remain in filenames
                    uppercase_files = [f for f in os.listdir(directory) if any(c.isupper() for c in f)]
                    if uppercase_files:
                        logger.warning(f"Found {len(uppercase_files)} files with uppercase letters in {data_type}/{split}")
                        for f in uppercase_files[:5]:  # Show first 5 examples
                            logger.warning(f"Example: {f}")
                    else:
                        logger.info(f"✓ All files in {data_type}/{split} are lowercase")

        logger.info("✓ Case sensitivity fix completed")

    except Exception as e:
        logger.error(f"Error fixing case sensitivity: {str(e)}")
        raise

if __name__ == "__main__":
    # Get the combined directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    combined_dir = os.path.join(current_dir, 'data', 'combined')
    
    # Fix case sensitivity issues
    fix_case_sensitivity(combined_dir) 