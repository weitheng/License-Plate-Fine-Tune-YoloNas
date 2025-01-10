import os
import glob

def update_label_files(labels_dir):
    """Update label files to use class index 80 for license plates"""
    label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    print(f"Found {len(label_files)} label files in {labels_dir}")
    
    updated_count = 0
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        updated_lines = []
        file_updated = False
        for line in lines:
            parts = line.strip().split()
            if parts[0] == '0':  # If it's a license plate (currently class 0)
                parts[0] = '80'  # Update to class 80
                file_updated = True
            updated_lines.append(' '.join(parts) + '\n')
        
        if file_updated:
            with open(label_file, 'w') as f:
                f.writelines(updated_lines)
            updated_count += 1
    
    print(f"Updated {updated_count} files in {labels_dir}")

if __name__ == "__main__":
    # Update both train and val labels
    update_label_files('labels/train')
    update_label_files('labels/val') 