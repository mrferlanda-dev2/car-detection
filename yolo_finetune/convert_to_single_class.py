#!/usr/bin/env python3
"""
Convert multi-class YOLO dataset to single class "car"
"""

import os
import yaml
import shutil

def convert_to_single_class(dataset_path: str, output_path: str = None):
    """
    Convert multi-class dataset to single class "car"
    
    Args:
        dataset_path: Path to the original dataset
        output_path: Path for the converted dataset (if None, overwrites original)
    """
    
    if output_path is None:
        output_path = dataset_path + "_single_class"
    
    print(f"Converting dataset from: {dataset_path}")
    print(f"Output directory: {output_path}")
    
    # Create output directory structure
    output_images_dir = os.path.join(output_path, "images", "train")
    output_labels_dir = os.path.join(output_path, "labels", "train")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Copy images
    source_images_dir = os.path.join(dataset_path, "images", "train")
    print(f"Copying images from {source_images_dir} to {output_images_dir}")
    
    for img_file in os.listdir(source_images_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            src_path = os.path.join(source_images_dir, img_file)
            dst_path = os.path.join(output_images_dir, img_file)
            shutil.copy2(src_path, dst_path)
    
    # Process labels
    source_labels_dir = os.path.join(dataset_path, "labels", "train")
    print(f"Processing labels from {source_labels_dir}")
    
    total_annotations = 0
    converted_annotations = 0
    
    for label_file in os.listdir(source_labels_dir):
        if label_file.endswith('.txt'):
            src_label_path = os.path.join(source_labels_dir, label_file)
            dst_label_path = os.path.join(output_labels_dir, label_file)
            
            with open(src_label_path, 'r') as f:
                lines = f.readlines()
            
            converted_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # Convert any vehicle class (0-7) to class 0 (car)
                        if 0 <= class_id <= 7:
                            parts[0] = '0'  # Set to class 0 (car)
                            converted_lines.append(' '.join(parts))
                            converted_annotations += 1
                        total_annotations += 1
            
            # Write converted labels
            with open(dst_label_path, 'w') as f:
                f.write('\n'.join(converted_lines))
    
    # Copy train.txt
    train_txt_src = os.path.join(dataset_path, "train.txt")
    train_txt_dst = os.path.join(output_path, "train.txt")
    if os.path.exists(train_txt_src):
        shutil.copy2(train_txt_src, train_txt_dst)
    
    # Create new data.yaml
    data_yaml = {
        'path': './',
        'train': 'train.txt',
        'names': {0: 'car'}
    }
    
    yaml_path = os.path.join(output_path, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f"Conversion completed!")
    print(f"   Total annotations processed: {total_annotations}")
    print(f"   Annotations converted to 'car': {converted_annotations}")
    print(f"   Output directory: {output_path}")
    print(f"   Classes: car (class_id: 0)")

if __name__ == "__main__":
    dataset_path = "yolo_finetune/cvat_dataset_updated"
    convert_to_single_class(dataset_path) 