#!/usr/bin/env python3
"""
Split existing dataset into train and val directories
"""

import os
import shutil
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

def split_dataset(dataset_dir, train_ratio=0.8, random_seed=42):
    """Split dataset into train and val directories"""
    
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"Error: Dataset directory '{dataset_dir}' not found!")
        return False
    
    # Get all class directories
    classes = [d for d in dataset_path.iterdir() if d.is_dir()]
    if not classes:
        print("Error: No class directories found in dataset!")
        return False
    
    print(f"Found {len(classes)} classes: {[c.name for c in classes]}")
    
    # Create train and val directories
    train_dir = dataset_path / 'train'
    val_dir = dataset_path / 'val'
    
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Set random seed for reproducible splits
    random.seed(random_seed)
    
    total_train = 0
    total_val = 0
    
    for class_dir in classes:
        class_name = class_dir.name
        print(f"\nProcessing {class_name}...")
        
        # Get all images in this class
        images = [f for f in class_dir.iterdir() 
                 if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
        
        if len(images) == 0:
            print(f"No images found in {class_name}")
            continue
        
        # Split images
        train_images, val_images = train_test_split(
            images, 
            test_size=1-train_ratio, 
            random_state=random_seed
        )
        
        # Create class directories in train and val
        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        train_class_dir.mkdir(exist_ok=True)
        val_class_dir.mkdir(exist_ok=True)
        
        # Copy training images
        for img_path in train_images:
            dst_path = train_class_dir / img_path.name
            shutil.copy2(img_path, dst_path)
        
        # Copy validation images
        for img_path in val_images:
            dst_path = val_class_dir / img_path.name
            shutil.copy2(img_path, dst_path)
        
        print(f"{class_name}: {len(train_images)} train, {len(val_images)} val")
        total_train += len(train_images)
        total_val += len(val_images)
    
    print(f"\n📊 Dataset Split Summary:")
    print(f"Total training images: {total_train}")
    print(f"Total validation images: {total_val}")
    print(f"Split ratio: {train_ratio:.1%} train, {1-train_ratio:.1%} val")
    
    return True

def main():
    print("Dataset Splitter")
    print("=" * 30)
    
    dataset_dir = "dataset"
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found!")
        return
    
    # Check if train/val already exist
    if os.path.exists(os.path.join(dataset_dir, 'train')) or os.path.exists(os.path.join(dataset_dir, 'val')):
        print("⚠️  Warning: train or val directories already exist!")
        response = input("Do you want to overwrite them? (y/N): ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return
    
    # Split the dataset
    success = split_dataset(dataset_dir, train_ratio=0.8)
    
    if success:
        print(f"\nDataset split completed!")
        print(f"Train: {dataset_dir}/train/")
        print(f"Val: {dataset_dir}/val/")
    else:
        print("Dataset split failed!")

if __name__ == "__main__":
    main() 