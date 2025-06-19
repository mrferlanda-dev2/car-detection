#!/usr/bin/env python3
"""
Augment LCGC images to match the largest class count
"""

import os
import shutil
import glob
import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm
import random

def count_files_in_directories(base_path):
    """Count files in each subdirectory"""
    counts = {}
    for subdir in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir)
        if os.path.isdir(subdir_path):
            file_count = len([f for f in os.listdir(subdir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            counts[subdir] = file_count
    return counts

def create_augmentation_pipeline():
    """Create augmentation pipeline using albumentations"""
    return A.Compose([
        A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        ], p=0.8),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.3),
            A.MotionBlur(blur_limit=3, p=0.3),
        ], p=0.3),
        A.OneOf([
            A.RandomShadow(p=0.3),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.3),
        ], p=0.2),
    ])

def augment_lcgc_split(split_path, target_count, current_count):
    """Augment LCGC images in a specific split to reach target count"""
    if current_count >= target_count:
        print(f"  Already has {current_count} images, no augmentation needed")
        return
    
    print(f"  Augmenting from {current_count} to {target_count} images (+{target_count - current_count} new)")
    
    # Get all image files
    images = glob.glob(os.path.join(split_path, "*.[jJ][pP][gG]")) + \
             glob.glob(os.path.join(split_path, "*.[jJ][pP][eE][gG]")) + \
             glob.glob(os.path.join(split_path, "*.[pP][nN][gG]"))
    
    if not images:
        print(f"  No images found in {split_path}")
        return
    
    # Create augmentation pipeline
    augmentation = create_augmentation_pipeline()
    
    # Generate augmented images
    needed_augmentations = target_count - current_count
    
    for i in tqdm(range(needed_augmentations), desc=f"Augmenting {os.path.basename(split_path)}"):
        # Select random source image
        source_img_path = random.choice(images)
        img = cv2.imread(source_img_path)
        
        if img is None:
            continue
            
        # Apply augmentation
        augmented = augmentation(image=img)
        augmented_img = augmented['image']
        
        # Save augmented image with unique name
        base_name = os.path.splitext(os.path.basename(source_img_path))[0]
        aug_name = f"{base_name}_aug_{i:04d}.jpg"
        aug_path = os.path.join(split_path, aug_name)
        
        # Ensure unique filename
        counter = 1
        while os.path.exists(aug_path):
            aug_name = f"{base_name}_aug_{i:04d}_{counter:02d}.jpg"
            aug_path = os.path.join(split_path, aug_name)
            counter += 1
        
        cv2.imwrite(aug_path, augmented_img)

def main():
    dataset_path = "dataset"
    
    print("=== Current Dataset Analysis ===")
    train_counts = count_files_in_directories(os.path.join(dataset_path, "train"))
    for class_name, count in sorted(train_counts.items()):
        print(f"{class_name}: {count} images")
    
    # Find the largest class count for target augmentation
    max_count = max(train_counts.values())
    lcgc_count = train_counts.get('LCGC', 0)
    
    print(f"\nLargest class has {max_count} images")
    print(f"LCGC currently has {lcgc_count} images")
    
    if lcgc_count >= max_count:
        print("LCGC already has enough images, no augmentation needed")
        return
    
    print(f"\n=== Augmenting LCGC from {lcgc_count} to {max_count} images ===")
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(dataset_path, split, 'LCGC')
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} not found, skipping")
            continue
        
        current_split_count = len([f for f in os.listdir(split_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # Scale target proportionally
        if split == 'train':
            target_split_count = max_count
        elif split == 'val':
            target_split_count = int(max_count * 0.2)  # 20% of train
        else:  # test
            target_split_count = int(max_count * 0.1)  # 10% of train
        
        print(f"\nProcessing {split} split:")
        print(f"  Current: {current_split_count} images")
        print(f"  Target: {target_split_count} images")
        
        augment_lcgc_split(split_path, target_split_count, current_split_count)
    
    print("\n=== Final LCGC Analysis ===")
    final_counts = count_files_in_directories(os.path.join(dataset_path, "train"))
    print(f"LCGC: {final_counts.get('LCGC', 0)} images")

if __name__ == "__main__":
    main() 