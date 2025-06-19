#!/usr/bin/env python3
"""
Fix Custom Model Training Data
Uses pre-trained YOLO to generate proper bounding boxes for VeRi images
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import cv2
from tqdm import tqdm
import random

class VeRiDatasetFixer:
    def __init__(self, veri_root: str, output_root: str):
        self.veri_root = Path(veri_root)
        self.output_root = Path(output_root)
        
        # Load pre-trained YOLO model to generate proper bounding boxes
        self.detector = YOLO('yolo11n.pt')
        
        # Vehicle classes in COCO that we want to map to 'Car'
        self.vehicle_classes = {2: 'car', 5: 'bus', 7: 'truck'}
        
        print(f"🔧 Fixing VeRi dataset with proper bounding boxes")
        print(f"   Source: {self.veri_root}")
        print(f"   Target: {self.output_root}")
    
    def create_directory_structure(self):
        """Create YOLO dataset directory structure"""
        directories = [
            'images/train',
            'images/val', 
            'images/test',
            'labels/train',
            'labels/val',
            'labels/test'
        ]
        
        for dir_path in directories:
            (self.output_root / dir_path).mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Created directory structure in {self.output_root}")
    
    def detect_vehicles_in_image(self, image_path: str):
        """Use pre-trained YOLO to detect vehicles and return bounding boxes"""
        try:
            # Run detection
            results = self.detector(image_path, conf=0.3, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls)
                        if cls_id in self.vehicle_classes:  # Only vehicle classes
                            # Get normalized coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            # Convert to image dimensions
                            img = cv2.imread(image_path)
                            if img is None:
                                continue
                                
                            h, w = img.shape[:2]
                            
                            # Convert to YOLO format (normalized center + width/height)
                            x_center = (x1 + x2) / 2 / w
                            y_center = (y1 + y2) / 2 / h
                            width = (x2 - x1) / w
                            height = (y2 - y1) / h
                            
                            # Only keep reasonable sized detections
                            if 0.01 < width < 0.8 and 0.01 < height < 0.8:
                                detections.append({
                                    'class_id': 0,  # All vehicles -> Car class
                                    'x_center': x_center,
                                    'y_center': y_center,
                                    'width': width,
                                    'height': height,
                                    'confidence': float(box.conf)
                                })
            
            return detections
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return []
    
    def process_image_split(self, split_name: str, image_dir: str):
        """Process images from a specific split"""
        print(f"\n🔄 Processing {split_name} split...")
        
        source_dir = self.veri_root / image_dir
        if not source_dir.exists():
            print(f"❌ Directory not found: {source_dir}")
            return 0
        
        # Get all image files
        image_files = list(source_dir.glob('*.jpg'))
        print(f"   Found {len(image_files)} images")
        
        processed = 0
        skipped = 0
        
        for image_file in tqdm(image_files, desc=f"Processing {split_name}"):
            # Detect vehicles in the image
            detections = self.detect_vehicles_in_image(str(image_file))
            
            if not detections:
                skipped += 1
                continue  # Skip images with no vehicle detections
            
            # Determine target split
            if split_name == 'train':
                target_split = 'train' if random.random() < 0.8 else 'val'
            else:
                target_split = 'test'
            
            # Copy image
            dst_image_path = self.output_root / 'images' / target_split / image_file.name
            shutil.copy2(image_file, dst_image_path)
            
            # Create label file
            label_name = image_file.name.replace('.jpg', '.txt')
            label_path = self.output_root / 'labels' / target_split / label_name
            
            with open(label_path, 'w') as f:
                for detection in detections:
                    f.write(f"{detection['class_id']} {detection['x_center']:.6f} "
                           f"{detection['y_center']:.6f} {detection['width']:.6f} "
                           f"{detection['height']:.6f}\n")
            
            processed += 1
        
        print(f"   ✅ Processed: {processed} images")
        print(f"   ⏭️  Skipped: {skipped} images (no vehicles detected)")
        return processed
    
    def create_yaml_config(self):
        """Create YOLO dataset configuration file"""
        yaml_content = f"""# Fixed VeRi Dataset for YOLO Car Detection
path: {self.output_root.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
nc: 1  # number of classes
names: ['Car']  # class names
"""
        
        yaml_path = self.output_root / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"✅ Created data.yaml configuration")
    
    def create_statistics(self):
        """Create dataset statistics"""
        stats = {}
        
        for split in ['train', 'val', 'test']:
            image_dir = self.output_root / 'images' / split
            label_dir = self.output_root / 'labels' / split
            
            if image_dir.exists():
                num_images = len(list(image_dir.glob('*.jpg')))
                num_labels = len(list(label_dir.glob('*.txt')))
                
                # Count total annotations
                total_annotations = 0
                for label_file in label_dir.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        total_annotations += len(f.readlines())
                
                stats[split] = {
                    'images': num_images,
                    'labels': num_labels,
                    'annotations': total_annotations
                }
        
        # Write statistics
        stats_file = self.output_root / 'dataset_stats.txt'
        with open(stats_file, 'w') as f:
            f.write("Fixed VeRi Dataset Statistics\n")
            f.write("=" * 30 + "\n\n")
            
            total_images = 0
            total_annotations = 0
            
            for split, data in stats.items():
                f.write(f"{split.upper()} Split:\n")
                f.write(f"  Images: {data['images']}\n")
                f.write(f"  Labels: {data['labels']}\n")
                f.write(f"  Annotations: {data['annotations']}\n")
                f.write(f"  Avg annotations per image: {data['annotations']/data['images']:.2f}\n\n")
                
                total_images += data['images']
                total_annotations += data['annotations']
            
            f.write(f"TOTAL:\n")
            f.write(f"  Images: {total_images}\n")
            f.write(f"  Annotations: {total_annotations}\n")
            f.write(f"  Average annotations per image: {total_annotations/total_images:.2f}\n")
        
        print(f"✅ Dataset statistics saved to {stats_file}")
        
        # Print summary
        print(f"\n📊 Dataset Summary:")
        for split, data in stats.items():
            print(f"   {split}: {data['images']} images, {data['annotations']} cars")
    
    def fix_dataset(self):
        """Main method to fix the VeRi dataset"""
        print(f"🔧 Starting VeRi dataset fix...")
        
        # Create directory structure
        self.create_directory_structure()
        
        # Process splits
        total_processed = 0
        total_processed += self.process_image_split('train', 'VeRi/image_train')
        total_processed += self.process_image_split('test', 'VeRi/image_test')
        
        if total_processed == 0:
            print("❌ No images processed! Check your VeRi dataset path.")
            return False
        
        # Create configuration and statistics
        self.create_yaml_config()
        self.create_statistics()
        
        print(f"\n✅ Dataset fixing complete!")
        print(f"   Total processed: {total_processed} images")
        print(f"   Output directory: {self.output_root}")
        
        return True

def main():
    # Default paths
    veri_root = "../train/VeRi"  # Adjust this path as needed
    output_root = "../yolo_finetune/fixed_dataset"
    
    # Create fixer and process
    fixer = VeRiDatasetFixer(veri_root, output_root)
    success = fixer.fix_dataset()
    
    if success:
        print(f"\n🎯 Next steps:")
        print(f"   1. Review the fixed dataset in: {output_root}")
        print(f"   2. Retrain your model using: ../yolo_finetune/train_optimized.py")
        print(f"   3. Use --data {output_root}/data.yaml for training")

if __name__ == "__main__":
    main() 