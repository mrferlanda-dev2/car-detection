#!/usr/bin/env python3
"""
Switch to using default YOLO model instead of custom model
"""

import os
import shutil

def switch_to_default_model():
    """Switch back to using default YOLO model"""
    
    print("🔄 Switching to default YOLO model...")
    
    # Backup the problematic custom model
    if os.path.exists('custom_best.pt'):
        backup_name = 'custom_best_backup.pt'
        shutil.move('custom_best.pt', backup_name)
        print(f"   📦 Backed up problematic model to: {backup_name}")
    
    # Update main.py to use default model
    main_py_path = 'main.py'
    if os.path.exists(main_py_path):
        with open(main_py_path, 'r') as f:
            content = f.read()
        
        # Replace custom model path with None to use default
        content = content.replace(
            "detector = YOLODetector('custom_best.pt')",
            "detector = YOLODetector()  # Use default YOLO11n model"
        )
        
        with open(main_py_path, 'w') as f:
            f.write(content)
        
        print(f"   ✅ Updated {main_py_path} to use default model")
    
    print(f"\n✅ Switch complete!")
    print(f"   Now using default YOLOv11n model (which detects multiple cars correctly)")
    print(f"   Your video pipeline should work properly now.")
    
    print(f"\n🎯 To get a better custom model:")
    print(f"   1. The fixed dataset is ready at: ../yolo_finetune/fixed_dataset")
    print(f"   2. Run: cd ../yolo_finetune && python train_optimized.py --data fixed_dataset/data.yaml --subset-ratio 0.2")
    print(f"   3. This will train a model with proper bounding boxes")

if __name__ == "__main__":
    switch_to_default_model() 