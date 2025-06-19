#!/usr/bin/env python3
"""
Setup script to use your custom trained single-class YOLO model
"""

import shutil
import os
from pathlib import Path

def setup_custom_model():
    """Copy the trained model to video pipeline directory"""
    
    # Find the most recent training run
    yolo_finetune_dir = Path("../yolo_finetune")
    training_dirs = list(yolo_finetune_dir.glob("**/weights/best.pt"))
    
    if not training_dirs:
        print("❌ No trained models found!")
        print("Please train a model first using the yolo_finetune scripts")
        return False
    
    # Get the most recent model
    latest_model = max(training_dirs, key=lambda p: p.stat().st_mtime)
    
    # Copy to video pipeline directory
    dest_path = Path("custom_best.pt")
    shutil.copy2(latest_model, dest_path)
    
    print(f"✅ Copied trained model:")
    print(f"   From: {latest_model}")
    print(f"   To: {dest_path}")
    print(f"   Model size: {dest_path.stat().st_size / 1e6:.1f}MB")
    
    return True

def test_model():
    """Test the custom model with a simple detection"""
    try:
        from detector import YOLODetector
        import cv2
        import numpy as np
        
        # Initialize detector with custom model
        detector = YOLODetector(model_path="custom_best.pt")
        
        # Create a test image (dummy)
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Test detection
        detections = detector.detect(test_image, conf_threshold=0.3)
        
        print(f"\n🧪 Model test completed:")
        print(f"   Custom model loaded: {detector.is_custom_single_class}")
        print(f"   Test detections: {len(detections)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    print("🚀 Setting up custom YOLO model for video pipeline...")
    
    if setup_custom_model():
        print("\n📝 Model setup complete!")
        print("\n🎯 Usage examples:")
        print("   # Process video with custom model:")
        print("   python main.py input_video.mp4 --yolo-model custom_best.pt")
        print("   ")
        print("   # Test with traffic video:")
        print("   python main.py traffic_test_10s.mp4 --yolo-model custom_best.pt --display")
        
        # Test the model
        test_model()
        
    else:
        print("❌ Setup failed!")

if __name__ == "__main__":
    main() 