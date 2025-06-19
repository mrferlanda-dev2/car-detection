#!/usr/bin/env python3
"""
Test custom model on fixed dataset
"""

import cv2
import glob
import os
from ultralytics import YOLO

def test_custom_model_on_fixed_dataset():
    """Test the custom model on images from the fixed dataset"""
    
    # Load the custom model
    model = YOLO('custom_best.pt')
    print(f"🧪 Testing custom model on fixed dataset")
    
    # Check if fixed dataset exists
    fixed_dataset_path = '../yolo_finetune/fixed_dataset'
    if not os.path.exists(fixed_dataset_path):
        print(f"❌ Fixed dataset not found at {fixed_dataset_path}")
        return
    
    # Get test images from fixed dataset
    test_images = glob.glob(f'{fixed_dataset_path}/images/test/*.jpg')
    if not test_images:
        print(f"❌ No test images found in fixed dataset")
        return
    
    print(f"📁 Found {len(test_images)} test images in fixed dataset")
    
    # Test on first 5 images
    for i, img_path in enumerate(test_images[:5]):
        print(f"\n🖼️  Testing image {i+1}: {os.path.basename(img_path)}")
        
        # Run detection
        results = model(img_path, conf=0.3, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                print(f"   Found {len(boxes)} detections")
                for j, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf_val = float(box.conf)
                    width = x2 - x1
                    height = y2 - y1
                    
                    print(f"     Detection {j}: ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
                    print(f"       Size: {width:.0f}x{height:.0f}, Conf: {conf_val:.3f}")
            else:
                print(f"   No detections found")
    
    # Also test on video frame for comparison
    print(f"\n🎥 Testing on video frame...")
    cap = cv2.VideoCapture('traffic_test_10s.mp4')
    ret, frame = cap.read()
    if ret:
        # Save frame temporarily
        cv2.imwrite('/tmp/video_frame.jpg', frame)
        
        # Test on video frame
        results = model('/tmp/video_frame.jpg', conf=0.3, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                print(f"   Video frame: {len(boxes)} detections")
                for j, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf_val = float(box.conf)
                    width = x2 - x1
                    height = y2 - y1
                    
                    print(f"     Detection {j}: ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
                    print(f"       Size: {width:.0f}x{height:.0f}, Conf: {conf_val:.3f}")
        
        # Clean up
        os.remove('/tmp/video_frame.jpg')
    
    cap.release()

if __name__ == "__main__":
    test_custom_model_on_fixed_dataset() 