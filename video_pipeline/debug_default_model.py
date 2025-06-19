#!/usr/bin/env python3
"""
Debug script to test default YOLO model
"""

from ultralytics import YOLO
import cv2
import numpy as np

def debug_default_model():
    """Debug the default YOLO model detections"""
    
    # Load the default model
    model = YOLO('yolo11n.pt')
    print('🔍 Default Model Analysis:')
    print(f'   Classes: {len(model.names)} total classes')
    
    # Vehicle classes in COCO
    vehicle_classes = {2: 'car', 5: 'bus', 7: 'truck'}
    print(f'   Vehicle classes: {vehicle_classes}')
    
    # Load a frame from the video to test
    cap = cv2.VideoCapture('traffic_test_10s.mp4')
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Could not read video frame")
        return
        
    print(f'   Frame shape: {frame.shape}')
    
    # Test detection with different confidence thresholds
    for conf in [0.3, 0.5]:
        print(f'\n📊 Testing with confidence threshold: {conf}')
        results = model(frame, conf=conf, verbose=False)
        
        total_detections = 0
        vehicle_detections = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                total_detections = len(boxes)
                
                # Filter for vehicles only
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls)
                    if cls_id in vehicle_classes:
                        vehicle_detections += 1
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf_val = float(box.conf)
                        
                        # Calculate box size
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        
                        print(f'     Vehicle {vehicle_detections}:')
                        print(f'       Type: {vehicle_classes[cls_id]}')
                        print(f'       BBox: ({x1:.0f},{y1:.0f}) to ({x2:.0f},{y2:.0f})')
                        print(f'       Size: {width:.0f}x{height:.0f} (area: {area:.0f})')
                        print(f'       Confidence: {conf_val:.3f}')
                        
                        # Check if this is a reasonable detection
                        frame_area = frame.shape[0] * frame.shape[1]
                        coverage = area / frame_area * 100
                        if coverage > 50:
                            print(f'       ⚠️  Large detection: {coverage:.1f}% of frame')
                        else:
                            print(f'       ✅ Normal detection: {coverage:.1f}% of frame')
                        
        print(f'   Total detections: {total_detections}')
        print(f'   Vehicle detections: {vehicle_detections}')
    
    cap.release()

if __name__ == "__main__":
    debug_default_model() 