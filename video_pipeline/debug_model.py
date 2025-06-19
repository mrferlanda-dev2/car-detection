#!/usr/bin/env python3
"""
Debug script to analyze custom model detections
"""

from ultralytics import YOLO
import cv2
import numpy as np

def debug_model():
    """Debug the custom model detections"""
    
    # Load the custom model
    model = YOLO('custom_best.pt')
    print('🔍 Model Analysis:')
    print(f'   Classes: {model.names}')
    print(f'   Number of classes: {len(model.names)}')
    
    # Load a frame from the video to test
    cap = cv2.VideoCapture('traffic_test_10s.mp4')
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Could not read video frame")
        return
        
    print(f'   Frame shape: {frame.shape}')
    
    # Test detection with different confidence thresholds
    for conf in [0.1, 0.3, 0.5]:
        print(f'\n📊 Testing with confidence threshold: {conf}')
        results = model(frame, conf=conf, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                print(f'   Found {len(boxes)} detections')
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf_val = float(box.conf)
                    cls_id = int(box.cls)
                    
                    # Calculate box size
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    print(f'     Detection {i}:')
                    print(f'       BBox: ({x1:.0f},{y1:.0f}) to ({x2:.0f},{y2:.0f})')
                    print(f'       Size: {width:.0f}x{height:.0f} (area: {area:.0f})')
                    print(f'       Confidence: {conf_val:.3f}')
                    print(f'       Class: {cls_id} ({model.names.get(cls_id, "unknown")})')
                    
                    # Check if this is an abnormally large detection
                    frame_area = frame.shape[0] * frame.shape[1]
                    if area > frame_area * 0.5:  # If detection covers >50% of frame
                        print(f'       ⚠️  WARNING: Very large detection! ({area/frame_area*100:.1f}% of frame)')
                        
            else:
                print(f'   No detections found')
    
    cap.release()
    
    # Also test with original YOLO model for comparison
    print(f'\n🔄 Comparing with default YOLOv11n...')
    default_model = YOLO('yolo11n.pt')
    results = default_model(frame, conf=0.3, verbose=False)
    
    vehicle_classes = {2: 'car', 5: 'bus', 7: 'truck'}  # COCO classes
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            vehicle_detections = []
            for box in boxes:
                cls_id = int(box.cls)
                if cls_id in vehicle_classes:
                    vehicle_detections.append(box)
            
            print(f'   Default model found {len(vehicle_detections)} vehicles')
            for i, box in enumerate(vehicle_detections[:3]):  # Show first 3
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf_val = float(box.conf)
                cls_id = int(box.cls)
                width = x2 - x1
                height = y2 - y1
                print(f'     Vehicle {i}: {width:.0f}x{height:.0f}, conf={conf_val:.3f}, class={vehicle_classes[cls_id]}')

if __name__ == "__main__":
    debug_model() 