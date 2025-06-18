#!/usr/bin/env python3
"""
Real-time Webcam Demo for Vehicle Detection Pipeline
"""

import cv2
import time
from video_processor import VideoProcessor

def demo_webcam():
    """Demo using webcam feed"""
    
    print("🚀 Starting Webcam Demo...")
    
    # Initialize pipeline
    try:
        processor = VideoProcessor(
            yolo_model_path="yolo11n.pt",
            classifier_model_path="improved_vehicle_classifier.pth",
            device='auto'
        )
        processor.conf_threshold = 0.5
        
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    print("✅ Webcam opened successfully")
    print("📹 Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    fps_counter = 0
    fps_start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        start_time = time.time()
        
        # Process frame
        annotated_frame, results = processor.process_frame(frame)
        
        processing_time = time.time() - start_time
        frame_count += 1
        fps_counter += 1
        
        # Calculate FPS every second
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
            
            # Display FPS and processing time on frame
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Processing: {processing_time:.3f}s", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display detections count
        cv2.putText(annotated_frame, f"Detections: {len(results)}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Vehicle Detection - Webcam Demo', annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            filename = f"webcam_capture_{frame_count}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"💾 Frame saved as {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam demo ended")

if __name__ == "__main__":
    demo_webcam() 