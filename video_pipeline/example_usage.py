#!/usr/bin/env python3
"""
Example Usage Script for Vehicle Detection Pipeline
"""

import os
from video_processor import VideoProcessor

def example_basic_video_processing():
    """Example 1: Basic video processing"""
    print("📹 Example 1: Basic Video Processing")
    print("="*50)
    
    # Initialize pipeline
    processor = VideoProcessor(
        yolo_model_path="yolo11n.pt",
        classifier_model_path="improved_vehicle_classifier.pth",
        device='auto'  # Automatically choose GPU if available
    )
    
    # Process video (replace with your video path)
    video_path = "sample_video.mp4"  # Replace with actual video
    if os.path.exists(video_path):
        summary = processor.process_video(
            input_path=video_path,
            output_path="output_basic.mp4",
            save_results=True,
            display=False
        )
        print(f"✅ Processed {summary['total_frames']} frames")
        print(f"   Found {summary['total_detections']} vehicles")
        print(f"   Vehicle types: {summary['vehicle_counts']}")
    else:
        print(f"⚠️  Video file not found: {video_path}")

def example_high_accuracy_processing():
    """Example 2: High accuracy processing with lower threshold"""
    print("\n📹 Example 2: High Accuracy Processing")
    print("="*50)
    
    processor = VideoProcessor(
        yolo_model_path="yolo11n.pt",
        classifier_model_path="improved_vehicle_classifier.pth",
        device='auto'
    )
    
    # Lower confidence threshold for more detections
    processor.conf_threshold = 0.3
    processor.min_detection_size = 30  # Detect smaller vehicles
    
    video_path = "sample_video.mp4"
    if os.path.exists(video_path):
        summary = processor.process_video(
            input_path=video_path,
            output_path="output_high_accuracy.mp4",
            save_results=True
        )
        print(f"✅ High accuracy mode: {summary['total_detections']} detections")

def example_fast_processing():
    """Example 3: Fast processing with higher threshold"""
    print("\n📹 Example 3: Fast Processing")
    print("="*50)
    
    processor = VideoProcessor(
        yolo_model_path="yolo11n.pt",
        classifier_model_path="improved_vehicle_classifier.pth",
        device='auto'
    )
    
    # Higher confidence threshold for faster processing
    processor.conf_threshold = 0.7
    processor.min_detection_size = 100  # Only large vehicles
    
    video_path = "sample_video.mp4"
    if os.path.exists(video_path):
        summary = processor.process_video(
            input_path=video_path,
            output_path="output_fast.mp4",
            save_results=False  # Skip detailed results for speed
        )
        print(f"✅ Fast mode: {summary['avg_processing_time']:.3f}s per frame")

def example_single_frame_processing():
    """Example 4: Process single frames from video"""
    print("\n📹 Example 4: Single Frame Processing")
    print("="*50)
    
    import cv2
    
    processor = VideoProcessor(
        yolo_model_path="yolo11n.pt",
        classifier_model_path="improved_vehicle_classifier.pth",
        device='auto'
    )
    
    # Process specific frames
    video_path = "sample_video.mp4"
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        
        # Jump to frame 100
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ret, frame = cap.read()
        
        if ret:
            annotated_frame, results = processor.process_frame(frame)
            
            # Save result
            cv2.imwrite("frame_100_result.jpg", annotated_frame)
            print(f"✅ Frame 100 processed: {len(results)} vehicles detected")
            
            for i, result in enumerate(results):
                print(f"   Vehicle {i+1}: {result['vehicle_class']} "
                      f"({result['class_confidence']:.2f})")
        
        cap.release()

def example_batch_processing():
    """Example 5: Batch process multiple videos"""
    print("\n📹 Example 5: Batch Processing")
    print("="*50)
    
    # List of videos to process
    video_files = [
        "video1.mp4",
        "video2.mp4", 
        "video3.mp4"
    ]
    
    processor = VideoProcessor(
        yolo_model_path="yolo11n.pt",
        classifier_model_path="improved_vehicle_classifier.pth",
        device='auto'
    )
    
    batch_results = []
    
    for i, video_path in enumerate(video_files):
        if os.path.exists(video_path):
            print(f"Processing video {i+1}/{len(video_files)}: {video_path}")
            
            summary = processor.process_video(
                input_path=video_path,
                output_path=f"batch_output_{i+1}.mp4",
                save_results=True
            )
            
            batch_results.append(summary)
            print(f"   ✅ {summary['total_detections']} vehicles detected")
        else:
            print(f"   ⚠️  Video not found: {video_path}")
    
    # Summary of batch processing
    if batch_results:
        total_detections = sum(r['total_detections'] for r in batch_results)
        avg_processing_time = sum(r['avg_processing_time'] for r in batch_results) / len(batch_results)
        
        print(f"\n📊 Batch Summary:")
        print(f"   Total videos processed: {len(batch_results)}")
        print(f"   Total vehicle detections: {total_detections}")
        print(f"   Average processing time: {avg_processing_time:.3f}s/frame")

def main():
    """Run all examples"""
    print("🚀 Vehicle Detection Pipeline - Usage Examples")
    print("="*60)
    
    # Check if we have the required models
    if not os.path.exists("improved_vehicle_classifier.pth"):
        print("❌ improved_vehicle_classifier.pth not found!")
        print("   Please ensure the model is in the current directory.")
        return
    
    if not os.path.exists("yolo11n.pt"):
        print("⚠️  yolo11n.pt not found. Will use PyTorch Hub fallback.")
    
    # Run examples (comment out examples you don't want to run)
    
    # Basic video processing
    # example_basic_video_processing()
    
    # High accuracy processing
    # example_high_accuracy_processing()
    
    # Fast processing
    # example_fast_processing()
    
    # Single frame processing
    # example_single_frame_processing()
    
    # Batch processing
    # example_batch_processing()
    
    print("\n💡 Tips:")
    print("   - Uncomment the examples you want to run")
    print("   - Replace 'sample_video.mp4' with your actual video file")
    print("   - Adjust confidence thresholds based on your needs")
    print("   - Use GPU (device='cuda') for faster processing")
    print("   - Lower conf_threshold = more detections, slower processing")
    print("   - Higher conf_threshold = fewer detections, faster processing")

if __name__ == "__main__":
    main() 