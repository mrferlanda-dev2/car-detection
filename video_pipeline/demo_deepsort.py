#!/usr/bin/env python3
"""
Demo script for DeepSORT Vehicle Tracking
"""

import cv2
import numpy as np
import time
import argparse
import os
from video_processor import VideoProcessor

def main():
    parser = argparse.ArgumentParser(description='DeepSORT Vehicle Tracking Demo')
    parser.add_argument('input', help='Input video path')
    parser.add_argument('-o', '--output', default='deepsort_demo_output.mp4',
                       help='Output video path')
    parser.add_argument('--yolo-model', default='yolo11n.pt',
                       help='Path to YOLO model')
    parser.add_argument('--classifier-model', 
                       default='veri_type_classifier_efficientnetv2s_20250618_103353.pth',
                       help='Path to vehicle classifier')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                       help='Detection confidence threshold')
    parser.add_argument('--simple-tracker', action='store_true',
                       help='Use simple tracker instead of DeepSORT for comparison')
    
    args = parser.parse_args()
    
    print("🚗 DeepSORT Vehicle Tracking Demo")
    print("=" * 50)
    
    if not os.path.exists(args.input):
        print(f"❌ Input video not found: {args.input}")
        return
    
    # Initialize video processor
    try:
        print(f"Initializing pipeline with {'DeepSORT' if not args.simple_tracker else 'Simple'} tracker...")
        
        processor = VideoProcessor(
            yolo_model_path=args.yolo_model if os.path.exists(args.yolo_model) else None,
            classifier_model_path=args.classifier_model,
            device=args.device,
            use_deepsort=not args.simple_tracker
        )
        
        processor.conf_threshold = args.conf_threshold
        
        print(f"✅ Pipeline initialized successfully!")
        print(f"   Tracker: {'DeepSORT' if not args.simple_tracker else 'Simple Centroid'}")
        print(f"   Confidence threshold: {args.conf_threshold}")
        print(f"   Device: {args.device}")
        
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return
    
    # Process video
    try:
        start_time = time.time()
        
        summary = processor.process_video(
            input_path=args.input,
            output_path=args.output,
            save_results=True,
            display=False
        )
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 Processing completed successfully!")
        print(f"📊 Summary:")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Output saved to: {args.output}")
        
        # Print tracking stats if available (DeepSORT only)
        if not args.simple_tracker and hasattr(processor.tracker, 'get_tracking_stats'):
            stats = processor.tracker.get_tracking_stats()
            print(f"   Tracking stats:")
            print(f"     Total unique tracks: {stats.get('total_tracks', 'N/A')}")
            print(f"     Active tracks: {stats.get('active_tracks', 'N/A')}")
        
        if summary:
            print(f"   Video stats:")
            print(f"     Frames processed: {summary.get('total_frames', 'N/A')}")
            print(f"     Total detections: {summary.get('total_detections', 'N/A')}")
            print(f"     Average FPS: {summary.get('avg_fps', 'N/A'):.1f}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        raise

def test_deepsort_components():
    """Test DeepSORT components separately"""
    print("\n🧪 Testing DeepSORT components...")
    
    try:
        from deepsort_tracker import DeepSORTTracker
        
        # Test tracker initialization
        tracker = DeepSORTTracker()
        print("✅ DeepSORT tracker initialized")
        
        # Test with dummy data
        dummy_detections = [
            {'bbox': [100, 100, 200, 200], 'confidence': 0.8, 'class': 'car'},
            {'bbox': [300, 150, 400, 250], 'confidence': 0.7, 'class': 'truck'}
        ]
        
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        tracked = tracker.update(dummy_detections, dummy_frame)
        print(f"✅ Tracking test: {len(tracked)} objects tracked")
        
        stats = tracker.get_tracking_stats()
        print(f"✅ Stats: {stats}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure to install: pip install deep-sort-realtime")
    except Exception as e:
        print(f"❌ Component test failed: {e}")

if __name__ == "__main__":
    # Run component test
    test_deepsort_components()
    
    # Run main demo
    main() 