#!/usr/bin/env python3
"""
Main Entry Point for Vehicle Detection Pipeline
"""

import argparse
import os
from video_processor import VideoProcessor

def main():
    parser = argparse.ArgumentParser(
        description='Vehicle Detection and Classification Pipeline',
        epilog='Note: If no classifier model is specified, the pipeline will run in detection-only mode, showing bounding boxes and tracking IDs without vehicle classification.'
    )
    parser.add_argument('input', help='Input video path')
    parser.add_argument('-o', '--output', help='Output video path', 
                       default='output_processed.mp4')
    parser.add_argument('--yolo-model', help='Path to YOLO model', 
                       default='det_model/yolo12_fine_tuned.pt')
    parser.add_argument('--classifier-model', help='Path to vehicle classifier (optional - if not provided, runs detection-only)', 
                       default='classifier_model/vehicle_classifier_resnet_20250619_222145.pth')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference')
    parser.add_argument('--display', action='store_true',
                       help='Display video while processing')
    parser.add_argument('--realtime', action='store_true',
                       help='Use real-time display mode with smooth playback and controls')
    parser.add_argument('--no-save-results', action='store_true',
                       help='Don\'t save detailed results to JSON')
    parser.add_argument('--simple-tracker', action='store_true',
                       help='Use simple centroid tracker instead of DeepSORT')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for processing frames (default: 8)')
    
    args = parser.parse_args()
    
    # Validate classifier model path if provided
    classifier_model_path = None
    if args.classifier_model and os.path.exists(args.classifier_model):
        classifier_model_path = args.classifier_model
        print(f"Using classifier model: {classifier_model_path}")
    elif args.classifier_model:
        print(f"Classifier model not found: {args.classifier_model}")
        print("   Running in detection-only mode")
    else:
        print("Running in detection-only mode (no classifier specified)")
    
    # Initialize pipeline
    try:
        processor = VideoProcessor(
            yolo_model_path=args.yolo_model if os.path.exists(args.yolo_model) else None,
            classifier_model_path=classifier_model_path,
            device=args.device,
            use_deepsort=not args.simple_tracker,
            batch_size=args.batch_size
        )
        
        # Set confidence threshold
        processor.conf_threshold = args.conf_threshold
        
        # Process video
        if args.realtime:
            # Use real-time mode with smooth display
            summary = processor.process_video_realtime(
                input_path=args.input,
                output_path=args.output,
                save_results=not args.no_save_results,
                display=True  # Real-time mode always shows display
            )
        else:
            # Use batch processing mode
            summary = processor.process_video(
                input_path=args.input,
                output_path=args.output,
                save_results=not args.no_save_results,
                display=args.display
            )
        
        print("\nPipeline completed successfully!")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 