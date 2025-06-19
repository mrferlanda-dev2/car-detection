#!/usr/bin/env python3
"""
Main Entry Point for Vehicle Detection Pipeline
"""

import argparse
import os
from video_processor import VideoProcessor

def main():
    parser = argparse.ArgumentParser(description='Vehicle Detection and Classification Pipeline')
    parser.add_argument('input', help='Input video path')
    parser.add_argument('-o', '--output', help='Output video path', 
                       default='output_processed.mp4')
    parser.add_argument('--yolo-model', help='Path to YOLO model', 
                       default='yolo11n.pt')
    parser.add_argument('--classifier-model', help='Path to vehicle classifier', 
                       default='veri_type_classifier_efficientnetv2s_20250618_103353.pth')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference')
    parser.add_argument('--display', action='store_true',
                       help='Display video while processing')
    parser.add_argument('--no-save-results', action='store_true',
                       help='Don\'t save detailed results to JSON')
    parser.add_argument('--simple-tracker', action='store_true',
                       help='Use simple centroid tracker instead of DeepSORT')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for processing frames (default: 8)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    try:
        processor = VideoProcessor(
            yolo_model_path=args.yolo_model if os.path.exists(args.yolo_model) else None,
            classifier_model_path=args.classifier_model,
            device=args.device,
            use_deepsort=not args.simple_tracker,
            batch_size=args.batch_size
        )
        
        # Set confidence threshold
        processor.conf_threshold = args.conf_threshold
        
        # Process video
        summary = processor.process_video(
            input_path=args.input,
            output_path=args.output,
            save_results=not args.no_save_results,
            display=args.display
        )
        
        print("\n🎉 Pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 