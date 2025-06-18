#!/usr/bin/env python3
"""
Test Pipeline on Single Image
"""

import cv2
import argparse
from video_processor import VideoProcessor

def test_image(image_path: str, output_path: str = None):
    """Test pipeline on a single image"""
    
    print(f"🖼️  Testing pipeline on image: {image_path}")
    
    # Initialize pipeline
    try:
        processor = VideoProcessor(
            yolo_model_path="yolo11n.pt",
            classifier_model_path="improved_vehicle_classifier.pth",
            device='auto'
        )
        processor.conf_threshold = 0.3  # Lower threshold for single image
        
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Cannot load image: {image_path}")
        return
    
    print(f"   Image shape: {image.shape}")
    
    # Process image
    annotated_image, results = processor.process_frame(image)
    
    # Print results
    print(f"✅ Processing complete!")
    print(f"   Total detections: {len(results)}")
    
    for i, result in enumerate(results):
        print(f"   Detection {i+1}:")
        print(f"      YOLO: {result['detection_class']} ({result['detection_confidence']:.3f})")
        print(f"      Classifier: {result['vehicle_class']} ({result['class_confidence']:.3f})")
        print(f"      Size: {result['width']}x{result['height']}")
    
    # Save or display result
    if output_path:
        cv2.imwrite(output_path, annotated_image)
        print(f"💾 Result saved to: {output_path}")
    else:
        # Display image
        cv2.imshow('Vehicle Detection Result', annotated_image)
        print("📺 Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Test Vehicle Detection Pipeline on Single Image')
    parser.add_argument('image', help='Input image path')
    parser.add_argument('-o', '--output', help='Output image path (optional)')
    
    args = parser.parse_args()
    
    test_image(args.image, args.output)

if __name__ == "__main__":
    main() 