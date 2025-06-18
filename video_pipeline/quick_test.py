#!/usr/bin/env python3
"""
Quick Test Script for Pipeline Verification
"""

import sys
import os
import torch
import cv2
import numpy as np

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from classifier import VehicleClassifier
        print("✅ VehicleClassifier imported successfully")
    except Exception as e:
        print(f"❌ Failed to import VehicleClassifier: {e}")
        return False
    
    try:
        from detector import YOLODetector
        print("✅ YOLODetector imported successfully")
    except Exception as e:
        print(f"❌ Failed to import YOLODetector: {e}")
        return False
    
    try:
        from video_processor import VideoProcessor
        print("✅ VideoProcessor imported successfully")
    except Exception as e:
        print(f"❌ Failed to import VideoProcessor: {e}")
        return False
    
    return True

def test_models():
    """Test if models can be loaded"""
    print("\n🔍 Testing model loading...")
    
    # Test classifier
    try:
        if os.path.exists("improved_vehicle_classifier.pth"):
            from classifier import VehicleClassifier
            classifier = VehicleClassifier("improved_vehicle_classifier.pth", device='cpu')
            print("✅ Vehicle classifier loaded successfully")
            print(f"   Classes: {len(classifier.class_names)}")
        else:
            print("❌ improved_vehicle_classifier.pth not found")
            return False
    except Exception as e:
        print(f"❌ Failed to load classifier: {e}")
        return False
    
    # Test detector
    try:
        from detector import YOLODetector
        detector = YOLODetector(device='cpu')
        print("✅ YOLO detector loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load detector: {e}")
        return False
    
    return True

def test_pipeline():
    """Test the complete pipeline with a dummy image"""
    print("\n🔍 Testing complete pipeline...")
    
    try:
        from video_processor import VideoProcessor
        
        # Create a dummy image (blue car-like rectangle)
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_image[200:280, 250:390] = [255, 100, 100]  # Blue rectangle
        
        # Initialize pipeline
        processor = VideoProcessor(
            yolo_model_path=None,  # Use PyTorch Hub fallback
            classifier_model_path="improved_vehicle_classifier.pth",
            device='cpu'
        )
        
        # Process dummy image
        annotated_frame, results = processor.process_frame(dummy_image)
        
        print(f"✅ Pipeline test completed")
        print(f"   Detections: {len(results)}")
        
        # Save test result
        cv2.imwrite("pipeline_test_result.jpg", annotated_frame)
        print("💾 Test result saved as pipeline_test_result.jpg")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Pipeline Verification Tests\n")
    
    # System info
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"OpenCV version: {cv2.__version__}")
    
    # Run tests
    tests = [
        ("Import Test", test_imports),
        ("Model Loading Test", test_models),
        ("Pipeline Test", test_pipeline)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*50}")
    print(f"SUMMARY: {passed}/{len(tests)} tests passed")
    print('='*50)
    
    if passed == len(tests):
        print("🎉 All tests passed! Pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Test with a real image: python test_single_image.py path/to/image.jpg")
        print("2. Test with webcam: python demo_webcam.py")
        print("3. Process a video: python main.py path/to/video.mp4")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 