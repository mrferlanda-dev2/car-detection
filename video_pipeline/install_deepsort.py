#!/usr/bin/env python3
"""
Installation script for DeepSORT dependencies
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def install_deepsort():
    """Install DeepSORT and related dependencies"""
    print("🚀 Installing DeepSORT dependencies...")
    print("=" * 50)
    
    # List of packages to install
    packages = [
        "deep-sort-realtime>=1.3.2",
        "scipy>=1.10.0", 
        "scikit-learn>=1.3.0"
    ]
    
    all_success = True
    
    for package in packages:
        success = run_command(f"pip install {package}", f"Installing {package}")
        if not success:
            all_success = False
    
    return all_success

def test_installation():
    """Test if DeepSORT installation works"""
    print("\n🧪 Testing DeepSORT installation...")
    print("=" * 50)
    
    try:
        # Test basic imports
        print("🔄 Testing imports...")
        from deep_sort_realtime.deepsort_tracker import DeepSort
        import scipy
        import sklearn
        print("✅ All imports successful")
        
        # Test DeepSORT initialization
        print("🔄 Testing DeepSORT initialization...")
        tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_cosine_distance=0.2,
            nn_budget=100,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=False,  # Use CPU for testing
            polygon=False
        )
        print("✅ DeepSORT initialization successful")
        
        # Test with dummy data
        print("🔄 Testing tracking with dummy data...")
        import numpy as np
        
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_detections = [([100, 100, 50, 50], 0.8, 'vehicle')]  # [(x, y, w, h), confidence, class]
        
        tracks = tracker.update_tracks(dummy_detections, frame=dummy_frame)
        print(f"✅ Tracking test successful: {len(tracks)} tracks")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all dependencies are installed correctly")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main installation and testing function"""
    print("🚗 DeepSORT Installation & Testing Script")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 7):
        print("❌ Python 3.7+ is required for DeepSORT")
        return False
    
    # Install dependencies
    install_success = install_deepsort()
    
    if not install_success:
        print("\n❌ Installation failed. Please check the errors above.")
        return False
    
    # Test installation
    test_success = test_installation()
    
    if test_success:
        print("\n🎉 DeepSORT installation completed successfully!")
        print("You can now use DeepSORT tracking in your video pipeline.")
        print("\nUsage examples:")
        print("  # Run with DeepSORT (default):")
        print("  python3 main.py input_video.mp4")
        print("  # Run with simple tracker:")
        print("  python3 main.py input_video.mp4 --simple-tracker")
        print("  # Run demo:")
        print("  python3 demo_deepsort.py input_video.mp4")
        return True
    else:
        print("\n❌ Installation test failed. DeepSORT may not work correctly.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 