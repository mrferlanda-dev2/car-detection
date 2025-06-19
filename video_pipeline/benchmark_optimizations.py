#!/usr/bin/env python3
"""
Benchmark script to test optimization improvements
"""

import time
import torch
import argparse
import os
from video_processor import VideoProcessor

def print_gpu_info():
    """Print GPU information"""
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name()}")
        print(f"   VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"   VRAM Available: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f}GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   PyTorch Version: {torch.__version__}")
    else:
        print("❌ No GPU available")

def benchmark_batch_sizes(video_path: str, model_path: str, batch_sizes: list = [1, 4, 8, 16]):
    """Benchmark different batch sizes"""
    print("\n🧪 Benchmarking Batch Sizes...")
    print("=" * 50)
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        # Initialize processor
        processor = VideoProcessor(
            classifier_model_path=model_path,
            device='auto',
            use_deepsort=True,
            batch_size=batch_size
        )
        
        # Test with first 50 frames for quick benchmark
        import cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = min(50, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
        
        start_time = time.time()
        
        # Process video with limited frames for benchmark
        output_path = f"benchmark_batch_{batch_size}.mp4"
        try:
            # Create a temporary limited video for testing
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            temp_out = cv2.VideoWriter("temp_benchmark.mp4", fourcc, fps, (width, height))
            
            frame_count = 0
            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                temp_out.write(frame)
                frame_count += 1
            
            cap.release()
            temp_out.release()
            
            # Process the temporary video
            summary = processor.process_video(
                input_path="temp_benchmark.mp4",
                output_path=output_path,
                save_results=False,
                display=False
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            fps_achieved = total_frames / processing_time
            
            results[batch_size] = {
                'total_time': processing_time,
                'fps': fps_achieved,
                'avg_frame_time': processing_time / total_frames,
                'total_detections': summary['total_detections']
            }
            
            print(f"   ✅ Batch {batch_size}: {fps_achieved:.2f} FPS ({processing_time:.2f}s total)")
            
            # Cleanup
            if os.path.exists(output_path):
                os.remove(output_path)
            if os.path.exists("temp_benchmark.mp4"):
                os.remove("temp_benchmark.mp4")
                
        except Exception as e:
            print(f"   ❌ Batch {batch_size} failed: {e}")
            results[batch_size] = {'error': str(e)}
    
    # Print summary
    print("\n📊 Batch Size Benchmark Results:")
    print("=" * 50)
    for batch_size, result in results.items():
        if 'error' not in result:
            print(f"Batch {batch_size:2d}: {result['fps']:6.2f} FPS | "
                  f"{result['avg_frame_time']:6.3f}s/frame | "
                  f"{result['total_detections']:4d} detections")
    
    # Find optimal batch size
    if results:
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_batch = max(valid_results.keys(), key=lambda x: valid_results[x]['fps'])
            print(f"\n🏆 Optimal batch size: {best_batch} ({valid_results[best_batch]['fps']:.2f} FPS)")
            return best_batch
    
    return 8  # Default fallback

def test_gpu_utilization():
    """Test GPU memory utilization"""
    print("\n🔍 GPU Utilization Test...")
    print("=" * 30)
    
    if not torch.cuda.is_available():
        print("❌ No GPU available for testing")
        return
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    
    print(f"Initial GPU memory: {initial_memory / 1e9:.3f}GB")
    
    # Test model loading
    from classifier import VehicleClassifier
    classifier = VehicleClassifier("veri_type_classifier_efficientnetv2s_20250618_103353.pth", "auto")
    
    after_model_memory = torch.cuda.memory_allocated()
    print(f"After model load: {after_model_memory / 1e9:.3f}GB (+{(after_model_memory - initial_memory) / 1e9:.3f}GB)")
    
    # Test batch processing
    import numpy as np
    dummy_images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)]
    
    # Single image processing
    start_mem = torch.cuda.memory_allocated()
    for img in dummy_images:
        classifier.classify(img)
    single_mem = torch.cuda.memory_allocated()
    
    torch.cuda.empty_cache()
    
    # Batch processing
    start_mem = torch.cuda.memory_allocated()
    classifier.classify_batch(dummy_images)
    batch_mem = torch.cuda.memory_allocated()
    
    print(f"Single processing peak: {single_mem / 1e9:.3f}GB")
    print(f"Batch processing peak: {batch_mem / 1e9:.3f}GB")
    print(f"Memory efficiency: {(single_mem - batch_mem) / 1e9:.3f}GB saved")

def main():
    parser = argparse.ArgumentParser(description='Benchmark video processing optimizations')
    parser.add_argument('video', help='Input video path')
    parser.add_argument('--model', default='veri_type_classifier_efficientnetv2s_20250618_103353.pth',
                       help='Classifier model path')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 4, 8, 16],
                       help='Batch sizes to test')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return
    
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    print("🚀 Video Processing Optimization Benchmark")
    print("=" * 50)
    
    print_gpu_info()
    test_gpu_utilization()
    optimal_batch = benchmark_batch_sizes(args.video, args.model, args.batch_sizes)
    
    print(f"\n💡 Recommendations:")
    print(f"   • Use batch size: {optimal_batch}")
    print(f"   • Enable GPU acceleration with CUDA")
    print(f"   • Use DeepSORT for better tracking")
    print(f"   • Process with: python main.py {args.video} --batch-size {optimal_batch}")

if __name__ == "__main__":
    main() 