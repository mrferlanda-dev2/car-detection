# Vehicle Detection and Classification Pipeline

A complete video processing pipeline that combines YOLO object detection with vehicle classification for traffic surveillance applications.

## 🚀 Features

- **YOLO Object Detection**: Detects vehicles (cars, trucks, buses, motorcycles) in video frames
- **Vehicle Classification**: Classifies detected vehicles into 10 Indonesian vehicle categories
- **Real-time Processing**: Supports webcam and video file processing
- **Flexible Model Support**: Works with custom YOLO models or falls back to PyTorch Hub
- **Comprehensive Output**: Annotated videos, detailed JSON results, and statistics
- **GPU Acceleration**: Automatic CUDA detection for faster processing

## 📁 Project Structure

```
video_pipeline/
├── main.py                     # Main entry point
├── video_processor.py          # Core video processing logic
├── classifier.py              # Vehicle classifier component
├── detector.py                # YOLO detector component
├── demo_webcam.py             # Real-time webcam demo
├── test_single_image.py       # Single image testing
├── requirements.txt           # Python dependencies
├── improved_vehicle_classifier.pth  # Trained classifier model
├── yolo11n.pt                 # YOLO detection model
└── README.md                  # This file
```

## 🔧 Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Models**:
   - `improved_vehicle_classifier.pth` - Vehicle classification model (76.6% accuracy)
   - `yolo11n.pt` - YOLO detection model (or will use PyTorch Hub fallback)

## 📖 Usage

### 1. Process Video File

**Basic Usage**:
```bash
python main.py input_video.mp4
```

**Advanced Usage**:
```bash
python main.py input_video.mp4 \
    --output processed_video.mp4 \
    --conf-threshold 0.6 \
    --device cuda \
    --display
```

**Parameters**:
- `input`: Input video file path
- `--output`: Output video path (default: `output_processed.mp4`)
- `--yolo-model`: Custom YOLO model path (default: `yolo11n.pt`)
- `--classifier-model`: Classifier model path (default: `improved_vehicle_classifier.pth`)
- `--conf-threshold`: Detection confidence threshold (default: 0.5)
- `--device`: Processing device (`auto`, `cpu`, `cuda`)
- `--display`: Show video while processing
- `--no-save-results`: Don't save detailed JSON results

### 2. Real-time Webcam Demo

```bash
python demo_webcam.py
```

**Controls**:
- Press `q` to quit
- Press `s` to save current frame

### 3. Test Single Image

```bash
python test_single_image.py image.jpg
```

**With output**:
```bash
python test_single_image.py image.jpg --output result.jpg
```

## 🎯 Vehicle Classes

The classifier can identify these Indonesian vehicle categories:

1. **City Car** - Compact urban vehicles
2. **Double Cabin** - Pickup trucks with double cab
3. **Hatchback** - Small passenger cars
4. **LCGC** - Low Cost Green Car
5. **MPV** - Multi-Purpose Vehicle
6. **Pick-up** - Single/extended cab trucks
7. **Sedan** - Traditional 4-door cars
8. **SUV** - Sport Utility Vehicle
9. **Truk** - Commercial trucks
10. **Van** - Commercial/passenger vans

## 📊 Output Format

### Video Output
- Annotated video with bounding boxes
- Dual labels: YOLO detection + vehicle classification
- Real-time statistics overlay

### JSON Results
```json
{
  "summary": {
    "total_frames": 1500,
    "total_detections": 450,
    "vehicle_counts": {
      "Sedan": 120,
      "SUV": 85,
      "MPV": 70,
      "..."
    },
    "avg_processing_time": 0.125,
    "fps": 30,
    "resolution": [1920, 1080]
  },
  "frame_results": [
    {
      "frame_number": 1,
      "timestamp": 0.033,
      "detections": [
        {
          "bbox": [100, 150, 300, 250],
          "detection_confidence": 0.85,
          "detection_class": "car",
          "vehicle_class": "Sedan",
          "class_confidence": 0.92,
          "width": 200,
          "height": 100
        }
      ]
    }
  ]
}
```

## ⚡ Performance

### Processing Speed
- **GPU (CUDA)**: ~8-12 FPS on RTX 3060
- **CPU**: ~2-4 FPS on modern CPU
- **Memory**: 2-4GB RAM usage
- **Model Loading**: ~5-10 seconds

### Accuracy
- **YOLO Detection**: Standard COCO performance
- **Vehicle Classification**: 76.6% validation accuracy
- **Combined Pipeline**: Optimized for Indonesian traffic scenarios

## 🔧 Technical Details

### Pipeline Architecture
1. **Frame Input** → YOLO Detection
2. **Vehicle Regions** → EfficientNet Classification  
3. **Results Combination** → Annotation & Output

### Model Specifications
- **YOLO**: YOLOv11n or YOLOv5s (fallback)
- **Classifier**: EfficientNet-V2-S with custom head
- **Input Size**: 224x224 for classification
- **Preprocessing**: ImageNet normalization

### Error Handling
- Automatic fallback to PyTorch Hub if custom YOLO fails
- Graceful handling of corrupted frames
- Memory management for long videos
- Device compatibility checks

## 🛠️ Customization

### Using Custom YOLO Model
```bash
python main.py video.mp4 --yolo-model path/to/custom_yolo.pt
```

### Adjusting Detection Sensitivity
```bash
python main.py video.mp4 --conf-threshold 0.3  # More sensitive
python main.py video.mp4 --conf-threshold 0.8  # Less sensitive
```

### Processing Specific Device
```bash
python main.py video.mp4 --device cuda    # Force GPU
python main.py video.mp4 --device cpu     # Force CPU
```

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM recommended
- CUDA-compatible GPU (optional, for acceleration)

### Python Dependencies
- torch>=2.0.0
- torchvision>=0.15.0
- opencv-python>=4.8.0
- ultralytics>=8.0.0
- numpy>=1.24.0

## 🔍 Troubleshooting

### Common Issues

**1. YOLO Model Loading Error**
```
⚠️ Failed to load custom YOLO: ...
   Falling back to PyTorch Hub model...
```
**Solution**: The pipeline automatically uses PyTorch Hub YOLOv5. No action needed.

**2. CUDA Out of Memory**
```bash
python main.py video.mp4 --device cpu
```

**3. Low Detection Rate**
```bash
python main.py video.mp4 --conf-threshold 0.3
```

**4. Slow Processing**
- Use GPU if available
- Reduce video resolution
- Process shorter segments

### Performance Optimization
1. **Use GPU**: Ensure CUDA is properly installed
2. **Batch Processing**: Process multiple short videos instead of one long video
3. **Resolution**: Lower input resolution for faster processing
4. **Confidence**: Higher threshold = faster processing

## 📈 Future Enhancements

- [ ] Multi-object tracking across frames
- [ ] Speed estimation for detected vehicles
- [ ] Traffic flow analysis
- [ ] Real-time streaming support
- [ ] Web interface for easy usage
- [ ] Mobile deployment optimization

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the pipeline.

## 📄 License

This project is part of the Pusan Scholarship Project for Indonesian vehicle classification research. 