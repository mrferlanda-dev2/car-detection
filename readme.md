# Vehicle Detection with YOLOv12n + ResNet18

![Demo GIF](assets/det.gif)


## Overview

This project implements a robust vehicle detection and classification pipeline using YOLOv12n for object detection and ResNet18 for vehicle type classification. The system is designed for real-time video processing and supports both advanced (DeepSORT) and simple tracking options.

---

## Features

- **Vehicle Detection**: Fast and accurate detection using YOLOv12n.
- **Vehicle Classification**: Classifies detected vehicles into 11 categories using a ResNet18-based classifier.
- **Tracking**: Supports both DeepSORT and a simple centroid tracker.
- **Flexible Pipeline**: Can run detection-only, detection+classification, and with/without real-time display.

---

![Demo Classifier](/assets/classifier.gif)

## Training Results


### Vehicle Classifier (ResNet18)
```
Classification Report:
                precision    recall  f1-score   support

           Bus       1.00      1.00      1.00        30
City-Hatchback       0.97      0.97      0.97        59
           MPV       1.00      0.97      0.98        30
       Pick-up       1.00      0.97      0.98        30
           SUV       0.97      0.93      0.95        30
         Sedan       0.90      0.93      0.92        30
          Truk       0.97      0.97      0.97        30
           Van       0.94      1.00      0.97        30

      accuracy                           0.97       269
     macro avg       0.97      0.97      0.97       269
  weighted avg       0.97      0.97      0.97       269
  ```
![Result](/assets/vehicle_classifier_resnet_training_history_20250622_143208.png)
![ConfusisonMatrix](/assets/vehicle_classifier_resnet_test_confusion_matrix_20250622_143208.png)

---

*For more details, see `classifier_finetune\logs\vehicle_classifier_resnet_20250622_141421_best.log`.*

---

## Object Detection Model

- **Model**: YOLOv12n

**Training Results:**
![YOLO Training Results](yolo_finetune/optimized_training/yolo12mn_full_20250619_171948/plots/results.png)


---

## How to Run

### 1. Install Requirements

If you have CUDA Installed
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
```bash
pip install -r requirements.txt
```

#### Download Pre-trained Models

**Vehicle Classifier Model:**
- Download from: https://drive.google.com/file/d/1jzGNa52w7i_pKoagJo_tuO5ESmVY2zno/view?usp=sharing
- Place the file in: `video_pipeline/classifier_model`

**Object Detection Model:**
- Download from: https://drive.google.com/file/d/1qK-hC_36A58VjrpH0CmAYv9YqJsi8fQY/view?usp=drive_link
- Place the file in: `video_pipeline/det_model`

### 3. Run the Pipeline

The main entry point for the pipeline is `video_pipeline/main.py`. You can run the pipeline in several modes:

#### Real-time Mode with DeepSORT

```bash
python video_pipeline/main.py video_pipeline/inputs/traffic_test_10s.mp4  --classifier-model .\video_pipeline\classifier_model\vehicle_classifier_resnet_20250622_143208_best_model.pth --yolo-model video_pipeline/det_model/yolo12_fine_tuned.pt --conf-threshold 0.3 --realtime --output output_with_classifier_and_deepsort.mp4  
```

- Runs with real-time display and classification.

#### Simple Tracker Mode

```bash
python video_pipeline/main.py video_pipeline/inputs/traffic_test_10s.mp4 --simple-tracker --classifier-model .\video_pipeline\classifier_model\vehicle_classifier_resnet_20250622_143208_best_model.pth --yolo-model video_pipeline/det_model/yolo12_fine_tuned.pt --conf-threshold 0.3 --realtime --output output_with_classifier.mp4
```

- Uses a simple centroid tracker instead of DeepSORT.

#### Detection Only (No Classifier)

```bash
python video_pipeline/main.py video_pipeline/inputs/traffic_test_10s.mp4 --yolo-model video_pipeline/det_model/yolo12_fine_tuned.pt
```

- Runs detection and tracking only, without vehicle type classification.

**Other options:**  
- `--display` to show video output during processing  
- `--output` to specify output video path  
- `--conf-threshold` to set detection confidence threshold  
- `--device` to select device (`auto`, `cpu`, `cuda`)

---

## Directory Structure

- `classifier_finetune/` – Classifier training scripts, logs, and models
- `yolo_finetune/` – YOLO training scripts and datasets
- `video_pipeline/` – Main pipeline code and utilities
  - `classifier_model/` – Pre-trained vehicle classifier models
  - `det_model/` – Pre-trained YOLO detection models
  - `inputs/` – Input video files
  - `outputs/` – Processed video outputs
- `assets/` – Demo GIFs and media

---

## Acknowledgements

- YOLOv12n: [YOLO official repo](https://github.com/ultralytics/yolov5)
- ResNet18: [PyTorch Models](https://pytorch.org/vision/stable/models.html)

---
