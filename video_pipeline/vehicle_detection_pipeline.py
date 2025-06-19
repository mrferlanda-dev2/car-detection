#!/usr/bin/env python3
"""
Vehicle Detection and Classification Pipeline
Combines YOLO object detection with vehicle classification for video processing
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import time
from typing import List, Tuple, Dict, Optional
import json

class VehicleClassifier:
    """Vehicle classifier using the improved EfficientNet model"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
        self.model = None
        self.class_names = None
        self.transform = None
        self.load_model(model_path)
        
    def load_model(self, model_path: str):
        """Load the trained vehicle classifier"""
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            
            # Get model info
            self.class_names = checkpoint['class_names']
            num_classes = checkpoint['num_classes']
            
            # Create model architecture (same as training)
            self.model = torchvision.models.efficientnet_v2_s(weights=None)
            
            # Recreate classifier (match VeRi training architecture)
            num_ftrs = 1280  # EfficientNet-V2-S features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(num_ftrs, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(256, num_classes)
            )
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Setup transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            print(f"✅ Vehicle classifier loaded successfully")
            print(f"   Classes: {self.class_names}")
            print(f"   Device: {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading vehicle classifier: {e}")
            raise
    
    def classify(self, image: np.ndarray) -> Tuple[str, float]:
        """Classify a vehicle image"""
        try:
            # Preprocess image
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Transform and add batch dimension
            input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = self.class_names[predicted.item()]
                confidence_score = confidence.item()
                
            return predicted_class, confidence_score
            
        except Exception as e:
            print(f"❌ Error in classification: {e}")
            return "Unknown", 0.0

class YOLODetector:
    """YOLO object detector for vehicles"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
        self.model = None
        self.load_model(model_path)
        
        # Vehicle classes in COCO dataset
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
    def load_model(self, model_path: Optional[str] = None):
        """Load YOLO model"""
        try:
            if model_path and os.path.exists(model_path):
                # Try to load custom YOLO model
                try:
                    import ultralytics
                    self.model = ultralytics.YOLO(model_path)
                    print(f"✅ Custom YOLO model loaded from {model_path}")
                    return
                except Exception as e:
                    print(f"⚠️  Failed to load custom YOLO: {e}")
                    print("   Falling back to PyTorch Hub model...")
            
            # Fallback to PyTorch Hub YOLOv5
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.to(self.device)
            print(f"✅ YOLOv5 model loaded from PyTorch Hub")
            print(f"   Device: {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            raise
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        """Detect vehicles in image"""
        try:
            # Run detection
            results = self.model(image)
            
            detections = []
            
            # Process results
            if hasattr(results, 'pandas'):
                # YOLOv5 from PyTorch Hub
                df = results.pandas().xyxy[0]
                for _, row in df.iterrows():
                    class_id = int(row['class'])
                    if class_id in self.vehicle_classes and row['confidence'] >= conf_threshold:
                        detections.append({
                            'bbox': [int(row['xmin']), int(row['ymin']), 
                                   int(row['xmax']), int(row['ymax'])],
                            'confidence': float(row['confidence']),
                            'class': self.vehicle_classes[class_id],
                            'class_id': class_id
                        })
            else:
                # Ultralytics YOLO
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            class_id = int(box.cls)
                            confidence = float(box.conf)
                            if class_id in self.vehicle_classes and confidence >= conf_threshold:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                detections.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': confidence,
                                    'class': self.vehicle_classes[class_id],
                                    'class_id': class_id
                                })
            
            return detections
            
        except Exception as e:
            print(f"❌ Error in detection: {e}")
            return []

class VideoProcessor:
    """Main video processing pipeline"""
    
    def __init__(self, yolo_model_path: Optional[str] = None, 
                 classifier_model_path: str = "improved_vehicle_classifier.pth",
                 device: str = 'auto'):
        
        print("🚀 Initializing Vehicle Detection Pipeline...")
        
        # Initialize models
        self.detector = YOLODetector(yolo_model_path, device)
        self.classifier = VehicleClassifier(classifier_model_path, device)
        
        # Processing parameters
        self.conf_threshold = 0.5
        self.min_detection_size = 50  # Minimum width/height for classification
        
        print("✅ Pipeline initialized successfully!")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single frame"""
        
        # Detect vehicles
        detections = self.detector.detect(frame, self.conf_threshold)
        
        results = []
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Check if detection is large enough for classification
            width = x2 - x1
            height = y2 - y1
            
            if width >= self.min_detection_size and height >= self.min_detection_size:
                # Extract vehicle region
                vehicle_crop = frame[y1:y2, x1:x2]
                
                # Classify vehicle
                vehicle_class, class_confidence = self.classifier.classify(vehicle_crop)
                
                # Store result
                result = {
                    'bbox': [x1, y1, x2, y2],
                    'detection_confidence': detection['confidence'],
                    'detection_class': detection['class'],
                    'vehicle_class': vehicle_class,
                    'class_confidence': class_confidence,
                    'width': width,
                    'height': height
                }
                results.append(result)
                
                # Annotate frame
                annotated_frame = self.annotate_detection(annotated_frame, result)
        
        return annotated_frame, results
    
    def annotate_detection(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Annotate frame with detection and classification results"""
        
        x1, y1, x2, y2 = result['bbox']
        
        # Colors
        bbox_color = (0, 255, 0)  # Green
        text_color = (255, 255, 255)  # White
        bg_color = (0, 0, 0)  # Black background for text
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
        
        # Prepare text
        detection_text = f"{result['detection_class']}: {result['detection_confidence']:.2f}"
        class_text = f"{result['vehicle_class']}: {result['class_confidence']:.2f}"
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        (det_w, det_h), _ = cv2.getTextSize(detection_text, font, font_scale, thickness)
        (cls_w, cls_h), _ = cv2.getTextSize(class_text, font, font_scale, thickness)
        
        # Draw text background
        text_bg_height = det_h + cls_h + 10
        text_bg_width = max(det_w, cls_w) + 10
        
        cv2.rectangle(frame, (x1, y1 - text_bg_height - 5), 
                     (x1 + text_bg_width, y1), bg_color, -1)
        
        # Draw text
        cv2.putText(frame, detection_text, (x1 + 5, y1 - cls_h - 5), 
                   font, font_scale, text_color, thickness)
        cv2.putText(frame, class_text, (x1 + 5, y1 - 5), 
                   font, font_scale, text_color, thickness)
        
        return frame
    
    def process_video(self, input_path: str, output_path: str, 
                     save_results: bool = True, display: bool = False) -> Dict:
        """Process entire video"""
        
        print(f"🎥 Processing video: {input_path}")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing statistics
        frame_count = 0
        total_detections = 0
        vehicle_counts = {}
        processing_times = []
        all_results = []
        
        print("🔄 Processing frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Process frame
            annotated_frame, frame_results = self.process_frame(frame)
            
            # Update statistics
            frame_count += 1
            total_detections += len(frame_results)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Count vehicle types
            for result in frame_results:
                vehicle_type = result['vehicle_class']
                vehicle_counts[vehicle_type] = vehicle_counts.get(vehicle_type, 0) + 1
            
            # Store results
            if save_results:
                frame_data = {
                    'frame_number': frame_count,
                    'timestamp': frame_count / fps,
                    'detections': frame_results
                }
                all_results.append(frame_data)
            
            # Write frame
            out.write(annotated_frame)
            
            # Display progress
            if frame_count % 30 == 0:  # Every 30 frames
                progress = (frame_count / total_frames) * 100
                avg_time = np.mean(processing_times[-30:])
                print(f"   Progress: {progress:.1f}% | "
                      f"Frame {frame_count}/{total_frames} | "
                      f"Avg time: {avg_time:.3f}s/frame")
            
            # Display frame (optional)
            if display:
                cv2.imshow('Vehicle Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Calculate final statistics
        avg_processing_time = np.mean(processing_times)
        total_time = sum(processing_times)
        
        summary = {
            'input_video': input_path,
            'output_video': output_path,
            'total_frames': frame_count,
            'total_detections': total_detections,
            'vehicle_counts': vehicle_counts,
            'avg_processing_time': avg_processing_time,
            'total_processing_time': total_time,
            'fps': fps,
            'resolution': (width, height)
        }
        
        # Save results
        if save_results:
            results_path = output_path.replace('.mp4', '_results.json')
            with open(results_path, 'w') as f:
                json.dump({
                    'summary': summary,
                    'frame_results': all_results
                }, f, indent=2)
            print(f"💾 Results saved to: {results_path}")
        
        print("✅ Video processing complete!")
        print(f"   Total detections: {total_detections}")
        print(f"   Vehicle counts: {vehicle_counts}")
        print(f"   Average processing time: {avg_processing_time:.3f}s/frame")
        print(f"   Output saved to: {output_path}")
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Vehicle Detection and Classification Pipeline')
    parser.add_argument('input', help='Input video path')
    parser.add_argument('-o', '--output', help='Output video path', 
                       default='output_processed.mp4')
    parser.add_argument('--yolo-model', help='Path to YOLO model', 
                       default='yolo11n.pt')
    parser.add_argument('--classifier-model', help='Path to vehicle classifier', 
                       default='improved_vehicle_classifier.pth')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference')
    parser.add_argument('--display', action='store_true',
                       help='Display video while processing')
    parser.add_argument('--no-save-results', action='store_true',
                       help='Don\'t save detailed results to JSON')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    try:
        processor = VideoProcessor(
            yolo_model_path=args.yolo_model if os.path.exists(args.yolo_model) else None,
            classifier_model_path=args.classifier_model,
            device=args.device
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