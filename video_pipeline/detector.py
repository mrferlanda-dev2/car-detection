#!/usr/bin/env python3
"""
YOLO Detector Component
"""

import torch
import numpy as np
import os
from typing import List, Dict, Optional

class YOLODetector:
    """YOLO object detector for vehicles"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
        self.model = None
        self.is_custom_single_class = False
        self.load_model(model_path)
        
        # Vehicle classes in COCO dataset (for default models)
        self.vehicle_classes = {
            2: 'car',
            # 3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
        # Single class mapping for custom trained models
        self.single_class_mapping = {
            0: 'car'  # Your trained model has only 1 class (index 0) = car
        }
        
    def load_model(self, model_path: Optional[str] = None):
        """Load YOLO model"""
        try:
            if model_path and os.path.exists(model_path):
                # Try to load custom YOLO model
                try:
                    import ultralytics
                    self.model = ultralytics.YOLO(model_path)
                    print(f"Custom YOLO model loaded from {model_path}")
                    
                    # Check if this is a single-class custom model
                    try:
                        if hasattr(self.model, 'names') and len(self.model.names) == 1:
                            self.is_custom_single_class = True
                            print(f"   Detected single-class model (1 class)")
                        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'nc') and self.model.model.nc == 1:
                            self.is_custom_single_class = True
                            print(f"   Detected single-class model (1 class)")
                    except Exception:
                        # Assume custom model if detection fails
                        self.is_custom_single_class = True
                        print(f"   Assuming single-class custom model")
                    
                    return
                except Exception as e:
                    print(f"Failed to load custom YOLO: {e}")
                    print("   Falling back to PyTorch Hub model...")
            
            # Fallback to PyTorch Hub YOLOv5
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.to(self.device)
            print(f"YOLOv5 model loaded from PyTorch Hub")
            print(f"   Device: {self.device}")
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
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
                            
                            # Handle single-class custom model
                            if self.is_custom_single_class:
                                if confidence >= conf_threshold:
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    detections.append({
                                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                        'confidence': confidence,
                                        'class': self.single_class_mapping.get(class_id, 'car'),
                                        'class_id': class_id
                                    })
                            # Handle multi-class COCO model
                            elif class_id in self.vehicle_classes and confidence >= conf_threshold:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                detections.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': confidence,
                                    'class': self.vehicle_classes[class_id],
                                    'class_id': class_id
                                })
            
            return detections
            
        except Exception as e:
            print(f"Error in detection: {e}")
            return [] 