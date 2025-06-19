#!/usr/bin/env python3
"""
Vehicle Classifier Component
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from typing import Tuple

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
            
            # Get model info - handle missing class_names with fallback
            if 'class_names' in checkpoint and 'num_classes' in checkpoint:
                self.class_names = checkpoint['class_names']
                num_classes = checkpoint['num_classes']
            else:
                # Fallback class names based on VeRi dataset structure
                print("⚠️  class_names or num_classes not found in checkpoint, using VeRi fallback mapping")
                self.class_names = ['City-Car', 'Double-Cabin', 'LCGC', 'MPV', 'Pick-Up', 'SUV', 'Sedan', 'Truk', 'Van']
                num_classes = len(self.class_names)
                print(f"   Using {num_classes} classes: {self.class_names}")
                
            # Ensure num_classes is valid
            if num_classes is None or num_classes <= 0:
                num_classes = len(self.class_names)
                print(f"   Fixed num_classes to: {num_classes}")
            
            # Create model architecture (same as training)
            self.model = torchvision.models.efficientnet_v2_m(weights=None)
            
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

            # Ensure transform, model, and class_names are set
            if self.transform is None:
                raise RuntimeError("Transform is not set. Model may not be loaded correctly.")
            if self.model is None:
                raise RuntimeError("Model is not set. Model may not be loaded correctly.")
            if self.class_names is None:
                raise RuntimeError("Class names are not set. Model may not be loaded correctly.")

            input_tensor = self.transform(image_rgb)
            if not isinstance(input_tensor, torch.Tensor):
                raise TypeError("Transform did not return a torch.Tensor.")
            input_tensor = input_tensor.unsqueeze(0).to(self.device)

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
    
    def classify_batch(self, images: list) -> Tuple[list, list]:
        """Classify a batch of vehicle images for better GPU utilization"""
        try:
            if not images:
                return [], []
            
            # Ensure transform, model, and class_names are set
            if self.transform is None:
                raise RuntimeError("Transform is not set. Model may not be loaded correctly.")
            if self.model is None:
                raise RuntimeError("Model is not set. Model may not be loaded correctly.")
            if self.class_names is None:
                raise RuntimeError("Class names are not set. Model may not be loaded correctly.")
            
            # Preprocess all images
            batch_tensors = []
            for image in images:
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
                
                input_tensor = self.transform(image_rgb)
                if not isinstance(input_tensor, torch.Tensor):
                    raise TypeError("Transform did not return a torch.Tensor.")
                batch_tensors.append(input_tensor)
            
            # Stack tensors into a batch
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Predict batch
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probabilities, 1)
                
                # Convert to lists
                predicted_classes = [self.class_names[pred.item()] for pred in predicted]
                confidence_scores = [conf.item() for conf in confidences]
            
            return predicted_classes, confidence_scores
            
        except Exception as e:
            print(f"❌ Error in batch classification: {e}")
            # Fallback to individual classification
            classes, confidences = [], []
            for image in images:
                cls, conf = self.classify(image)
                classes.append(cls)
                confidences.append(conf)
            return classes, confidences 