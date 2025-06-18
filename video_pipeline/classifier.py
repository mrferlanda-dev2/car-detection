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
            
            # Get model info
            self.class_names = checkpoint['class_names']
            num_classes = checkpoint['num_classes']
            
            # Create model architecture (same as training)
            self.model = torchvision.models.efficientnet_v2_s(weights=None)
            
            # Recreate classifier
            num_ftrs = 1280  # EfficientNet-V2-S features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.4),
                nn.Linear(num_ftrs, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(512, num_classes)
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