#!/usr/bin/env python3
"""
DeepSORT Tracker Component
Advanced multi-object tracking with re-identification features
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from collections import defaultdict, deque

class DeepSORTTracker:
    """DeepSORT tracker with re-identification features"""
    
    def __init__(self, max_age: int = 30, n_init: int = 3, 
                 max_cosine_distance: float = 0.2, nn_budget: int = 100):
        """
        Initialize DeepSORT tracker
        
        Args:
            max_age: Maximum number of frames to keep alive a track without detections
            n_init: Number of consecutive detections before a track is confirmed
            max_cosine_distance: Maximum cosine distance for matching
            nn_budget: Maximum size of the appearance descriptors gallery
        """
        
        # Initialize DeepSORT
        self.object_tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            override_track_class=None,
            embedder="mobilenet",  # Use MobileNet for feature extraction
            half=True,  # Use FP16 for faster inference
            bgr=True,  # Input format is BGR
            embedder_gpu=torch.cuda.is_available(),  # Use GPU if available
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
        
        # Tracking state
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        self.confirmed_tracks = set()
        
        print("DeepSORT tracker initialized")
        print(f"Max age: {max_age} frames")
        print(f"Confirmation threshold: {n_init} detections")
        print(f"Max cosine distance: {max_cosine_distance}")
        print(f"Feature extractor: MobileNet")
        print(f"GPU acceleration: {torch.cuda.is_available()}")
    
    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dictionaries with 'bbox', 'confidence', 'class'
            frame: Current frame for feature extraction
            
        Returns:
            List of tracked objects with track IDs
        """
        if not detections:
            # Update tracker with empty detections to handle disappearing tracks
            tracks = self.object_tracker.update_tracks([], frame=frame)
            return []
        
        # Convert detections to format expected by DeepSORT
        # DeepSORT expects: list of tuples [(bbox, confidence, class), ...]
        # where bbox is [left, top, width, height]
        raw_detections = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            width = x2 - x1
            height = y2 - y1
            
            # DeepSORT format: ([left, top, width, height], confidence, class)
            bbox = [x1, y1, width, height]
            confidence = det['confidence']
            detection_class = det.get('class', 'vehicle')
            
            raw_detections.append((bbox, confidence, detection_class))
        
        # Update tracker
        tracks = self.object_tracker.update_tracks(raw_detections, frame=frame)
        
        # Convert tracks back to our format
        tracked_objects = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltwh = track.to_ltwh()
            
            # Convert back to [x1, y1, x2, y2] format
            x1, y1, w, h = ltwh
            x2 = x1 + w
            y2 = y1 + h
            
            # Ensure coordinates are integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Find corresponding detection
            corresponding_detection = None
            min_distance = float('inf')
            
            for det in detections:
                det_x1, det_y1, det_x2, det_y2 = det['bbox']
                
                # Calculate IoU or distance to match
                det_center = [(det_x1 + det_x2) / 2, (det_y1 + det_y2) / 2]
                track_center = [x1 + w/2, y1 + h/2]
                
                distance = np.sqrt((det_center[0] - track_center[0])**2 + 
                                 (det_center[1] - track_center[1])**2)
                
                if distance < min_distance:
                    min_distance = distance
                    corresponding_detection = det
            
            # Create tracked object
            tracked_obj = {
                'track_id': track_id,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': corresponding_detection['confidence'] if corresponding_detection else 0.5,
                'class': corresponding_detection['class'] if corresponding_detection else 'vehicle',
                'age': track.age,
                'hits': track.hits,
                'time_since_update': track.time_since_update
            }
            
            # Update track history
            self.track_history[track_id].append([int(x1 + w/2), int(y1 + h/2)])
            self.confirmed_tracks.add(track_id)
            
            tracked_objects.append(tracked_obj)
        
        return tracked_objects
    
    def get_track_history(self, track_id: int, max_points: int = 30) -> List[Tuple[int, int]]:
        """Get track history for visualization"""
        if track_id in self.track_history:
            history = list(self.track_history[track_id])
            return history[-max_points:] if len(history) > max_points else history
        return []
    
    def draw_tracks(self, frame: np.ndarray, tracked_objects: List[Dict], 
                   draw_trails: bool = True, trail_length: int = 15) -> np.ndarray:
        """
        Draw tracking information on frame
        
        Args:
            frame: Input frame
            tracked_objects: List of tracked objects
            draw_trails: Whether to draw movement trails
            trail_length: Length of movement trails
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for obj in tracked_objects:
            track_id = obj['track_id']
            x1, y1, x2, y2 = obj['bbox']
            
            # Colors for different track IDs
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
            ]
            color = colors[int(track_id) % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # # Draw track ID
            # label = f"ID: {track_id}"
            # label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            # cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
            #              (x1 + label_size[0], y1), color, -1)
            # cv2.putText(annotated_frame, label, (x1, y1 - 5),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw movement trail
            if draw_trails:
                history = self.get_track_history(track_id, trail_length)
                if len(history) > 1:
                    points = np.array(history, np.int32)
                    for i in range(1, len(points)):
                        thickness = int(3 * (i / len(points)))
                        cv2.line(annotated_frame, tuple(points[i-1]), 
                               tuple(points[i]), color, max(1, thickness))
        
        return annotated_frame
    
    def get_tracking_stats(self) -> Dict:
        """Get tracking statistics"""
        total_tracks = len(self.confirmed_tracks)
        # Get active tracks from the internal tracker
        try:
            active_tracks = len([t for t in self.object_tracker.tracker.tracks if t.is_confirmed()])
        except:
            active_tracks = 0  # Fallback if tracker structure is different
        
        return {
            'total_tracks': total_tracks,
            'active_tracks': active_tracks,
            'confirmed_tracks': len(self.confirmed_tracks)
        }
    
    def reset(self):
        """Reset tracker state"""
        # Reinitialize with default parameters since we can't access them
        self.object_tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_cosine_distance=0.2,
            nn_budget=100,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=torch.cuda.is_available(),
            polygon=False
        )
        self.track_history.clear()
        self.confirmed_tracks.clear()
        print("DeepSORT tracker reset") 