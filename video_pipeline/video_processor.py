#!/usr/bin/env python3
"""
Video Processor Component
"""

import cv2
import numpy as np
import time
import json
import torch
from typing import List, Tuple, Dict, Optional
from classifier import VehicleClassifier
from detector import YOLODetector
from deepsort_tracker import DeepSORTTracker
from collections import defaultdict, deque
from tqdm import tqdm

class VideoProcessor:
    """Main video processing pipeline"""
    
    def __init__(self, yolo_model_path: Optional[str] = None, 
                 classifier_model_path: Optional[str] = None,
                 device: str = 'auto', use_deepsort: bool = True, batch_size: int = 8):
        
        print("🚀 Initializing Vehicle Detection Pipeline...")
        
        # Set device and optimize GPU usage
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # GPU optimization settings
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.cuda.empty_cache()  # Clear GPU cache
            print(f"🔥 GPU detected: {torch.cuda.get_device_name()}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # Initialize detector (required)
        self.detector = YOLODetector(yolo_model_path, str(self.device))
        
        # Initialize classifier (optional)
        self.classifier = None
        if classifier_model_path:
            try:
                self.classifier = VehicleClassifier(classifier_model_path, str(self.device))
                print("✅ Vehicle classifier loaded")
            except Exception as e:
                print(f"⚠️  Failed to load classifier: {e}")
                print("   Continuing in detection-only mode")
        else:
            print("🔍 Running in detection-only mode")
        
        # Processing parameters
        self.conf_threshold = 0.5
        self.min_detection_size = 30  # Minimum width/height for classification
        self.use_deepsort = use_deepsort
        self.batch_size = batch_size
        
        # Initialize tracker
        if use_deepsort:
            self.tracker = DeepSORTTracker(
                max_age=30,           # Keep tracks for 30 frames without detection
                n_init=3,             # Require 3 consecutive detections to confirm track
                max_cosine_distance=0.2,  # Feature similarity threshold
                nn_budget=100         # Feature gallery size
            )
        else:
            # Fallback to simple tracker
            from simple_tracker import SimpleCentroidTracker
            self.tracker = SimpleCentroidTracker(max_distance=50)
        
        # Classification smoothing (only if classifier is available)
        if self.classifier:
            self.class_history = defaultdict(lambda: deque(maxlen=5))  # 5-frame window
            self.displayed_class = {}  # vehicle_id: current displayed class
            self.class_counter = {}    # vehicle_id: (last predicted class, count)
            self.consecutive_threshold = 3  # Reduced for DeepSORT (tracks are more stable)
        
        print("✅ Pipeline initialized successfully!")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Process a single frame"""
        
        # Detect vehicles
        detections = self.detector.detect(frame, self.conf_threshold)
        
        # Update tracker with detections
        if self.use_deepsort:
            # DeepSORT needs the full frame for feature extraction
            tracked_objects = self.tracker.update(detections, frame)
        else:
            # Simple tracker only needs bounding boxes
            bboxes = [d['bbox'] for d in detections]
            if bboxes:
                vehicle_ids = self.tracker.update(bboxes)
                # Convert to tracked_objects format for compatibility
                tracked_objects = []
                for detection, vehicle_id in zip(detections, vehicle_ids):
                    tracked_objects.append({
                        'track_id': vehicle_id,
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'class': detection['class']
                    })
            else:
                tracked_objects = []

        results = []
        annotated_frame = frame.copy()

        for tracked_obj in tracked_objects:
            track_id = tracked_obj['track_id']
            x1, y1, x2, y2 = tracked_obj['bbox']
            width = x2 - x1
            height = y2 - y1

            # Base result with detection info
            result = {
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'detection_confidence': tracked_obj['confidence'],
                'detection_class': tracked_obj['class'],
                'track_id': track_id,
                'width': width,
                'height': height
            }

            # Add classification if classifier is available and object is large enough
            if (self.classifier and 
                width >= self.min_detection_size and height >= self.min_detection_size):
                
                # Ensure valid crop bounds
                y1, y2 = max(0, y1), min(frame.shape[0], y2)
                x1, x2 = max(0, x1), min(frame.shape[1], x2)
                
                # Validate crop bounds before extracting
                if y2 > y1 and x2 > x1:
                    vehicle_crop = frame[y1:y2, x1:x2]
                    
                    # Check if crop is valid before resizing
                    if vehicle_crop.size > 0 and vehicle_crop.shape[0] > 0 and vehicle_crop.shape[1] > 0:
                        # Add padding to small crops
                        if vehicle_crop.shape[0] < 50 or vehicle_crop.shape[1] < 50:
                            vehicle_crop = cv2.resize(vehicle_crop, (64, 64))  # Minimum size for classifier
                        
                        # Final validation before classification
                        if vehicle_crop.shape[0] >= 10 and vehicle_crop.shape[1] >= 10:
                            vehicle_class, class_confidence = self.classifier.classify(vehicle_crop)

                            # Consecutive prediction logic for classification smoothing
                            last_class, count = self.class_counter.get(track_id, (vehicle_class, 0))
                            if vehicle_class == last_class:
                                count += 1
                            else:
                                count = 1  # reset counter if class changes
                            self.class_counter[track_id] = (vehicle_class, count)

                            # Only update displayed class if threshold is reached
                            if track_id not in self.displayed_class or (vehicle_class != self.displayed_class[track_id] and count >= self.consecutive_threshold):
                                self.displayed_class[track_id] = vehicle_class

                            smoothed_class = self.displayed_class[track_id]
                            
                            # Add classification results
                            result.update({
                                'vehicle_class': smoothed_class,
                                'class_confidence': class_confidence
                            })
                        else:
                            # Add default classification for small/invalid crops
                            result.update({
                                'vehicle_class': 'Unknown',
                                'class_confidence': 0.0
                            })
                    else:
                        # Invalid crop - use detection class
                        result.update({
                            'vehicle_class': tracked_obj.get('class', 'Vehicle'),
                            'class_confidence': 0.0
                        })
                else:
                    # Invalid bounding box - use detection class
                    result.update({
                        'vehicle_class': tracked_obj.get('class', 'Vehicle'),
                        'class_confidence': 0.0
                    })
            else:
                # No classification available - use detection class or default
                result.update({
                    'vehicle_class': tracked_obj.get('class', 'Vehicle'),
                    'class_confidence': 0.0
                })
            
            # Add DeepSORT-specific info if available
            if self.use_deepsort and 'age' in tracked_obj:
                result.update({
                    'track_age': tracked_obj['age'],
                    'track_hits': tracked_obj['hits'],
                    'time_since_update': tracked_obj['time_since_update']
                })
            
            results.append(result)
            annotated_frame = self.annotate_detection(annotated_frame, result)
        
        # Add tracking trails if using DeepSORT
        if self.use_deepsort and hasattr(self.tracker, 'draw_tracks'):
            annotated_frame = self.tracker.draw_tracks(annotated_frame, tracked_objects, 
                                                     draw_trails=True, trail_length=15)
        
        return annotated_frame, results
    
    def annotate_detection(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Annotate frame with detection and classification results"""
        
        x1, y1, x2, y2 = result['bbox']
        track_id = result['track_id']
        
        # Colors
        bbox_color = (0, 255, 0)  # Green
        text_color = (255, 255, 255)  # White
        track_bg_color = (0, 165, 255)  # Orange background for track ID
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
        
        # Show track ID and vehicle class/type
        vehicle_class = result['vehicle_class']
        class_confidence = result.get('class_confidence', 0.0)
        
        # Format display text based on whether we have classification
        if self.classifier and class_confidence > 0:
            track_text = f"ID:{track_id} | {vehicle_class} ({class_confidence:.2f})"
        else:
            track_text = f"ID:{track_id} | {vehicle_class}"
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        track_font_scale = 0.5
        thickness = 1
        
        (track_w, track_h), _ = cv2.getTextSize(track_text, font, track_font_scale, thickness)
        
        # Draw track ID and class at bottom of bounding box
        track_bg_width = track_w + 8
        track_bg_height = track_h + 6
        
        # Position at bottom-center of bounding box
        track_x = x1 + (x2 - x1 - track_bg_width) // 2
        track_y = y2
        
        # Ensure the label doesn't go outside frame bounds
        track_x = max(0, min(track_x, frame.shape[1] - track_bg_width))
        
        # Draw track ID background
        cv2.rectangle(frame, (track_x, track_y), 
                     (track_x + track_bg_width, track_y + track_bg_height), 
                     track_bg_color, -1)
        
        # Draw track ID and class text
        cv2.putText(frame, track_text, (track_x + 4, track_y + track_h + 2), 
                   font, track_font_scale, text_color, thickness)
        
        return frame
    
    def process_video_batch(self, frames_batch: List[np.ndarray]) -> List[Tuple[np.ndarray, List[Dict]]]:
        """Process a batch of frames for better GPU utilization"""
        results = []
        
        # Process batch through detector (can be batched)
        batch_detections = []
        for frame in frames_batch:
            detections = self.detector.detect(frame, self.conf_threshold)
            batch_detections.append(detections)
        
        # Process each frame with tracking and classification
        for frame, detections in zip(frames_batch, batch_detections):
            # Update tracker with detections
            if self.use_deepsort:
                tracked_objects = self.tracker.update(detections, frame)
            else:
                bboxes = [d['bbox'] for d in detections]
                if bboxes:
                    vehicle_ids = self.tracker.update(bboxes)
                    tracked_objects = []
                    for detection, vehicle_id in zip(detections, vehicle_ids):
                        tracked_objects.append({
                            'track_id': vehicle_id,
                            'bbox': detection['bbox'],
                            'confidence': detection['confidence'],
                            'class': detection['class']
                        })
                else:
                    tracked_objects = []

            # Batch classification for better GPU utilization
            frame_results = []
            annotated_frame = frame.copy()
            
            # Process all tracked objects
            if self.classifier:
                # Collect all valid vehicle crops for batch classification
                valid_objects = []
                vehicle_crops = []
                
                for tracked_obj in tracked_objects:
                    track_id = tracked_obj['track_id']
                    x1, y1, x2, y2 = tracked_obj['bbox']
                    width = x2 - x1
                    height = y2 - y1

                    # Different thresholds based on object position or confidence
                    min_size = 30 if tracked_obj['confidence'] > 0.8 else 50

                    if width >= min_size and height >= min_size:
                        # Ensure valid crop bounds
                        y1, y2 = max(0, y1), min(frame.shape[0], y2)
                        x1, x2 = max(0, x1), min(frame.shape[1], x2)
                        
                        # Validate crop bounds before extracting
                        if y2 > y1 and x2 > x1:
                            vehicle_crop = frame[y1:y2, x1:x2]
                            
                            # Check if crop is valid before resizing
                            if vehicle_crop.size > 0 and vehicle_crop.shape[0] > 0 and vehicle_crop.shape[1] > 0:
                                # Add padding to small crops
                                if vehicle_crop.shape[0] < 50 or vehicle_crop.shape[1] < 50:
                                    vehicle_crop = cv2.resize(vehicle_crop, (64, 64))  # Minimum size for classifier
                                
                                # Final validation before adding to batch
                                if vehicle_crop.shape[0] >= 10 and vehicle_crop.shape[1] >= 10:
                                    valid_objects.append(tracked_obj)
                                    vehicle_crops.append(vehicle_crop)
                
                # Batch classify all crops at once
                if vehicle_crops:
                    batch_classes, batch_confidences = self.classifier.classify_batch(vehicle_crops)
                    
                    for tracked_obj, vehicle_class, class_confidence in zip(valid_objects, batch_classes, batch_confidences):
                        track_id = tracked_obj['track_id']
                        x1, y1, x2, y2 = tracked_obj['bbox']
                        
                        # Consecutive prediction logic for classification smoothing
                        last_class, count = self.class_counter.get(track_id, (vehicle_class, 0))
                        if vehicle_class == last_class:
                            count += 1
                        else:
                            count = 1
                        self.class_counter[track_id] = (vehicle_class, count)

                        # Only update displayed class if threshold is reached
                        if track_id not in self.displayed_class or (vehicle_class != self.displayed_class[track_id] and count >= self.consecutive_threshold):
                            self.displayed_class[track_id] = vehicle_class

                        smoothed_class = self.displayed_class[track_id]

                        result = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'detection_confidence': tracked_obj['confidence'],
                            'detection_class': tracked_obj['class'],
                            'vehicle_class': smoothed_class,
                            'class_confidence': class_confidence,
                            'track_id': track_id,
                            'width': x2 - x1,
                            'height': y2 - y1
                        }
                        
                        # Add DeepSORT-specific info if available
                        if self.use_deepsort and 'age' in tracked_obj:
                            result.update({
                                'track_age': tracked_obj['age'],
                                'track_hits': tracked_obj['hits'],
                                'time_since_update': tracked_obj['time_since_update']
                            })
                        
                        frame_results.append(result)
                        annotated_frame = self.annotate_detection(annotated_frame, result)
            else:
                # No classifier - just process detections
                for tracked_obj in tracked_objects:
                    track_id = tracked_obj['track_id']
                    x1, y1, x2, y2 = tracked_obj['bbox']
                    
                    result = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'detection_confidence': tracked_obj['confidence'],
                        'detection_class': tracked_obj['class'],
                        'vehicle_class': tracked_obj.get('class', 'Vehicle'),
                        'class_confidence': 0.0,
                        'track_id': track_id,
                        'width': x2 - x1,
                        'height': y2 - y1
                    }
                    
                    # Add DeepSORT-specific info if available
                    if self.use_deepsort and 'age' in tracked_obj:
                        result.update({
                            'track_age': tracked_obj['age'],
                            'track_hits': tracked_obj['hits'],
                            'time_since_update': tracked_obj['time_since_update']
                        })
                    
                    frame_results.append(result)
                    annotated_frame = self.annotate_detection(annotated_frame, result)
            
            # Add tracking trails if using DeepSORT
            if self.use_deepsort and hasattr(self.tracker, 'draw_tracks'):
                annotated_frame = self.tracker.draw_tracks(annotated_frame, tracked_objects, 
                                                         draw_trails=True, trail_length=15)
            
            results.append((annotated_frame, frame_results))
        
        return results

    def process_video(self, input_path: str, output_path: str, 
                     save_results: bool = True, display: bool = False) -> Dict:
        """Process entire video with optimizations"""
        
        print(f"Processing video: {input_path}")
        
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
        print(f"   Batch size: {self.batch_size}")
        
        # Setup video writer with better codec
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing statistics
        frame_count = 0
        total_detections = 0
        vehicle_counts = {}
        processing_times = []
        all_results = []
        
        # GPU warmup
        print("Warming up GPU...")
        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.process_frame(dummy_frame)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        print("Processing frames with optimizations...")
        
        # Use tqdm for progress bar
        with tqdm(total=total_frames, desc="Processing", unit="frames") as pbar:
            frames_buffer = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    # Process remaining frames in buffer
                    if frames_buffer:
                        start_time = time.time()
                        batch_results = self.process_video_batch(frames_buffer)
                        processing_time = time.time() - start_time
                        processing_times.append(processing_time / len(frames_buffer))
                        
                        for annotated_frame, frame_results in batch_results:
                            frame_count += 1
                            total_detections += len(frame_results)
                            
                            # Count vehicle types
                            for result in frame_results:
                                vehicle_type = result.get('vehicle_class', 'Unknown')
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
                            
                            # Update progress bar
                            pbar.update(1)
                            if processing_times:
                                avg_time = np.mean(processing_times[-10:])
                                pbar.set_postfix({
                                    'Detections': total_detections,
                                    'Avg Time': f"{avg_time:.3f}s/frame",
                                    'GPU': f"{torch.cuda.memory_allocated() / 1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
                                })
                    break
                
                frames_buffer.append(frame)
                
                # Process batch when buffer is full
                if len(frames_buffer) >= self.batch_size:
                    start_time = time.time()
                    batch_results = self.process_video_batch(frames_buffer)
                    processing_time = time.time() - start_time
                    processing_times.append(processing_time / len(frames_buffer))
                    
                    for annotated_frame, frame_results in batch_results:
                        frame_count += 1
                        total_detections += len(frame_results)
                        
                        # Count vehicle types
                        for result in frame_results:
                            vehicle_type = result.get('vehicle_class', 'Unknown')
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
                        
                        # Display frame (optional)
                        if display:
                            cv2.imshow('Vehicle Detection', annotated_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                    
                    # Update progress bar
                    pbar.update(len(frames_buffer))
                    if processing_times:
                        avg_time = np.mean(processing_times[-10:])
                        pbar.set_postfix({
                            'Detections': total_detections,
                            'Avg Time': f"{avg_time:.3f}s/frame",
                            'GPU': f"{torch.cuda.memory_allocated() / 1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
                        })
                    
                    frames_buffer = []
        
        # Cleanup
        cap.release()
        out.release()
        # cv2.destroyAllWindows()
        
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
            print(f"Results saved to: {results_path}")
        
        print("    Video processing complete!")
        print(f"   Total detections: {total_detections}")
        print(f"   Vehicle counts: {vehicle_counts}")
        print(f"   Average processing time: {avg_processing_time:.3f}s/frame")
        print(f"   Output saved to: {output_path}")
        
        return summary
    
    def process_video_realtime(self, input_path: str, output_path: str, 
                               save_results: bool = True, display: bool = True) -> Dict:
        """
        Process video in real-time mode with display and controls.
        This mode prioritizes smooth playback and immediate feedback.
        """
        print(f"Processing video in REAL-TIME mode: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")

        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        total_detections = 0
        vehicle_counts = defaultdict(int)
        processing_times = deque(maxlen=fps) # Keep recent processing times for smooth FPS calc
        all_results = []

        # GPU warmup
        print("Warming up GPU...")
        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.process_frame(dummy_frame)
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        print("Starting real-time processing...")
        print("Press 'q' to quit, 'p' to pause/play")

        paused = False
        
        # Initial timestamp for FPS calculation
        fps_start_time = time.time()
        frames_since_last_fps_calc = 0

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break # End of video

                start_time = time.time()
                
                # Process single frame
                annotated_frame, frame_results = self.process_frame(frame)
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                frame_count += 1
                frames_since_last_fps_calc += 1

                total_detections += len(frame_results)
                for result in frame_results:
                    vehicle_type = result.get('vehicle_class', 'Unknown')
                    vehicle_counts[vehicle_type] += 1

                if save_results:
                    frame_data = {
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps,
                        'detections': frame_results
                    }
                    all_results.append(frame_data)
                
                out.write(annotated_frame)

                # Calculate and display real-time FPS
                if time.time() - fps_start_time >= 1.0:
                    current_fps = frames_since_last_fps_calc / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    frames_since_last_fps_calc = 0
                    
                    avg_processing_time_display = np.mean(processing_times) if processing_times else 0
                    
                    cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Proc: {avg_processing_time_display:.3f}s", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Detections: {len(frame_results)}", (10, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    if torch.cuda.is_available():
                        cv2.putText(annotated_frame, f"GPU: {torch.cuda.memory_allocated() / 1e9:.1f}GB", (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


                if display:
                    cv2.imshow('Vehicle Detection - Real-time', annotated_frame)
                    
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print(f"{'▶️ Resumed' if not paused else '⏸️ Paused'} processing.")
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        avg_processing_time_final = np.mean(processing_times) if processing_times else 0
        total_processing_time = sum(processing_times)
        
        summary = {
            'input_video': input_path,
            'output_video': output_path,
            'total_frames': frame_count,
            'total_detections': total_detections,
            'vehicle_counts': dict(vehicle_counts), # Convert defaultdict to dict for JSON
            'avg_processing_time': avg_processing_time_final,
            'total_processing_time': total_processing_time,
            'fps': fps,
            'resolution': (width, height),
            'real_time_capable': True # Indicate this was run in real-time mode
        }

        if save_results:
            results_path = output_path.replace('.mp4', '_results.json')
            with open(results_path, 'w') as f:
                json.dump({
                    'summary': summary,
                    'frame_results': all_results
                }, f, indent=2)
            print(f"Real-time results saved to: {results_path}")

        print("    Real-time video processing complete!")
        print(f"   Total detections: {total_detections}")
        print(f"   Vehicle counts: {dict(vehicle_counts)}")
        print(f"   Average processing time: {avg_processing_time_final:.3f}s/frame")
        print(f"   Output saved to: {output_path}")

        return summary 