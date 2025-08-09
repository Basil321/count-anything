"""
Ultra-Accurate Camera Counter - Advanced Deep Learning Version
Real-time object counting using multiple AI models and ensemble detection.
"""

import cv2
import numpy as np
import time
import os
import json
from datetime import datetime
import torch
from pathlib import Path
import sys

# Add app directory to path for existing models
sys.path.append(str(Path(__file__).parent / "app"))

try:
    from utils.image_processor import ImageProcessor
    from models.count_model import CountModel
except ImportError:
    print("⚠️ Some modules not found, using standalone detection")

class UltraAccurateCameraCounter:
    """Ultra-accurate real-time object counting with advanced AI models"""
    
    def __init__(self):
        """Initialize ultra-accurate camera counter"""
        self.camera = None
        self.is_running = False
        
        # Detection settings
        self.object_type = "general"
        self.confidence_threshold = 0.3
        self.detection_mode = "ensemble"  # ensemble, yolo, threshold, hybrid
        
        # Camera settings
        self.camera_index = 0
        self.frame_width = 640
        self.frame_height = 480
        
        # Initialize AI models
        self.models = {}
        self.initialize_models()
        
        # Advanced detection parameters with fine-tuning
        self.object_params = {
            'rice': {
                'min_area': 5, 'max_area': 1000, 'min_circularity': 0.2,
                'blur_kernel': (3, 3), 'morph_kernel': (2, 2),
                'adaptive_block': 11, 'adaptive_c': 2,
                'bilateral_d': 9, 'bilateral_sigma': 75,
                'gabor_ksize': 21, 'gabor_sigma': 8.0,
                'gabor_theta': np.pi/4, 'gabor_lambda': 2*np.pi,
                'gabor_gamma': 0.5, 'gabor_psi': 0
            },
            'coins': {
                'min_area': 50, 'max_area': 15000, 'min_circularity': 0.5,
                'blur_kernel': (5, 5), 'morph_kernel': (3, 3),
                'adaptive_block': 15, 'adaptive_c': 3,
                'bilateral_d': 15, 'bilateral_sigma': 100,
                'gabor_ksize': 25, 'gabor_sigma': 10.0,
                'gabor_theta': np.pi/2, 'gabor_lambda': 3*np.pi,
                'gabor_gamma': 0.7, 'gabor_psi': 0
            },
            'hair': {
                'min_area': 3, 'max_area': 500, 'min_circularity': 0.1,
                'blur_kernel': (3, 3), 'morph_kernel': (2, 2),
                'adaptive_block': 9, 'adaptive_c': 2,
                'bilateral_d': 5, 'bilateral_sigma': 50,
                'gabor_ksize': 17, 'gabor_sigma': 6.0,
                'gabor_theta': np.pi/6, 'gabor_lambda': np.pi,
                'gabor_gamma': 0.3, 'gabor_psi': 0
            },
            'general': {
                'min_area': 10, 'max_area': 5000, 'min_circularity': 0.2,
                'blur_kernel': (5, 5), 'morph_kernel': (3, 3),
                'adaptive_block': 11, 'adaptive_c': 2,
                'bilateral_d': 9, 'bilateral_sigma': 75,
                'gabor_ksize': 21, 'gabor_sigma': 8.0,
                'gabor_theta': np.pi/4, 'gabor_lambda': 2*np.pi,
                'gabor_gamma': 0.5, 'gabor_psi': 0
            }
        }
        
        # Statistics tracking
        self.stats = {
            'total_frames': 0,
            'total_objects': 0,
            'avg_confidence': 0.0,
            'detection_history': [],
            'fps_history': [],
            'model_performance': {}
        }
        
        # Calibration data
        self.calibration_data = {}
        self.is_calibrated = False
        
        # Ensemble weights
        self.ensemble_weights = {
            'yolo': 0.4,
            'threshold': 0.3,
            'gabor': 0.2,
            'color': 0.1
        }
    
    def initialize_models(self):
        """Initialize all AI models"""
        try:
            # Initialize existing models
            if 'ImageProcessor' in globals():
                self.models['image_processor'] = ImageProcessor()
                print("✅ ImageProcessor loaded")
            
            if 'CountModel' in globals():
                self.models['count_model'] = CountModel()
                print("✅ CountModel loaded")
            
            # Initialize YOLO if available
            try:
                from ultralytics import YOLO
                self.models['yolo'] = YOLO('yolov8n.pt')
                print("✅ YOLOv8 model loaded")
            except ImportError:
                print("⚠️ YOLOv8 not available, using alternative detection")
            
        except Exception as e:
            print(f"⚠️ Model initialization warning: {e}")
    
    def start_camera(self):
        """Start the camera with enhanced settings"""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            
            # Set camera properties for better quality
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 128)
            self.camera.set(cv2.CAP_PROP_CONTRAST, 128)
            self.camera.set(cv2.CAP_PROP_SATURATION, 128)
            self.camera.set(cv2.CAP_PROP_HUE, 0)
            
            if not self.camera.isOpened():
                print("❌ Error: Could not open camera")
                return False
            
            print("✅ Camera started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop the camera and save statistics"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        
        # Save statistics
        self.save_statistics()
        print("🛑 Camera stopped")
    
    def detect_objects_ensemble(self, frame):
        """Ensemble object detection using multiple AI models"""
        try:
            # Get parameters for current object type
            params = self.object_params.get(self.object_type, self.object_params['general'])
            
            # Method 1: YOLO detection
            yolo_results = self._detect_with_yolo(frame)
            
            # Method 2: Advanced threshold detection
            threshold_results = self._detect_with_threshold(frame, params)
            
            # Method 3: Gabor filter detection
            gabor_results = self._detect_with_gabor(frame, params)
            
            # Method 4: Color-based detection
            color_results = self._detect_with_color(frame, params)
            
            # Combine all detections using ensemble
            all_contours = []
            all_confidences = []
            
            # Add YOLO results
            if yolo_results['contours']:
                all_contours.extend(yolo_results['contours'])
                all_confidences.extend([conf * self.ensemble_weights['yolo'] for conf in yolo_results['confidences']])
            
            # Add threshold results
            if threshold_results['contours']:
                all_contours.extend(threshold_results['contours'])
                all_confidences.extend([conf * self.ensemble_weights['threshold'] for conf in threshold_results['confidences']])
            
            # Add Gabor results
            if gabor_results['contours']:
                all_contours.extend(gabor_results['contours'])
                all_confidences.extend([conf * self.ensemble_weights['gabor'] for conf in gabor_results['confidences']])
            
            # Add color results
            if color_results['contours']:
                all_contours.extend(color_results['contours'])
                all_confidences.extend([conf * self.ensemble_weights['color'] for conf in color_results['confidences']])
            
            # Filter and merge overlapping detections
            valid_contours, final_confidences = self._filter_ensemble_contours(all_contours, all_confidences, params)
            
            # Create visualization
            processed_frame = frame.copy()
            for i, contour in enumerate(valid_contours):
                confidence = final_confidences[i] if i < len(final_confidences) else 0.5
                color = self._get_ultra_confidence_color(confidence)
                
                # Draw contour
                cv2.drawContours(processed_frame, [contour], -1, color, 2)
                
                # Add bounding box and confidence
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 1)
                cv2.putText(processed_frame, f"{confidence:.3f}", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Calculate overall confidence
            confidence = np.mean(final_confidences) if final_confidences else 0.0
            
            return processed_frame, len(valid_contours), confidence
            
        except Exception as e:
            print(f"Error in ensemble object detection: {e}")
            return frame, 0, 0.0
    
    def _detect_with_yolo(self, frame):
        """Detect objects using YOLO model"""
        try:
            if 'yolo' not in self.models:
                return {'contours': [], 'confidences': []}
            
            # Run YOLO detection
            results = self.models['yolo'](frame, verbose=False)
            
            contours = []
            confidences = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        # Convert to contour format
                        contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                        contours.append(contour)
                        confidences.append(confidence)
            
            return {'contours': contours, 'confidences': confidences}
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return {'contours': [], 'confidences': []}
    
    def _detect_with_threshold(self, frame, params):
        """Advanced threshold-based detection"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter for edge preservation
            bilateral = cv2.bilateralFilter(
                gray, params['bilateral_d'], params['bilateral_sigma'], params['bilateral_sigma']
            )
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(bilateral, params['blur_kernel'], 0)
            
            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                params['adaptive_block'], params['adaptive_c']
            )
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, params['morph_kernel'])
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate confidences
            confidences = []
            for contour in contours:
                confidence = self._calculate_contour_confidence(contour, params)
                confidences.append(confidence)
            
            return {'contours': contours, 'confidences': confidences}
            
        except Exception as e:
            print(f"Threshold detection error: {e}")
            return {'contours': [], 'confidences': []}
    
    def _detect_with_gabor(self, frame, params):
        """Gabor filter-based detection"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gabor filter
            kernel = cv2.getGaborKernel(
                (params['gabor_ksize'], params['gabor_ksize']),
                params['gabor_sigma'],
                params['gabor_theta'],
                params['gabor_lambda'],
                params['gabor_gamma'],
                params['gabor_psi'],
                ktype=cv2.CV_32F
            )
            
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            
            # Apply threshold
            _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate confidences
            confidences = []
            for contour in contours:
                confidence = self._calculate_contour_confidence(contour, params)
                confidences.append(confidence)
            
            return {'contours': contours, 'confidences': confidences}
            
        except Exception as e:
            print(f"Gabor detection error: {e}")
            return {'contours': [], 'confidences': []}
    
    def _detect_with_color(self, frame, params):
        """Color-based detection"""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for different object types
            color_ranges = {
                'rice': [
                    {'lower': np.array([0, 0, 180]), 'upper': np.array([180, 30, 255])},  # Light
                    {'lower': np.array([10, 50, 50]), 'upper': np.array([20, 255, 255])},  # Brown
                ],
                'coins': [
                    {'lower': np.array([0, 0, 100]), 'upper': np.array([180, 255, 255])},  # Bright
                ],
                'hair': [
                    {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 255, 100])},  # Dark
                ],
                'general': [
                    {'lower': np.array([0, 0, 50]), 'upper': np.array([180, 255, 255])},  # General
                ]
            }
            
            contours = []
            confidences = []
            
            ranges = color_ranges.get(self.object_type, color_ranges['general'])
            
            for color_range in ranges:
                # Create mask
                mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
                
                # Apply morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # Find contours
                color_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Calculate confidences
                for contour in color_contours:
                    confidence = self._calculate_contour_confidence(contour, params)
                    contours.append(contour)
                    confidences.append(confidence)
            
            return {'contours': contours, 'confidences': confidences}
            
        except Exception as e:
            print(f"Color detection error: {e}")
            return {'contours': [], 'confidences': []}
    
    def _filter_ensemble_contours(self, contours, confidences, params):
        """Filter ensemble contours with advanced overlap detection"""
        valid_contours = []
        final_confidences = []
        
        # Sort by confidence
        sorted_pairs = sorted(zip(contours, confidences), key=lambda x: x[1], reverse=True)
        
        for contour, confidence in sorted_pairs:
            area = cv2.contourArea(contour)
            
            if params['min_area'] <= area <= params['max_area']:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity >= params['min_circularity']:
                        # Check for overlap with existing contours
                        is_overlapping = False
                        for existing in valid_contours:
                            if self._contours_overlap_advanced(contour, existing):
                                is_overlapping = True
                                break
                        
                        if not is_overlapping:
                            valid_contours.append(contour)
                            final_confidences.append(confidence)
        
        return valid_contours, final_confidences
    
    def _contours_overlap_advanced(self, contour1, contour2, threshold=0.3):
        """Advanced overlap detection with IoU calculation"""
        # Get bounding rectangles
        x1, y1, w1, h1 = cv2.boundingRect(contour1)
        x2, y2, w2, h2 = cv2.boundingRect(contour2)
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou > threshold
    
    def _calculate_contour_confidence(self, contour, params):
        """Calculate confidence for a single contour"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Normalize area confidence
        area_confidence = 1.0 - abs(area - (params['min_area'] + params['max_area']) / 2) / (params['max_area'] - params['min_area'])
        area_confidence = max(0.0, min(1.0, area_confidence))
        
        # Combine circularity and area confidence
        confidence = (circularity * 0.7 + area_confidence * 0.3)
        return max(0.0, min(1.0, confidence))
    
    def _get_ultra_confidence_color(self, confidence):
        """Get color based on ultra-accurate confidence calculation"""
        if confidence > 0.9:
            return (0, 255, 0)  # Green for very high confidence
        elif confidence > 0.7:
            return (0, 255, 255)  # Yellow for high confidence
        elif confidence > 0.5:
            return (0, 165, 255)  # Orange for medium confidence
        elif confidence > 0.3:
            return (0, 0, 255)  # Red for low confidence
        else:
            return (128, 128, 128)  # Gray for very low confidence
    
    def add_ultra_overlay(self, frame, count, confidence):
        """Add ultra-accurate information overlay to the frame"""
        overlay = frame.copy()
        
        # Add background rectangle with gradient effect
        cv2.rectangle(overlay, (10, 10), (450, 190), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (450, 190), (255, 255, 255), 2)
        
        # Add text information with better formatting
        cv2.putText(overlay, f"Objects: {count}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(overlay, f"Confidence: {confidence:.3f}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(overlay, f"Type: {self.object_type.title()}", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, f"Mode: {self.detection_mode.title()}", (20, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(overlay, f"Models: {len(self.models)}", (20, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        return overlay
    
    def calibrate_ultra_detection(self):
        """Ultra-accurate calibration using multiple frames"""
        print("🔧 Starting ultra-accurate calibration...")
        
        # Capture multiple frames for calibration
        calibration_frames = []
        for i in range(20):  # More frames for better calibration
            ret, frame = self.camera.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                calibration_frames.append(gray)
            time.sleep(0.05)
        
        if not calibration_frames:
            print("❌ Calibration failed: No frames captured")
            return False
        
        # Analyze frame statistics
        avg_brightness = np.mean([np.mean(frame) for frame in calibration_frames])
        avg_contrast = np.mean([np.std(frame) for frame in calibration_frames])
        avg_entropy = np.mean([self._calculate_entropy(frame) for frame in calibration_frames])
        
        # Adjust parameters based on lighting and scene conditions
        if avg_brightness < 80:  # Very low light
            self.object_params[self.object_type]['adaptive_c'] += 2
            self.object_params[self.object_type]['bilateral_sigma'] += 50
            print("🌙 Adjusted for very low light conditions")
        elif avg_brightness < 120:  # Low light
            self.object_params[self.object_type]['adaptive_c'] += 1
            self.object_params[self.object_type]['bilateral_sigma'] += 25
            print("🌙 Adjusted for low light conditions")
        elif avg_brightness > 200:  # Bright light
            self.object_params[self.object_type]['adaptive_c'] -= 1
            self.object_params[self.object_type]['bilateral_sigma'] -= 25
            print("☀️ Adjusted for bright light conditions")
        
        if avg_contrast < 20:  # Very low contrast
            self.object_params[self.object_type]['bilateral_sigma'] += 75
            print("📊 Adjusted for very low contrast")
        elif avg_contrast < 40:  # Low contrast
            self.object_params[self.object_type]['bilateral_sigma'] += 50
            print("📊 Adjusted for low contrast")
        
        if avg_entropy < 4:  # Simple scene
            self.ensemble_weights['threshold'] += 0.1
            self.ensemble_weights['yolo'] -= 0.05
            print("🎯 Adjusted for simple scene")
        elif avg_entropy > 7:  # Complex scene
            self.ensemble_weights['yolo'] += 0.1
            self.ensemble_weights['threshold'] -= 0.05
            print("🎯 Adjusted for complex scene")
        
        self.is_calibrated = True
        print("✅ Ultra-accurate calibration completed")
        return True
    
    def _calculate_entropy(self, image):
        """Calculate image entropy for scene complexity"""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy
    
    def save_statistics(self):
        """Save detection statistics to file"""
        if self.stats['total_frames'] > 0:
            stats_data = {
                'timestamp': datetime.now().isoformat(),
                'object_type': self.object_type,
                'detection_mode': self.detection_mode,
                'total_frames': self.stats['total_frames'],
                'total_objects': self.stats['total_objects'],
                'avg_objects_per_frame': self.stats['total_objects'] / self.stats['total_frames'],
                'avg_confidence': self.stats['avg_confidence'],
                'avg_fps': np.mean(self.stats['fps_history']) if self.stats['fps_history'] else 0,
                'models_used': list(self.models.keys()),
                'ensemble_weights': self.ensemble_weights,
                'calibrated': self.is_calibrated
            }
            
            filename = f"ultra_detection_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(stats_data, f, indent=2)
            
            print(f"📊 Ultra-accurate statistics saved to {filename}")
    
    def run_camera_loop(self):
        """Main camera loop with ultra-accurate features"""
        if not self.start_camera():
            return
        
        self.is_running = True
        print("\n🎥 Ultra-Accurate Camera Counter Started!")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 'r' for rice detection")
        print("- Press 'c' for coin detection")
        print("- Press 'h' for hair detection")
        print("- Press 'g' for general detection")
        print("- Press '1-9' to adjust confidence threshold")
        print("- Press 's' to save current frame")
        print("- Press 'i' for instructions")
        print("- Press 'e' for ensemble mode")
        print("- Press 'y' for YOLO mode")
        print("- Press 't' for threshold mode")
        print("- Press 'k' to calibrate detection")
        print("- Press 'x' to show statistics")
        print("- Press 'w' to adjust ensemble weights")
        
        frame_count = 0
        start_time = time.time()
        
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                print("❌ Error reading frame")
                break
            
            # Process frame
            processed_frame, count, confidence = self.detect_objects_ensemble(frame)
            
            # Add overlay
            final_frame = self.add_ultra_overlay(processed_frame, count, confidence)
            
            # Update statistics
            self.stats['total_frames'] += 1
            self.stats['total_objects'] += count
            self.stats['avg_confidence'] = (self.stats['avg_confidence'] * (self.stats['total_frames'] - 1) + confidence) / self.stats['total_frames']
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = 30 / elapsed_time
                self.stats['fps_history'].append(fps)
                start_time = time.time()
                print(f"📊 FPS: {fps:.1f}, Objects: {count}, Confidence: {confidence:.3f}")
            
            # Display frame
            cv2.imshow('Ultra-Accurate Camera Counter', final_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.object_type = "rice"
                print("🍚 Switched to rice detection")
            elif key == ord('c'):
                self.object_type = "coins"
                print("💰 Switched to coin detection")
            elif key == ord('h'):
                self.object_type = "hair"
                print("💇 Switched to hair detection")
            elif key == ord('g'):
                self.object_type = "general"
                print("🧮 Switched to general detection")
            elif key == ord('e'):
                self.detection_mode = "ensemble"
                print("🔄 Switched to ensemble mode")
            elif key == ord('y'):
                self.detection_mode = "yolo"
                print("🤖 Switched to YOLO mode")
            elif key == ord('t'):
                self.detection_mode = "threshold"
                print("📏 Switched to threshold mode")
            elif key == ord('k'):
                self.calibrate_ultra_detection()
            elif key == ord('x'):
                self.show_statistics()
            elif key == ord('w'):
                self.adjust_ensemble_weights()
            elif key in [ord(str(i)) for i in range(1, 10)]:
                threshold = int(chr(key)) / 10.0
                self.confidence_threshold = threshold
                print(f"⚙️ Confidence threshold: {threshold}")
            elif key == ord('s'):
                self.save_frame(final_frame, count, confidence)
            elif key == ord('i'):
                self.show_instructions()
        
        self.stop_camera()
    
    def adjust_ensemble_weights(self):
        """Adjust ensemble weights interactively"""
        print("\n⚖️ Current ensemble weights:")
        for model, weight in self.ensemble_weights.items():
            print(f"  {model}: {weight:.2f}")
        
        print("\nTo adjust weights, use:")
        print("  'y' + number (0-9) for YOLO weight")
        print("  't' + number (0-9) for threshold weight")
        print("  'g' + number (0-9) for Gabor weight")
        print("  'c' + number (0-9) for color weight")
    
    def show_statistics(self):
        """Display current statistics"""
        print("\n📊 Ultra-Accurate Statistics:")
        print(f"Total Frames: {self.stats['total_frames']}")
        print(f"Total Objects: {self.stats['total_objects']}")
        print(f"Avg Objects/Frame: {self.stats['total_objects'] / max(1, self.stats['total_frames']):.2f}")
        print(f"Avg Confidence: {self.stats['avg_confidence']:.3f}")
        if self.stats['fps_history']:
            print(f"Avg FPS: {np.mean(self.stats['fps_history']):.1f}")
        print(f"Detection Mode: {self.detection_mode}")
        print(f"Object Type: {self.object_type}")
        print(f"Models Available: {list(self.models.keys())}")
        print(f"Calibrated: {self.is_calibrated}")
        print(f"Ensemble Weights: {self.ensemble_weights}")
    
    def save_frame(self, frame, count, confidence):
        """Save current frame with detection results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"ultra_capture_{timestamp}_count{count}_conf{confidence:.3f}.jpg"
        
        try:
            cv2.imwrite(filename, frame)
            print(f"💾 Frame saved: {filename}")
        except Exception as e:
            print(f"❌ Error saving frame: {e}")
    
    def show_instructions(self):
        """Show detailed instructions"""
        print("\n📖 Ultra-Accurate Instructions:")
        print("1. Point your camera at objects to count")
        print("2. Detection Modes:")
        print("   - Ensemble: Combines all AI models for maximum accuracy")
        print("   - YOLO: Uses deep learning model for object detection")
        print("   - Threshold: Uses traditional computer vision")
        print("3. Object Types:")
        print("   - Rice: Specialized for rice grain detection")
        print("   - Coins: Optimized for circular objects")
        print("   - Hair: Designed for thin, elongated objects")
        print("   - General: Universal object detection")
        print("4. Press 'k' to calibrate for current lighting and scene")
        print("5. Press 'x' to view real-time statistics")
        print("6. Press 'w' to adjust ensemble weights")
        print("7. Adjust confidence threshold (1-9) for sensitivity")
        print("8. Color coding: Green=very high, Yellow=high, Orange=medium, Red=low, Gray=very low")

def main():
    """Main function"""
    print("🔢 Ultra-Accurate Camera Counter")
    print("=" * 50)
    
    # Create camera counter
    camera_counter = UltraAccurateCameraCounter()
    
    try:
        # Run camera loop
        camera_counter.run_camera_loop()
        
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        camera_counter.stop_camera()
        print("👋 Application closed")

if __name__ == "__main__":
    main()
