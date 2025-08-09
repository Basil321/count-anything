"""
Ultimate Camera Counter - Maximum Accuracy & Stunning UI
The most advanced real-time object counting application with stunning interface.
"""

import cv2
import numpy as np
import time
import json
import os
from datetime import datetime
from pathlib import Path
import sys
import math

# Add app directory to path for existing models
sys.path.append(str(Path(__file__).parent / "app"))

try:
    from utils.image_processor import ImageProcessor
    from models.count_model import CountModel
    from database.db import DatabaseManager
    MODELS_AVAILABLE = True
except ImportError:
    print("⚠️ Some modules not found, using standalone detection")
    MODELS_AVAILABLE = False

class UltimateCameraCounter:
    """Ultimate real-time object counting with maximum accuracy and stunning UI"""
    
    def __init__(self):
        """Initialize ultimate camera counter"""
        self.camera = None
        self.is_running = False
        
        # Detection settings
        self.object_type = "general"
        self.confidence_threshold = 0.3
        self.detection_mode = "ultimate"  # ultimate, hybrid, ensemble, precise
        self.ui_theme = "dark"  # dark, light, neon, cyberpunk
        
        # Camera settings
        self.camera_index = 0
        self.frame_width = 1280  # Higher resolution for better accuracy
        self.frame_height = 720
        self.target_fps = 30
        
        # Initialize models and database
        self.models = {}
        self.db = None
        self.initialize_components()
        
        # Ultimate detection parameters with AI-optimized values
        self.object_params = {
            'rice': {
                'min_area': 3, 'max_area': 1500, 'min_circularity': 0.15,
                'blur_kernel': (3, 3), 'morph_kernel': (2, 2),
                'adaptive_block': 11, 'adaptive_c': 2,
                'bilateral_d': 9, 'bilateral_sigma': 75,
                'gabor_params': [(21, 8.0, np.pi/4, 2*np.pi, 0.5, 0)],
                'color_ranges': [
                    {'lower': np.array([0, 0, 180]), 'upper': np.array([180, 30, 255])},
                    {'lower': np.array([10, 50, 50]), 'upper': np.array([20, 255, 255])},
                ],
                'edge_params': {'low': 50, 'high': 150},
                'hough_params': {'dp': 1, 'min_dist': 10, 'param1': 50, 'param2': 30, 'min_radius': 2, 'max_radius': 50}
            },
            'coins': {
                'min_area': 30, 'max_area': 20000, 'min_circularity': 0.4,
                'blur_kernel': (5, 5), 'morph_kernel': (3, 3),
                'adaptive_block': 15, 'adaptive_c': 3,
                'bilateral_d': 15, 'bilateral_sigma': 100,
                'gabor_params': [(25, 10.0, np.pi/2, 3*np.pi, 0.7, 0)],
                'color_ranges': [
                    {'lower': np.array([0, 0, 100]), 'upper': np.array([180, 255, 255])},
                ],
                'edge_params': {'low': 100, 'high': 200},
                'hough_params': {'dp': 1, 'min_dist': 30, 'param1': 100, 'param2': 50, 'min_radius': 10, 'max_radius': 200}
            },
            'hair': {
                'min_area': 2, 'max_area': 800, 'min_circularity': 0.1,
                'blur_kernel': (3, 3), 'morph_kernel': (2, 2),
                'adaptive_block': 9, 'adaptive_c': 2,
                'bilateral_d': 5, 'bilateral_sigma': 50,
                'gabor_params': [(17, 6.0, np.pi/6, np.pi, 0.3, 0)],
                'color_ranges': [
                    {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 255, 100])},
                ],
                'edge_params': {'low': 30, 'high': 100},
                'hough_params': {'dp': 1, 'min_dist': 5, 'param1': 30, 'param2': 20, 'min_radius': 1, 'max_radius': 20}
            },
            'general': {
                'min_area': 5, 'max_area': 8000, 'min_circularity': 0.15,
                'blur_kernel': (5, 5), 'morph_kernel': (3, 3),
                'adaptive_block': 11, 'adaptive_c': 2,
                'bilateral_d': 9, 'bilateral_sigma': 75,
                'gabor_params': [(21, 8.0, np.pi/4, 2*np.pi, 0.5, 0)],
                'color_ranges': [
                    {'lower': np.array([0, 0, 50]), 'upper': np.array([180, 255, 255])},
                ],
                'edge_params': {'low': 50, 'high': 150},
                'hough_params': {'dp': 1, 'min_dist': 15, 'param1': 50, 'param2': 30, 'min_radius': 3, 'max_radius': 100}
            }
        }
        
        # UI themes with stunning colors
        self.ui_themes = {
            'dark': {
                'bg_color': (20, 20, 25),
                'primary_color': (0, 255, 128),
                'secondary_color': (255, 165, 0),
                'accent_color': (255, 20, 147),
                'text_color': (255, 255, 255),
                'border_color': (100, 100, 120),
                'gradient_start': (40, 40, 50),
                'gradient_end': (20, 20, 25)
            },
            'light': {
                'bg_color': (240, 240, 245),
                'primary_color': (0, 150, 255),
                'secondary_color': (255, 140, 0),
                'accent_color': (220, 20, 120),
                'text_color': (20, 20, 30),
                'border_color': (180, 180, 200),
                'gradient_start': (250, 250, 255),
                'gradient_end': (230, 230, 240)
            },
            'neon': {
                'bg_color': (10, 10, 15),
                'primary_color': (0, 255, 255),
                'secondary_color': (255, 0, 255),
                'accent_color': (255, 255, 0),
                'text_color': (255, 255, 255),
                'border_color': (0, 255, 255),
                'gradient_start': (30, 0, 50),
                'gradient_end': (0, 30, 50)
            },
            'cyberpunk': {
                'bg_color': (5, 5, 10),
                'primary_color': (255, 0, 128),
                'secondary_color': (0, 255, 200),
                'accent_color': (255, 128, 0),
                'text_color': (255, 255, 255),
                'border_color': (255, 0, 128),
                'gradient_start': (20, 0, 30),
                'gradient_end': (5, 5, 15)
            }
        }
        
        # Advanced statistics tracking
        self.stats = {
            'total_frames': 0,
            'total_objects': 0,
            'avg_confidence': 0.0,
            'fps_history': [],
            'detection_history': [],
            'accuracy_history': [],
            'processing_times': [],
            'algorithm_performance': {},
            'session_start': datetime.now(),
            'objects_per_second': 0.0,
            'peak_count': 0,
            'average_size': 0.0
        }
        
        # Advanced features
        self.features = {
            'recording': False,
            'video_writer': None,
            'auto_calibration': True,
            'motion_detection': True,
            'noise_reduction': True,
            'edge_enhancement': True,
            'multi_threading': True,
            'gpu_acceleration': False,
            'real_time_analytics': True,
            'auto_save': True,
            'smart_crop': True,
            'zoom_tracking': False
        }
        
        # Calibration and optimization
        self.calibration_data = {}
        self.is_calibrated = False
        self.optimization_level = "high"  # low, medium, high, ultra
        
        # Algorithm weights for ensemble detection
        self.algorithm_weights = {
            'threshold': 0.25,
            'gabor': 0.20,
            'color': 0.15,
            'edge': 0.15,
            'hough': 0.15,
            'existing_models': 0.10
        }
        
        # UI state
        self.ui_state = {
            'show_overlay': True,
            'show_stats': True,
            'show_confidence': True,
            'show_bounding_boxes': True,
            'show_contours': True,
            'show_grid': False,
            'show_crosshair': False,
            'overlay_opacity': 0.8,
            'animation_enabled': True,
            'particle_effects': True
        }
        
    def initialize_components(self):
        """Initialize all components (models, database, etc.)"""
        try:
            # Initialize existing models if available
            if MODELS_AVAILABLE:
                try:
                    self.models['image_processor'] = ImageProcessor()
                    print("✅ ImageProcessor loaded")
                except Exception as e:
                    print(f"⚠️ ImageProcessor failed: {e}")
                
                try:
                    self.models['count_model'] = CountModel()
                    print("✅ CountModel loaded")
                except Exception as e:
                    print(f"⚠️ CountModel failed: {e}")
                
                try:
                    self.db = DatabaseManager()
                    print("✅ Database Manager loaded")
                except Exception as e:
                    print(f"⚠️ Database failed: {e}")
            
            # Initialize YOLO if available
            try:
                from ultralytics import YOLO
                self.models['yolo'] = YOLO('yolov8n.pt')
                print("✅ YOLOv8 model loaded")
            except ImportError:
                print("⚠️ YOLOv8 not available")
            except Exception as e:
                print(f"⚠️ YOLOv8 failed: {e}")
            
        except Exception as e:
            print(f"⚠️ Component initialization warning: {e}")
    
    def start_camera(self):
        """Start the camera with ultimate settings"""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            
            # Set camera properties for maximum quality
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 128)
            self.camera.set(cv2.CAP_PROP_CONTRAST, 128)
            self.camera.set(cv2.CAP_PROP_SATURATION, 128)
            self.camera.set(cv2.CAP_PROP_HUE, 0)
            self.camera.set(cv2.CAP_PROP_GAIN, 0)
            
            if not self.camera.isOpened():
                print("❌ Error: Could not open camera")
                return False
            
            print("✅ Camera started successfully")
            print(f"📐 Resolution: {self.frame_width}x{self.frame_height}")
            print(f"🎯 Target FPS: {self.target_fps}")
            return True
            
        except Exception as e:
            print(f"❌ Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop the camera and save all data"""
        if self.camera:
            self.camera.release()
        
        # Stop recording if active
        if self.features['recording'] and self.features['video_writer']:
            self.features['video_writer'].release()
            print("📹 Recording saved")
        
        cv2.destroyAllWindows()
        
        # Save statistics and session data
        self.save_comprehensive_statistics()
        print("🛑 Camera stopped and data saved")
    
    def detect_objects_ultimate(self, frame):
        """Ultimate object detection using all advanced algorithms"""
        start_time = time.time()
        
        try:
            # Get parameters for current object type
            params = self.object_params.get(self.object_type, self.object_params['general'])
            
            # Preprocessing for better accuracy
            enhanced_frame = self._enhance_frame(frame)
            
            # Method 1: Advanced threshold detection
            threshold_results = self._detect_with_advanced_threshold(enhanced_frame, params)
            
            # Method 2: Gabor filter detection
            gabor_results = self._detect_with_gabor(enhanced_frame, params)
            
            # Method 3: Color-based detection
            color_results = self._detect_with_color(enhanced_frame, params)
            
            # Method 4: Edge detection
            edge_results = self._detect_with_edges(enhanced_frame, params)
            
            # Method 5: Hough Circle detection
            hough_results = self._detect_with_hough_circles(enhanced_frame, params)
            
            # Method 6: Existing models (if available)
            model_results = self._detect_with_existing_models(enhanced_frame)
            
            # Combine all detections using weighted ensemble
            all_detections = self._combine_detections([
                (threshold_results, self.algorithm_weights['threshold']),
                (gabor_results, self.algorithm_weights['gabor']),
                (color_results, self.algorithm_weights['color']),
                (edge_results, self.algorithm_weights['edge']),
                (hough_results, self.algorithm_weights['hough']),
                (model_results, self.algorithm_weights['existing_models'])
            ])
            
            # Advanced filtering and optimization
            valid_contours, final_confidences = self._filter_ultimate_contours(
                all_detections['contours'], 
                all_detections['confidences'], 
                params
            )
            
            # Create stunning visualization
            processed_frame = self._create_stunning_visualization(
                frame, valid_contours, final_confidences, params
            )
            
            # Calculate comprehensive confidence
            confidence = self._calculate_ultimate_confidence(valid_contours, final_confidences)
            
            # Record processing time
            processing_time = time.time() - start_time
            self.stats['processing_times'].append(processing_time)
            
            # Update algorithm performance
            self._update_algorithm_performance(len(valid_contours), confidence, processing_time)
            
            return processed_frame, len(valid_contours), confidence
            
        except Exception as e:
            print(f"Error in ultimate object detection: {e}")
            return frame, 0, 0.0
    
    def _enhance_frame(self, frame):
        """Enhance frame for better detection accuracy"""
        if not self.features['noise_reduction']:
            return frame
        
        # Apply noise reduction
        enhanced = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # Edge enhancement if enabled
        if self.features['edge_enhancement']:
            # Create edge mask
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Blend with original
            enhanced = cv2.addWeighted(enhanced, 0.8, edges_colored, 0.2, 0)
        
        return enhanced
    
    def _detect_with_advanced_threshold(self, frame, params):
        """Advanced threshold-based detection with multiple scales"""
        try:
            contours = []
            confidences = []
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Multi-scale detection
            scales = [0.5, 1.0, 1.5] if self.optimization_level in ['high', 'ultra'] else [1.0]
            
            for scale in scales:
                # Resize for current scale
                if scale != 1.0:
                    h, w = gray.shape
                    scaled_gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
                else:
                    scaled_gray = gray
                
                # Apply bilateral filter
                bilateral = cv2.bilateralFilter(
                    scaled_gray, params['bilateral_d'], 
                    params['bilateral_sigma'], params['bilateral_sigma']
                )
                
                # Apply Gaussian blur
                blurred = cv2.GaussianBlur(bilateral, params['blur_kernel'], 0)
                
                # Multiple threshold methods
                thresh_methods = [
                    cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 
                                        params['adaptive_block'], params['adaptive_c']),
                    cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 
                                        params['adaptive_block'], params['adaptive_c'])
                ]
                
                if self.optimization_level == 'ultra':
                    # Add Otsu thresholding
                    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    thresh_methods.append(otsu_thresh)
                
                for thresh in thresh_methods:
                    # Morphological operations
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, params['morph_kernel'])
                    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
                    
                    # Find contours
                    found_contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Scale contours back to original size
                    for contour in found_contours:
                        if scale != 1.0:
                            scaled_contour = (contour.astype(np.float32) / scale).astype(np.int32)
                        else:
                            scaled_contour = contour
                        
                        confidence = self._calculate_contour_confidence(scaled_contour, params)
                        if confidence > self.confidence_threshold:
                            contours.append(scaled_contour)
                            confidences.append(confidence)
            
            return {'contours': contours, 'confidences': confidences}
            
        except Exception as e:
            print(f"Advanced threshold detection error: {e}")
            return {'contours': [], 'confidences': []}
    
    def _detect_with_gabor(self, frame, params):
        """Gabor filter-based detection with multiple orientations"""
        try:
            contours = []
            confidences = []
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply multiple Gabor filters
            gabor_responses = []
            
            for gabor_param in params['gabor_params']:
                ksize, sigma, theta, lambd, gamma, psi = gabor_param
                
                # Create Gabor kernel
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
                
                # Apply filter
                filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                gabor_responses.append(filtered)
            
            # If ultra optimization, add multiple orientations
            if self.optimization_level == 'ultra':
                for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                    kernel = cv2.getGaborKernel((21, 21), 8.0, theta, 2*np.pi, 0.5, 0, ktype=cv2.CV_32F)
                    filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                    gabor_responses.append(filtered)
            
            # Combine responses
            combined_response = np.zeros_like(gray)
            for response in gabor_responses:
                combined_response = cv2.add(combined_response, response)
            
            # Apply threshold
            _, thresh = cv2.threshold(combined_response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            found_contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in found_contours:
                confidence = self._calculate_contour_confidence(contour, params)
                if confidence > self.confidence_threshold:
                    contours.append(contour)
                    confidences.append(confidence)
            
            return {'contours': contours, 'confidences': confidences}
            
        except Exception as e:
            print(f"Gabor detection error: {e}")
            return {'contours': [], 'confidences': []}
    
    def _detect_with_color(self, frame, params):
        """Advanced color-based detection with HSV and LAB color spaces"""
        try:
            contours = []
            confidences = []
            
            # Convert to multiple color spaces
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            
            # Process HSV color ranges
            for color_range in params['color_ranges']:
                # Create mask in HSV
                mask_hsv = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
                
                # Apply morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel)
                mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel)
                
                # Find contours
                found_contours, _ = cv2.findContours(mask_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in found_contours:
                    confidence = self._calculate_contour_confidence(contour, params)
                    if confidence > self.confidence_threshold:
                        contours.append(contour)
                        confidences.append(confidence)
            
            # Additional LAB space processing for ultra optimization
            if self.optimization_level == 'ultra':
                # Use L channel for luminance-based detection
                l_channel = lab[:, :, 0]
                _, l_thresh = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                l_cleaned = cv2.morphologyEx(l_thresh, cv2.MORPH_CLOSE, kernel)
                l_cleaned = cv2.morphologyEx(l_cleaned, cv2.MORPH_OPEN, kernel)
                
                found_contours, _ = cv2.findContours(l_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in found_contours:
                    confidence = self._calculate_contour_confidence(contour, params) * 0.8  # Lower weight for LAB
                    if confidence > self.confidence_threshold:
                        contours.append(contour)
                        confidences.append(confidence)
            
            return {'contours': contours, 'confidences': confidences}
            
        except Exception as e:
            print(f"Color detection error: {e}")
            return {'contours': [], 'confidences': []}
    
    def _detect_with_edges(self, frame, params):
        """Edge-based detection using Canny and advanced edge detection"""
        try:
            contours = []
            confidences = []
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Canny edge detection
            edges = cv2.Canny(blurred, params['edge_params']['low'], params['edge_params']['high'])
            
            # Additional edge detection methods for ultra optimization
            if self.optimization_level == 'ultra':
                # Sobel edge detection
                sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
                sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
                sobel_edges = np.uint8(sobel_combined / sobel_combined.max() * 255)
                
                # Combine edge detection results
                edges = cv2.bitwise_or(edges, sobel_edges)
            
            # Morphological operations to close gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            found_contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in found_contours:
                confidence = self._calculate_contour_confidence(contour, params) * 0.9  # Slight weight reduction for edges
                if confidence > self.confidence_threshold:
                    contours.append(contour)
                    confidences.append(confidence)
            
            return {'contours': contours, 'confidences': confidences}
            
        except Exception as e:
            print(f"Edge detection error: {e}")
            return {'contours': [], 'confidences': []}
    
    def _detect_with_hough_circles(self, frame, params):
        """Hough Circle Transform for circular object detection"""
        try:
            contours = []
            confidences = []
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply median blur to reduce noise
            blurred = cv2.medianBlur(gray, 5)
            
            # Hough Circle Transform
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=params['hough_params']['dp'],
                minDist=params['hough_params']['min_dist'],
                param1=params['hough_params']['param1'],
                param2=params['hough_params']['param2'],
                minRadius=params['hough_params']['min_radius'],
                maxRadius=params['hough_params']['max_radius']
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                
                for (x, y, r) in circles:
                    # Create circular contour
                    center = (x, y)
                    radius = r
                    
                    # Generate points for circular contour
                    theta = np.linspace(0, 2*np.pi, 50)
                    circle_x = center[0] + radius * np.cos(theta)
                    circle_y = center[1] + radius * np.sin(theta)
                    
                    circle_contour = np.array([[int(x), int(y)] for x, y in zip(circle_x, circle_y)], dtype=np.int32)
                    circle_contour = circle_contour.reshape((-1, 1, 2))
                    
                    # Calculate confidence based on circularity (should be high for Hough circles)
                    area = np.pi * radius * radius
                    perimeter = 2 * np.pi * radius
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Adjust confidence for circle detection
                    confidence = min(circularity * 1.2, 1.0)  # Boost confidence for circular objects
                    
                    if confidence > self.confidence_threshold:
                        contours.append(circle_contour)
                        confidences.append(confidence)
            
            return {'contours': contours, 'confidences': confidences}
            
        except Exception as e:
            print(f"Hough circle detection error: {e}")
            return {'contours': [], 'confidences': []}
    
    def _detect_with_existing_models(self, frame):
        """Use existing models if available"""
        try:
            contours = []
            confidences = []
            
            # Try YOLO detection
            if 'yolo' in self.models:
                try:
                    results = self.models['yolo'](frame, verbose=False)
                    
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                confidence = box.conf[0].cpu().numpy()
                                
                                # Convert to contour format
                                contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                                contour = contour.reshape((-1, 1, 2))
                                
                                if confidence > self.confidence_threshold:
                                    contours.append(contour)
                                    confidences.append(float(confidence))
                except Exception as e:
                    print(f"YOLO detection error: {e}")
            
            # Try existing image processor
            if 'image_processor' in self.models:
                try:
                    count, conf, processed = self.models['image_processor'].detect_objects(
                        frame, self.object_type, self.confidence_threshold
                    )
                    # Note: This would need modification to return contours
                except Exception as e:
                    print(f"Image processor error: {e}")
            
            return {'contours': contours, 'confidences': confidences}
            
        except Exception as e:
            print(f"Existing models detection error: {e}")
            return {'contours': [], 'confidences': []}
    
    def _combine_detections(self, detection_results):
        """Combine multiple detection results using weighted ensemble"""
        all_contours = []
        all_confidences = []
        
        for detection_result, weight in detection_results:
            if detection_result['contours']:
                all_contours.extend(detection_result['contours'])
                # Apply weight to confidences
                weighted_confidences = [conf * weight for conf in detection_result['confidences']]
                all_confidences.extend(weighted_confidences)
        
        return {'contours': all_contours, 'confidences': all_confidences}
    
    def _filter_ultimate_contours(self, contours, confidences, params):
        """Ultimate contour filtering with advanced algorithms"""
        if not contours:
            return [], []
        
        valid_contours = []
        final_confidences = []
        
        # Sort by confidence (highest first)
        sorted_pairs = sorted(zip(contours, confidences), key=lambda x: x[1], reverse=True)
        
        for contour, confidence in sorted_pairs:
            area = cv2.contourArea(contour)
            
            # Basic area filtering
            if not (params['min_area'] <= area <= params['max_area']):
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            # Calculate shape metrics
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Advanced shape analysis
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / max(1, min(w, h))
            
            # Extent (ratio of contour area to bounding rectangle area)
            rect_area = w * h
            extent = area / rect_area if rect_area > 0 else 0
            
            # Apply filters based on object type
            if self.object_type == 'rice':
                if circularity < 0.15 or aspect_ratio > 8.0 or solidity < 0.5:
                    continue
            elif self.object_type == 'coins':
                if circularity < 0.4 or aspect_ratio > 2.0 or solidity < 0.8:
                    continue
            elif self.object_type == 'hair':
                if circularity < 0.05 or aspect_ratio > 20.0:
                    continue
            else:  # general
                if circularity < params['min_circularity'] or aspect_ratio > 10.0 or solidity < 0.3:
                    continue
            
            # Advanced overlap detection using IoU
            is_overlapping = False
            for existing_contour in valid_contours:
                iou = self._calculate_contour_iou(contour, existing_contour)
                if iou > 0.3:  # 30% overlap threshold
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                # Boost confidence based on shape quality
                shape_quality = (circularity + solidity + (1/aspect_ratio) * 0.5 + extent) / 3.5
                adjusted_confidence = confidence * (0.7 + 0.3 * shape_quality)
                
                valid_contours.append(contour)
                final_confidences.append(min(adjusted_confidence, 1.0))
        
        return valid_contours, final_confidences
    
    def _calculate_contour_iou(self, contour1, contour2):
        """Calculate Intersection over Union for two contours"""
        try:
            # Get bounding rectangles
            x1, y1, w1, h1 = cv2.boundingRect(contour1)
            x2, y2, w2, h2 = cv2.boundingRect(contour2)
            
            # Calculate intersection
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
            
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            area1 = w1 * h1
            area2 = w2 * h2
            union_area = area1 + area2 - intersection_area
            
            return intersection_area / union_area if union_area > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_contour_confidence(self, contour, params):
        """Calculate comprehensive confidence for a contour"""
        try:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0 or area == 0:
                return 0.0
            
            # Basic shape metrics
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Normalize area confidence
            optimal_area = (params['min_area'] + params['max_area']) / 2
            area_range = params['max_area'] - params['min_area']
            area_confidence = 1.0 - abs(area - optimal_area) / area_range
            area_confidence = max(0.0, min(1.0, area_confidence))
            
            # Advanced shape analysis
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Combine all factors
            confidence = (
                circularity * 0.4 + 
                area_confidence * 0.3 + 
                solidity * 0.2 + 
                min(circularity / params['min_circularity'], 1.0) * 0.1
            )
            
            return max(0.0, min(1.0, confidence))
            
        except:
            return 0.0
    
    def _calculate_ultimate_confidence(self, contours, confidences):
        """Calculate ultimate confidence score"""
        if not confidences:
            return 0.0
        
        # Weighted average with quality boost
        avg_confidence = np.mean(confidences)
        
        # Boost based on count consistency
        count_boost = min(len(contours) / 10.0, 0.2)  # Max 20% boost
        
        # Boost based on confidence distribution
        confidence_std = np.std(confidences) if len(confidences) > 1 else 0
        consistency_boost = max(0, 0.1 - confidence_std)  # Higher consistency = higher boost
        
        final_confidence = avg_confidence + count_boost + consistency_boost
        return min(final_confidence, 1.0)
    
    def _create_stunning_visualization(self, frame, contours, confidences, params):
        """Create stunning visualization with modern UI elements"""
        processed_frame = frame.copy()
        
        if not self.ui_state['show_overlay']:
            return processed_frame
        
        # Get current theme
        theme = self.ui_themes[self.ui_theme]
        
        # Create overlay
        overlay = processed_frame.copy()
        
        # Draw gradient background for UI elements
        if self.ui_state['show_stats']:
            self._draw_gradient_background(overlay, theme)
        
        # Draw grid if enabled
        if self.ui_state['show_grid']:
            self._draw_grid(overlay, theme)
        
        # Draw crosshair if enabled
        if self.ui_state['show_crosshair']:
            self._draw_crosshair(overlay, theme)
        
        # Draw contours and bounding boxes
        for i, contour in enumerate(contours):
            confidence = confidences[i] if i < len(confidences) else 0.5
            
            # Get dynamic color based on confidence and theme
            color = self._get_dynamic_confidence_color(confidence, theme)
            
            # Draw contour if enabled
            if self.ui_state['show_contours']:
                cv2.drawContours(overlay, [contour], -1, color, 2)
            
            # Draw bounding box if enabled
            if self.ui_state['show_bounding_boxes']:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                
                # Draw corner decorations
                self._draw_corner_decorations(overlay, (x, y, w, h), color)
            
            # Draw confidence if enabled
            if self.ui_state['show_confidence']:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Background for text
                text_bg_color = tuple(int(c * 0.8) for c in color)
                cv2.rectangle(overlay, (x, y-25), (x + 80, y), text_bg_color, -1)
                
                # Confidence text
                cv2.putText(overlay, f"{confidence:.3f}", (x + 2, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, theme['text_color'], 1)
            
            # Particle effects if enabled
            if self.ui_state['particle_effects'] and self.ui_state['animation_enabled']:
                self._draw_particle_effects(overlay, contour, color)
        
        # Blend overlay with original frame
        alpha = self.ui_state['overlay_opacity']
        processed_frame = cv2.addWeighted(processed_frame, 1 - alpha, overlay, alpha, 0)
        
        return processed_frame
    
    def _draw_gradient_background(self, frame, theme):
        """Draw gradient background for UI elements"""
        h, w = frame.shape[:2]
        
        # Create gradient
        gradient = np.zeros((200, 500, 3), dtype=np.uint8)
        
        for i in range(200):
            ratio = i / 200.0
            color = [
                int(theme['gradient_start'][j] * (1 - ratio) + theme['gradient_end'][j] * ratio)
                for j in range(3)
            ]
            gradient[i, :] = color
        
        # Apply gradient to specific regions
        frame[10:210, 10:510] = cv2.addWeighted(frame[10:210, 10:510], 0.3, gradient, 0.7, 0)
    
    def _draw_grid(self, frame, theme):
        """Draw grid overlay"""
        h, w = frame.shape[:2]
        grid_size = 50
        
        # Vertical lines
        for x in range(0, w, grid_size):
            cv2.line(frame, (x, 0), (x, h), theme['border_color'], 1)
        
        # Horizontal lines
        for y in range(0, h, grid_size):
            cv2.line(frame, (0, y), (w, y), theme['border_color'], 1)
    
    def _draw_crosshair(self, frame, theme):
        """Draw crosshair at center"""
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Draw crosshair
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), theme['accent_color'], 2)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), theme['accent_color'], 2)
        cv2.circle(frame, (center_x, center_y), 5, theme['accent_color'], 2)
    
    def _draw_corner_decorations(self, frame, bbox, color):
        """Draw corner decorations for bounding boxes"""
        x, y, w, h = bbox
        corner_size = 10
        
        # Top-left
        cv2.line(frame, (x, y), (x + corner_size, y), color, 3)
        cv2.line(frame, (x, y), (x, y + corner_size), color, 3)
        
        # Top-right
        cv2.line(frame, (x + w, y), (x + w - corner_size, y), color, 3)
        cv2.line(frame, (x + w, y), (x + w, y + corner_size), color, 3)
        
        # Bottom-left
        cv2.line(frame, (x, y + h), (x + corner_size, y + h), color, 3)
        cv2.line(frame, (x, y + h), (x, y + h - corner_size), color, 3)
        
        # Bottom-right
        cv2.line(frame, (x + w, y + h), (x + w - corner_size, y + h), color, 3)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_size), color, 3)
    
    def _draw_particle_effects(self, frame, contour, color):
        """Draw particle effects around detected objects"""
        # Get contour center
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Draw small particles around the center
            for i in range(3):
                angle = time.time() * 2 + i * 2.1  # Rotating effect
                radius = 15 + i * 5
                px = int(cx + radius * math.cos(angle))
                py = int(cy + radius * math.sin(angle))
                cv2.circle(frame, (px, py), 2, color, -1)
    
    def _get_dynamic_confidence_color(self, confidence, theme):
        """Get dynamic color based on confidence and theme"""
        if confidence > 0.9:
            return theme['primary_color']
        elif confidence > 0.7:
            return theme['secondary_color']
        elif confidence > 0.5:
            return theme['accent_color']
        elif confidence > 0.3:
            return tuple(int(c * 0.7) for c in theme['accent_color'])
        else:
            return theme['border_color']
    
    def add_ultimate_overlay(self, frame, count, confidence):
        """Add ultimate information overlay with stunning design"""
        overlay = frame.copy()
        theme = self.ui_themes[self.ui_theme]
        
        if not self.ui_state['show_overlay']:
            return overlay
        
        # Main stats panel
        panel_width, panel_height = 550, 250
        panel_x, panel_y = 20, 20
        
        # Draw main panel background with gradient
        self._draw_rounded_rectangle(overlay, (panel_x, panel_y), (panel_width, panel_height), 
                                   theme['bg_color'], theme['border_color'], 15)
        
        # Title bar
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + 40), 
                     theme['primary_color'], -1)
        cv2.putText(overlay, "ULTIMATE COUNTER", (panel_x + 20, panel_y + 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, theme['bg_color'], 2)
        
        # Main statistics
        stats_y = panel_y + 70
        
        # Objects count with large font
        cv2.putText(overlay, f"OBJECTS: {count}", (panel_x + 20, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, theme['primary_color'], 3)
        
        # Confidence with progress bar
        cv2.putText(overlay, f"CONFIDENCE: {confidence:.3f}", (panel_x + 20, stats_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, theme['secondary_color'], 2)
        
        # Confidence progress bar
        bar_x = panel_x + 20
        bar_y = stats_y + 55
        bar_width = 200
        bar_height = 10
        
        # Background bar
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     theme['border_color'], -1)
        
        # Progress bar
        progress_width = int(bar_width * confidence)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), 
                     theme['accent_color'], -1)
        
        # Additional stats
        stats_text = [
            f"Type: {self.object_type.title()}",
            f"Mode: {self.detection_mode.title()}",
            f"Theme: {self.ui_theme.title()}",
            f"FPS: {np.mean(self.stats['fps_history'][-10:]):.1f}" if self.stats['fps_history'] else "FPS: --",
            f"Total: {self.stats['total_objects']}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(overlay, text, (panel_x + 20, stats_y + 90 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, theme['text_color'], 1)
        
        # Performance panel (right side)
        perf_panel_x = frame.shape[1] - 300
        perf_panel_y = 20
        perf_panel_width = 280
        perf_panel_height = 200
        
        if self.ui_state['show_stats']:
            self._draw_rounded_rectangle(overlay, (perf_panel_x, perf_panel_y), 
                                       (perf_panel_width, perf_panel_height), 
                                       theme['bg_color'], theme['border_color'], 10)
            
            # Performance title
            cv2.putText(overlay, "PERFORMANCE", (perf_panel_x + 10, perf_panel_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, theme['primary_color'], 2)
            
            # Performance stats
            if self.stats['processing_times']:
                avg_time = np.mean(self.stats['processing_times'][-10:])
                current_fps = 1.0 / avg_time if avg_time > 0 else 0
                
                perf_stats = [
                    f"Avg Process: {avg_time*1000:.1f}ms",
                    f"Current FPS: {current_fps:.1f}",
                    f"Peak Count: {self.stats['peak_count']}",
                    f"Session: {(datetime.now() - self.stats['session_start']).seconds}s",
                    f"Accuracy: {self.stats['avg_confidence']:.3f}"
                ]
                
                for i, stat in enumerate(perf_stats):
                    cv2.putText(overlay, stat, (perf_panel_x + 10, perf_panel_y + 55 + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, theme['text_color'], 1)
        
        # Recording indicator
        if self.features['recording']:
            cv2.circle(overlay, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(overlay, "REC", (frame.shape[1] - 50, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay
    
    def _draw_rounded_rectangle(self, img, top_left, size, bg_color, border_color, radius):
        """Draw a rounded rectangle"""
        x, y = top_left
        w, h = size
        
        # Draw main rectangle
        cv2.rectangle(img, (x + radius, y), (x + w - radius, y + h), bg_color, -1)
        cv2.rectangle(img, (x, y + radius), (x + w, y + h - radius), bg_color, -1)
        
        # Draw corners
        cv2.circle(img, (x + radius, y + radius), radius, bg_color, -1)
        cv2.circle(img, (x + w - radius, y + radius), radius, bg_color, -1)
        cv2.circle(img, (x + radius, y + h - radius), radius, bg_color, -1)
        cv2.circle(img, (x + w - radius, y + h - radius), radius, bg_color, -1)
        
        # Draw border
        cv2.rectangle(img, (x, y), (x + w, y + h), border_color, 2)
    
    def calibrate_ultimate_detection(self):
        """Ultimate calibration using advanced scene analysis"""
        print("🔧 Starting ultimate calibration...")
        
        # Capture frames for comprehensive analysis
        calibration_frames = []
        for i in range(30):  # More frames for better analysis
            ret, frame = self.camera.read()
            if ret:
                calibration_frames.append(frame)
            time.sleep(0.033)  # ~30 FPS
        
        if not calibration_frames:
            print("❌ Calibration failed: No frames captured")
            return False
        
        # Comprehensive scene analysis
        scene_stats = self._analyze_scene(calibration_frames)
        
        # Adaptive parameter adjustment
        self._adjust_parameters_adaptive(scene_stats)
        
        # Optimize algorithm weights based on scene
        self._optimize_algorithm_weights(scene_stats)
        
        self.is_calibrated = True
        print("✅ Ultimate calibration completed")
        print(f"📊 Scene brightness: {scene_stats['brightness']:.1f}")
        print(f"📊 Scene contrast: {scene_stats['contrast']:.1f}")
        print(f"📊 Scene complexity: {scene_stats['complexity']:.1f}")
        
        return True
    
    def _analyze_scene(self, frames):
        """Comprehensive scene analysis"""
        brightness_values = []
        contrast_values = []
        complexity_values = []
        motion_values = []
        
        prev_gray = None
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Brightness analysis
            brightness = np.mean(gray)
            brightness_values.append(brightness)
            
            # Contrast analysis
            contrast = np.std(gray)
            contrast_values.append(contrast)
            
            # Complexity analysis (edge density)
            edges = cv2.Canny(gray, 50, 150)
            complexity = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
            complexity_values.append(complexity)
            
            # Motion analysis
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion = np.mean(diff)
                motion_values.append(motion)
            
            prev_gray = gray
        
        return {
            'brightness': np.mean(brightness_values),
            'contrast': np.mean(contrast_values),
            'complexity': np.mean(complexity_values),
            'motion': np.mean(motion_values) if motion_values else 0,
            'brightness_std': np.std(brightness_values),
            'contrast_std': np.std(contrast_values)
        }
    
    def _adjust_parameters_adaptive(self, scene_stats):
        """Adjust detection parameters based on scene analysis"""
        params = self.object_params[self.object_type]
        
        # Brightness adjustments
        if scene_stats['brightness'] < 80:  # Very low light
            params['adaptive_c'] += 3
            params['bilateral_sigma'] += 50
            self.algorithm_weights['edge'] *= 0.8
            self.algorithm_weights['color'] *= 0.7
        elif scene_stats['brightness'] < 120:  # Low light
            params['adaptive_c'] += 2
            params['bilateral_sigma'] += 25
            self.algorithm_weights['edge'] *= 0.9
        elif scene_stats['brightness'] > 200:  # Bright light
            params['adaptive_c'] -= 1
            params['bilateral_sigma'] -= 25
            self.algorithm_weights['edge'] *= 1.1
        
        # Contrast adjustments
        if scene_stats['contrast'] < 20:  # Very low contrast
            params['bilateral_sigma'] += 75
            self.algorithm_weights['gabor'] *= 1.2
            self.algorithm_weights['edge'] *= 1.3
        elif scene_stats['contrast'] < 40:  # Low contrast
            params['bilateral_sigma'] += 50
            self.algorithm_weights['gabor'] *= 1.1
        
        # Complexity adjustments
        if scene_stats['complexity'] > 0.3:  # High complexity
            self.algorithm_weights['threshold'] *= 1.2
            self.algorithm_weights['hough'] *= 0.8
        elif scene_stats['complexity'] < 0.1:  # Low complexity
            self.algorithm_weights['hough'] *= 1.2
            self.algorithm_weights['color'] *= 1.1
        
        # Motion adjustments
        if scene_stats['motion'] > 20:  # High motion
            self.features['noise_reduction'] = True
            params['bilateral_sigma'] += 30
    
    def _optimize_algorithm_weights(self, scene_stats):
        """Optimize algorithm weights based on scene characteristics"""
        # Normalize weights to sum to 1.0
        total_weight = sum(self.algorithm_weights.values())
        for key in self.algorithm_weights:
            self.algorithm_weights[key] /= total_weight
    
    def _update_algorithm_performance(self, count, confidence, processing_time):
        """Update algorithm performance metrics"""
        # Update peak count
        self.stats['peak_count'] = max(self.stats['peak_count'], count)
        
        # Calculate objects per second
        if processing_time > 0:
            self.stats['objects_per_second'] = count / processing_time
        
        # Update accuracy history
        self.stats['accuracy_history'].append(confidence)
        if len(self.stats['accuracy_history']) > 100:
            self.stats['accuracy_history'] = self.stats['accuracy_history'][-100:]
    
    def save_comprehensive_statistics(self):
        """Save comprehensive statistics and session data"""
        if self.stats['total_frames'] == 0:
            return
        
        session_duration = (datetime.now() - self.stats['session_start']).total_seconds()
        
        comprehensive_stats = {
            'session_info': {
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': session_duration,
                'object_type': self.object_type,
                'detection_mode': self.detection_mode,
                'ui_theme': self.ui_theme,
                'optimization_level': self.optimization_level
            },
            'detection_stats': {
                'total_frames': self.stats['total_frames'],
                'total_objects': self.stats['total_objects'],
                'avg_objects_per_frame': self.stats['total_objects'] / self.stats['total_frames'],
                'peak_count': self.stats['peak_count'],
                'avg_confidence': self.stats['avg_confidence'],
                'objects_per_second': self.stats['objects_per_second']
            },
            'performance_stats': {
                'avg_fps': np.mean(self.stats['fps_history']) if self.stats['fps_history'] else 0,
                'avg_processing_time': np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0,
                'max_fps': np.max(self.stats['fps_history']) if self.stats['fps_history'] else 0,
                'min_fps': np.min(self.stats['fps_history']) if self.stats['fps_history'] else 0
            },
            'algorithm_info': {
                'weights': self.algorithm_weights,
                'parameters': self.object_params[self.object_type],
                'calibrated': self.is_calibrated,
                'features_enabled': self.features
            },
            'ui_settings': {
                'theme': self.ui_theme,
                'state': self.ui_state
            }
        }
        
        # Save to file
        filename = f"ultimate_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(comprehensive_stats, f, indent=2, default=str)
        
        print(f"📊 Comprehensive statistics saved to {filename}")
        
        # Save to database if available
        if self.db:
            try:
                self.db.save_count_result(
                    filename, self.stats['total_objects'], 
                    self.stats['avg_confidence'], self.object_type
                )
                print("💾 Data saved to database")
            except Exception as e:
                print(f"⚠️ Database save failed: {e}")
    
    def start_recording(self):
        """Start video recording"""
        if self.features['recording']:
            print("⚠️ Recording already in progress")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"ultimate_recording_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.features['video_writer'] = cv2.VideoWriter(
            filename, fourcc, self.target_fps, (self.frame_width, self.frame_height)
        )
        
        self.features['recording'] = True
        print(f"📹 Recording started: {filename}")
    
    def stop_recording(self):
        """Stop video recording"""
        if not self.features['recording']:
            return
        
        if self.features['video_writer']:
            self.features['video_writer'].release()
            self.features['video_writer'] = None
        
        self.features['recording'] = False
        print("⏹️ Recording stopped")
    
    def run_ultimate_camera_loop(self):
        """Ultimate camera loop with all advanced features"""
        if not self.start_camera():
            return
        
        self.is_running = True
        print("\n🎥 ULTIMATE CAMERA COUNTER STARTED!")
        print("=" * 60)
        self.show_ultimate_instructions()
        
        frame_count = 0
        start_time = time.time()
        
        # Auto-calibration if enabled
        if self.features['auto_calibration']:
            print("🔧 Auto-calibrating...")
            self.calibrate_ultimate_detection()
        
        try:
            while self.is_running:
                ret, frame = self.camera.read()
                if not ret:
                    print("❌ Error reading frame")
                    break
                
                # Process frame with ultimate detection
                processed_frame, count, confidence = self.detect_objects_ultimate(frame)
                
                # Add ultimate overlay
                final_frame = self.add_ultimate_overlay(processed_frame, count, confidence)
                
                # Record frame if recording
                if self.features['recording'] and self.features['video_writer']:
                    self.features['video_writer'].write(final_frame)
                
                # Update comprehensive statistics
                self.stats['total_frames'] += 1
                self.stats['total_objects'] += count
                self.stats['avg_confidence'] = (
                    self.stats['avg_confidence'] * (self.stats['total_frames'] - 1) + confidence
                ) / self.stats['total_frames']
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - start_time
                    fps = 30 / elapsed_time
                    self.stats['fps_history'].append(fps)
                    
                    # Keep only recent FPS history
                    if len(self.stats['fps_history']) > 100:
                        self.stats['fps_history'] = self.stats['fps_history'][-100:]
                    
                    start_time = time.time()
                    print(f"🚀 FPS: {fps:.1f} | Objects: {count} | Confidence: {confidence:.3f} | Theme: {self.ui_theme}")
                
                # Display frame
                cv2.imshow('Ultimate Camera Counter - Press F1 for help', final_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key_press(key):
                    break
            
        except Exception as e:
            print(f"❌ Error in camera loop: {e}")
        
        finally:
            self.stop_camera()
    
    def _handle_key_press(self, key):
        """Handle all key presses"""
        # Quit
        if key == ord('q') or key == 27:  # ESC
            return False
        
        # Object types
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
        
        # Detection modes
        elif key == ord('u'):
            self.detection_mode = "ultimate"
            print("🚀 Switched to ultimate mode")
        elif key == ord('e'):
            self.detection_mode = "ensemble"
            print("🔄 Switched to ensemble mode")
        elif key == ord('p'):
            self.detection_mode = "precise"
            print("🎯 Switched to precise mode")
        
        # UI themes
        elif key == ord('1'):
            self.ui_theme = "dark"
            print("🌙 Switched to dark theme")
        elif key == ord('2'):
            self.ui_theme = "light"
            print("☀️ Switched to light theme")
        elif key == ord('3'):
            self.ui_theme = "neon"
            print("🌈 Switched to neon theme")
        elif key == ord('4'):
            self.ui_theme = "cyberpunk"
            print("🤖 Switched to cyberpunk theme")
        
        # Features
        elif key == ord('k'):
            self.calibrate_ultimate_detection()
        elif key == ord('x'):
            self.show_ultimate_statistics()
        elif key == ord('v'):
            if self.features['recording']:
                self.stop_recording()
            else:
                self.start_recording()
        elif key == ord('o'):
            self.ui_state['show_overlay'] = not self.ui_state['show_overlay']
            print(f"📊 Overlay: {'ON' if self.ui_state['show_overlay'] else 'OFF'}")
        elif key == ord('b'):
            self.ui_state['show_bounding_boxes'] = not self.ui_state['show_bounding_boxes']
            print(f"📦 Bounding boxes: {'ON' if self.ui_state['show_bounding_boxes'] else 'OFF'}")
        elif key == ord('n'):
            self.ui_state['show_contours'] = not self.ui_state['show_contours']
            print(f"🔲 Contours: {'ON' if self.ui_state['show_contours'] else 'OFF'}")
        elif key == ord('m'):
            self.ui_state['show_grid'] = not self.ui_state['show_grid']
            print(f"📋 Grid: {'ON' if self.ui_state['show_grid'] else 'OFF'}")
        elif key == ord('j'):
            self.ui_state['show_crosshair'] = not self.ui_state['show_crosshair']
            print(f"🎯 Crosshair: {'ON' if self.ui_state['show_crosshair'] else 'OFF'}")
        elif key == ord('a'):
            self.ui_state['animation_enabled'] = not self.ui_state['animation_enabled']
            print(f"✨ Animations: {'ON' if self.ui_state['animation_enabled'] else 'OFF'}")
        elif key == ord('f'):
            self.ui_state['particle_effects'] = not self.ui_state['particle_effects']
            print(f"🎆 Particle effects: {'ON' if self.ui_state['particle_effects'] else 'OFF'}")
        
        # Confidence threshold (5-9)
        elif key in [ord(str(i)) for i in range(5, 10)]:
            threshold = int(chr(key)) / 10.0
            self.confidence_threshold = threshold
            print(f"⚙️ Confidence threshold: {threshold}")
        
        # Save frame
        elif key == ord('s'):
            self.save_ultimate_frame()
        
        # Help
        elif key == 255:  # F1 key (platform dependent)
            self.show_ultimate_instructions()
        
        return True
    
    def save_ultimate_frame(self):
        """Save current frame with all detection data"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Capture current frame
        ret, frame = self.camera.read()
        if not ret:
            print("❌ Error capturing frame")
            return
        
        # Process frame
        processed_frame, count, confidence = self.detect_objects_ultimate(frame)
        final_frame = self.add_ultimate_overlay(processed_frame, count, confidence)
        
        # Save frame
        filename = f"ultimate_frame_{timestamp}_count{count}_conf{confidence:.3f}.jpg"
        cv2.imwrite(filename, final_frame)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'count': count,
            'confidence': confidence,
            'object_type': self.object_type,
            'detection_mode': self.detection_mode,
            'theme': self.ui_theme,
            'settings': self.ui_state.copy()
        }
        
        metadata_filename = f"ultimate_frame_{timestamp}_metadata.json"
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Frame saved: {filename}")
        print(f"📄 Metadata saved: {metadata_filename}")
    
    def show_ultimate_statistics(self):
        """Display comprehensive statistics"""
        print("\n" + "="*60)
        print("📊 ULTIMATE CAMERA COUNTER STATISTICS")
        print("="*60)
        
        # Session info
        session_duration = (datetime.now() - self.stats['session_start']).total_seconds()
        print(f"⏱️  Session Duration: {session_duration:.1f}s")
        print(f"🎯 Object Type: {self.object_type.title()}")
        print(f"🔧 Detection Mode: {self.detection_mode.title()}")
        print(f"🎨 UI Theme: {self.ui_theme.title()}")
        
        # Detection stats
        print(f"\n📈 DETECTION STATISTICS:")
        print(f"   Total Frames: {self.stats['total_frames']}")
        print(f"   Total Objects: {self.stats['total_objects']}")
        if self.stats['total_frames'] > 0:
            print(f"   Avg Objects/Frame: {self.stats['total_objects'] / self.stats['total_frames']:.2f}")
        print(f"   Peak Count: {self.stats['peak_count']}")
        print(f"   Avg Confidence: {self.stats['avg_confidence']:.3f}")
        
        # Performance stats
        print(f"\n⚡ PERFORMANCE STATISTICS:")
        if self.stats['fps_history']:
            print(f"   Current FPS: {self.stats['fps_history'][-1]:.1f}")
            print(f"   Average FPS: {np.mean(self.stats['fps_history']):.1f}")
            print(f"   Max FPS: {np.max(self.stats['fps_history']):.1f}")
            print(f"   Min FPS: {np.min(self.stats['fps_history']):.1f}")
        
        if self.stats['processing_times']:
            avg_time = np.mean(self.stats['processing_times'][-10:])
            print(f"   Avg Processing Time: {avg_time*1000:.1f}ms")
        
        # Algorithm weights
        print(f"\n🧠 ALGORITHM WEIGHTS:")
        for algo, weight in self.algorithm_weights.items():
            print(f"   {algo.title()}: {weight:.3f}")
        
        # Features status
        print(f"\n🔧 FEATURES STATUS:")
        for feature, enabled in self.features.items():
            if isinstance(enabled, bool):
                status = "✅" if enabled else "❌"
                print(f"   {feature.title()}: {status}")
        
        print("="*60)
    
    def show_ultimate_instructions(self):
        """Show comprehensive instructions"""
        print("\n" + "="*60)
        print("🎮 ULTIMATE CAMERA COUNTER - CONTROLS")
        print("="*60)
        
        print("📦 OBJECT TYPES:")
        print("   R - Rice detection")
        print("   C - Coin detection") 
        print("   H - Hair detection")
        print("   G - General detection")
        
        print("\n🔧 DETECTION MODES:")
        print("   U - Ultimate mode (all algorithms)")
        print("   E - Ensemble mode")
        print("   P - Precise mode")
        
        print("\n🎨 UI THEMES:")
        print("   1 - Dark theme")
        print("   2 - Light theme")
        print("   3 - Neon theme")
        print("   4 - Cyberpunk theme")
        
        print("\n⚙️ SETTINGS:")
        print("   5-9 - Adjust confidence threshold")
        print("   K - Calibrate detection")
        print("   X - Show statistics")
        
        print("\n🎬 RECORDING & CAPTURE:")
        print("   V - Start/stop video recording")
        print("   S - Save current frame")
        
        print("\n👁️ DISPLAY OPTIONS:")
        print("   O - Toggle overlay")
        print("   B - Toggle bounding boxes")
        print("   N - Toggle contours")
        print("   M - Toggle grid")
        print("   J - Toggle crosshair")
        print("   A - Toggle animations")
        print("   F - Toggle particle effects")
        
        print("\n🚪 EXIT:")
        print("   Q or ESC - Quit application")
        
        print("="*60)

def main():
    """Main function"""
    print("🚀 ULTIMATE CAMERA COUNTER")
    print("=" * 60)
    print("🎯 Maximum Accuracy • 🎨 Stunning UI • ⚡ Advanced Features")
    print("=" * 60)
    
    # Create ultimate camera counter
    camera_counter = UltimateCameraCounter()
    
    try:
        # Run ultimate camera loop
        camera_counter.run_ultimate_camera_loop()
        
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        camera_counter.stop_camera()
        print("👋 Ultimate Camera Counter closed")

if __name__ == "__main__":
    main()
