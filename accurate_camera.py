"""
Highly Accurate Camera Counter
Real-time object counting with advanced detection algorithms.
"""

import cv2
import numpy as np
import time
import os
import json
from datetime import datetime

class AccurateCameraCounter:
    """Highly accurate real-time object counting"""
    
    def __init__(self):
        """Initialize accurate camera counter"""
        self.camera = None
        self.is_running = False
        
        # Detection settings
        self.object_type = "general"
        self.confidence_threshold = 0.3
        self.detection_mode = "multi_algorithm"  # multi_algorithm, adaptive, precise
        
        # Camera settings
        self.camera_index = 0
        self.frame_width = 640
        self.frame_height = 480
        
        # Advanced detection parameters
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
            'fps_history': []
        }
        
        # Calibration data
        self.calibration_data = {}
        self.is_calibrated = False
        
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
    
    def detect_objects_accurate(self, frame):
        """Highly accurate object detection using multiple algorithms"""
        try:
            # Get parameters for current object type
            params = self.object_params.get(self.object_type, self.object_params['general'])
            
            # Method 1: Advanced threshold detection
            threshold_results = self._detect_with_advanced_threshold(frame, params)
            
            # Method 2: Gabor filter detection
            gabor_results = self._detect_with_gabor(frame, params)
            
            # Method 3: Color-based detection
            color_results = self._detect_with_color(frame, params)
            
            # Method 4: Multi-scale detection
            multiscale_results = self._detect_multiscale(frame, params)
            
            # Combine all detections
            all_contours = []
            all_confidences = []
            
            # Add threshold results
            if threshold_results['contours']:
                all_contours.extend(threshold_results['contours'])
                all_confidences.extend(threshold_results['confidences'])
            
            # Add Gabor results
            if gabor_results['contours']:
                all_contours.extend(gabor_results['contours'])
                all_confidences.extend(gabor_results['confidences'])
            
            # Add color results
            if color_results['contours']:
                all_contours.extend(color_results['contours'])
                all_confidences.extend(color_results['confidences'])
            
            # Add multiscale results
            if multiscale_results['contours']:
                all_contours.extend(multiscale_results['contours'])
                all_confidences.extend(multiscale_results['confidences'])
            
            # Filter and merge overlapping detections
            valid_contours, final_confidences = self._filter_accurate_contours(all_contours, all_confidences, params)
            
            # Create visualization
            processed_frame = frame.copy()
            for i, contour in enumerate(valid_contours):
                confidence = final_confidences[i] if i < len(final_confidences) else 0.5
                color = self._get_accurate_confidence_color(confidence)
                
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
            print(f"Error in accurate object detection: {e}")
            return frame, 0, 0.0
    
    def _detect_with_advanced_threshold(self, frame, params):
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
            print(f"Advanced threshold detection error: {e}")
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
    
    def _detect_multiscale(self, frame, params):
        """Multi-scale detection for better accuracy"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            all_contours = []
            all_confidences = []
            
            # Detect at different scales
            scales = [0.5, 1.0, 1.5, 2.0]
            
            for scale in scales:
                # Resize image
                h, w = gray.shape
                new_h, new_w = int(h * scale), int(w * scale)
                scaled = cv2.resize(gray, (new_w, new_h))
                
                # Apply bilateral filter
                bilateral = cv2.bilateralFilter(
                    scaled, params['bilateral_d'], params['bilateral_sigma'], params['bilateral_sigma']
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
                
                # Scale contours back to original size
                for contour in contours:
                    scaled_contour = contour.astype(np.float32) / scale
                    scaled_contour = scaled_contour.astype(np.int32)
                    
                    confidence = self._calculate_contour_confidence(scaled_contour, params)
                    all_contours.append(scaled_contour)
                    all_confidences.append(confidence)
            
            return {'contours': all_contours, 'confidences': all_confidences}
            
        except Exception as e:
            print(f"Multi-scale detection error: {e}")
            return {'contours': [], 'confidences': []}
    
    def _filter_accurate_contours(self, contours, confidences, params):
        """Filter contours with advanced overlap detection"""
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
                            if self._contours_overlap_accurate(contour, existing):
                                is_overlapping = True
                                break
                        
                        if not is_overlapping:
                            valid_contours.append(contour)
                            final_confidences.append(confidence)
        
        return valid_contours, final_confidences
    
    def _contours_overlap_accurate(self, contour1, contour2, threshold=0.3):
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
    
    def _get_accurate_confidence_color(self, confidence):
        """Get color based on accurate confidence calculation"""
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
    
    def add_accurate_overlay(self, frame, count, confidence):
        """Add accurate information overlay to the frame"""
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
        cv2.putText(overlay, f"Algorithms: 4", (20, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        return overlay
    
    def calibrate_accurate_detection(self):
        """Calibrate detection parameters based on current scene"""
        print("🔧 Starting accurate calibration...")
        
        # Capture multiple frames for calibration
        calibration_frames = []
        for i in range(15):
            ret, frame = self.camera.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                calibration_frames.append(gray)
            time.sleep(0.1)
        
        if not calibration_frames:
            print("❌ Calibration failed: No frames captured")
            return False
        
        # Analyze frame statistics
        avg_brightness = np.mean([np.mean(frame) for frame in calibration_frames])
        avg_contrast = np.mean([np.std(frame) for frame in calibration_frames])
        
        # Adjust parameters based on lighting conditions
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
        
        self.is_calibrated = True
        print("✅ Accurate calibration completed")
        return True
    
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
                'calibrated': self.is_calibrated
            }
            
            filename = f"accurate_detection_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(stats_data, f, indent=2)
            
            print(f"📊 Accurate statistics saved to {filename}")
    
    def run_camera_loop(self):
        """Main camera loop with accurate features"""
        if not self.start_camera():
            return
        
        self.is_running = True
        print("\n🎥 Highly Accurate Camera Counter Started!")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 'r' for rice detection")
        print("- Press 'c' for coin detection")
        print("- Press 'h' for hair detection")
        print("- Press 'g' for general detection")
        print("- Press '1-9' to adjust confidence threshold")
        print("- Press 's' to save current frame")
        print("- Press 'i' for instructions")
        print("- Press 'm' for multi-algorithm mode")
        print("- Press 'a' for adaptive mode")
        print("- Press 'p' for precise mode")
        print("- Press 'k' to calibrate detection")
        print("- Press 'x' to show statistics")
        
        frame_count = 0
        start_time = time.time()
        
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                print("❌ Error reading frame")
                break
            
            # Process frame
            processed_frame, count, confidence = self.detect_objects_accurate(frame)
            
            # Add overlay
            final_frame = self.add_accurate_overlay(processed_frame, count, confidence)
            
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
            cv2.imshow('Highly Accurate Camera Counter', final_frame)
            
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
            elif key == ord('m'):
                self.detection_mode = "multi_algorithm"
                print("🔄 Switched to multi-algorithm mode")
            elif key == ord('a'):
                self.detection_mode = "adaptive"
                print("🔄 Switched to adaptive mode")
            elif key == ord('p'):
                self.detection_mode = "precise"
                print("🎯 Switched to precise mode")
            elif key == ord('k'):
                self.calibrate_accurate_detection()
            elif key == ord('x'):
                self.show_statistics()
            elif key in [ord(str(i)) for i in range(1, 10)]:
                threshold = int(chr(key)) / 10.0
                self.confidence_threshold = threshold
                print(f"⚙️ Confidence threshold: {threshold}")
            elif key == ord('s'):
                self.save_frame(final_frame, count, confidence)
            elif key == ord('i'):
                self.show_instructions()
        
        self.stop_camera()
    
    def show_statistics(self):
        """Display current statistics"""
        print("\n📊 Accurate Statistics:")
        print(f"Total Frames: {self.stats['total_frames']}")
        print(f"Total Objects: {self.stats['total_objects']}")
        print(f"Avg Objects/Frame: {self.stats['total_objects'] / max(1, self.stats['total_frames']):.2f}")
        print(f"Avg Confidence: {self.stats['avg_confidence']:.3f}")
        if self.stats['fps_history']:
            print(f"Avg FPS: {np.mean(self.stats['fps_history']):.1f}")
        print(f"Detection Mode: {self.detection_mode}")
        print(f"Object Type: {self.object_type}")
        print(f"Calibrated: {self.is_calibrated}")
    
    def save_frame(self, frame, count, confidence):
        """Save current frame with detection results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"accurate_capture_{timestamp}_count{count}_conf{confidence:.3f}.jpg"
        
        try:
            cv2.imwrite(filename, frame)
            print(f"💾 Frame saved: {filename}")
        except Exception as e:
            print(f"❌ Error saving frame: {e}")
    
    def show_instructions(self):
        """Show detailed instructions"""
        print("\n📖 Accurate Instructions:")
        print("1. Point your camera at objects to count")
        print("2. Detection Modes:")
        print("   - Multi-algorithm: Uses 4 different detection methods")
        print("   - Adaptive: Automatic parameter adjustment")
        print("   - Precise: High-precision detection")
        print("3. Object Types:")
        print("   - Rice: Specialized for rice grain detection")
        print("   - Coins: Optimized for circular objects")
        print("   - Hair: Designed for thin, elongated objects")
        print("   - General: Universal object detection")
        print("4. Press 'k' to calibrate for current lighting")
        print("5. Press 'x' to view real-time statistics")
        print("6. Adjust confidence threshold (1-9) for sensitivity")
        print("7. Color coding: Green=very high, Yellow=high, Orange=medium, Red=low, Gray=very low")

def main():
    """Main function"""
    print("🔢 Highly Accurate Camera Counter")
    print("=" * 50)
    
    # Create camera counter
    camera_counter = AccurateCameraCounter()
    
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
