"""
Advanced Camera Counter - Enhanced Version
Real-time object counting with improved accuracy and advanced features.
"""

import cv2
import numpy as np
import time
import os
import json
from datetime import datetime

class AdvancedCameraCounter:
    """Advanced real-time object counting with enhanced accuracy"""
    
    def __init__(self):
        """Initialize advanced camera counter"""
        self.camera = None
        self.is_running = False
        
        # Detection settings
        self.object_type = "general"
        self.confidence_threshold = 0.3
        self.detection_mode = "adaptive"  # adaptive, fixed, multi_scale
        
        # Camera settings
        self.camera_index = 0
        self.frame_width = 640
        self.frame_height = 480
        
        # Advanced detection parameters
        self.object_params = {
            'rice': {
                'min_area': 10, 'max_area': 500, 'min_circularity': 0.3,
                'blur_kernel': (3, 3), 'morph_kernel': (2, 2),
                'adaptive_block': 11, 'adaptive_c': 2,
                'bilateral_d': 9, 'bilateral_sigma': 75
            },
            'coins': {
                'min_area': 100, 'max_area': 10000, 'min_circularity': 0.6,
                'blur_kernel': (5, 5), 'morph_kernel': (3, 3),
                'adaptive_block': 15, 'adaptive_c': 3,
                'bilateral_d': 15, 'bilateral_sigma': 100
            },
            'hair': {
                'min_area': 5, 'max_area': 300, 'min_circularity': 0.2,
                'blur_kernel': (3, 3), 'morph_kernel': (2, 2),
                'adaptive_block': 9, 'adaptive_c': 2,
                'bilateral_d': 5, 'bilateral_sigma': 50
            },
            'general': {
                'min_area': 20, 'max_area': 3000, 'min_circularity': 0.3,
                'blur_kernel': (5, 5), 'morph_kernel': (3, 3),
                'adaptive_block': 11, 'adaptive_c': 2,
                'bilateral_d': 9, 'bilateral_sigma': 75
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
    
    def detect_objects_advanced(self, frame):
        """Advanced object detection with multiple algorithms"""
        try:
            # Get parameters for current object type
            params = self.object_params.get(self.object_type, self.object_params['general'])
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter for edge preservation
            bilateral = cv2.bilateralFilter(
                gray, params['bilateral_d'], params['bilateral_sigma'], params['bilateral_sigma']
            )
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(bilateral, params['blur_kernel'], 0)
            
            # Multi-scale detection for better accuracy
            all_contours = []
            processed_frame = frame.copy()
            
            # Scale 1: Original size
            contours1 = self._detect_at_scale(blurred, params, 1.0)
            all_contours.extend(contours1)
            
            # Scale 2: 1.5x larger (for smaller objects)
            if self.detection_mode == "multi_scale":
                h, w = blurred.shape
                scale_factor = 1.5
                scaled = cv2.resize(blurred, (int(w * scale_factor), int(h * scale_factor)))
                contours2 = self._detect_at_scale(scaled, params, 1.0 / scale_factor)
                all_contours.extend(contours2)
            
            # Filter and merge contours
            valid_contours = self._filter_contours(all_contours, params)
            
            # Draw contours with enhanced visualization
            for contour in valid_contours:
                color = self._get_enhanced_confidence_color(contour, params)
                cv2.drawContours(processed_frame, [contour], -1, color, 2)
                
                # Add bounding box and confidence
                x, y, w, h = cv2.boundingRect(contour)
                confidence = self._calculate_contour_confidence(contour, params)
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 1)
                cv2.putText(processed_frame, f"{confidence:.2f}", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Calculate overall confidence
            confidence = self._calculate_overall_confidence(valid_contours, params)
            
            return processed_frame, len(valid_contours), confidence
            
        except Exception as e:
            print(f"Error in advanced object detection: {e}")
            return frame, 0, 0.0
    
    def _detect_at_scale(self, image, params, scale_factor):
        """Detect objects at a specific scale"""
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 
            params['adaptive_block'], params['adaptive_c']
        )
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, params['morph_kernel'])
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Scale contours back to original size
        scaled_contours = []
        for contour in contours:
            scaled_contour = contour.astype(np.float32) * scale_factor
            scaled_contours.append(scaled_contour.astype(np.int32))
        
        return scaled_contours
    
    def _filter_contours(self, contours, params):
        """Enhanced contour filtering with overlap detection"""
        valid_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if params['min_area'] <= area <= params['max_area']:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity >= params['min_circularity']:
                        # Check for overlap with existing contours
                        is_overlapping = False
                        for existing in valid_contours:
                            if self._contours_overlap(contour, existing):
                                is_overlapping = True
                                break
                        
                        if not is_overlapping:
                            valid_contours.append(contour)
        
        return valid_contours
    
    def _contours_overlap(self, contour1, contour2, threshold=0.5):
        """Check if two contours overlap significantly"""
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
        
        # Check if overlap exceeds threshold
        overlap_ratio = intersection_area / min(area1, area2)
        return overlap_ratio > threshold
    
    def _get_enhanced_confidence_color(self, contour, params):
        """Get color based on enhanced confidence calculation"""
        confidence = self._calculate_contour_confidence(contour, params)
        
        if confidence > 0.8:
            return (0, 255, 0)  # Green for high confidence
        elif confidence > 0.6:
            return (0, 255, 255)  # Yellow for medium confidence
        elif confidence > 0.4:
            return (0, 165, 255)  # Orange for low-medium confidence
        else:
            return (0, 0, 255)  # Red for low confidence
    
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
    
    def _calculate_overall_confidence(self, contours, params):
        """Calculate overall confidence for all contours"""
        if not contours:
            return 0.0
        
        total_confidence = 0.0
        for contour in contours:
            total_confidence += self._calculate_contour_confidence(contour, params)
        
        return total_confidence / len(contours)
    
    def add_enhanced_overlay(self, frame, count, confidence):
        """Add enhanced information overlay to the frame"""
        overlay = frame.copy()
        
        # Add background rectangle with gradient effect
        cv2.rectangle(overlay, (10, 10), (400, 160), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (400, 160), (255, 255, 255), 2)
        
        # Add text information with better formatting
        cv2.putText(overlay, f"Objects: {count}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(overlay, f"Confidence: {confidence:.3f}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(overlay, f"Type: {self.object_type.title()}", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, f"Mode: {self.detection_mode.title()}", (20, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return overlay
    
    def calibrate_detection(self):
        """Calibrate detection parameters based on current scene"""
        print("🔧 Starting calibration...")
        
        # Capture multiple frames for calibration
        calibration_frames = []
        for i in range(10):
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
        if avg_brightness < 100:  # Low light
            self.object_params[self.object_type]['adaptive_c'] += 1
            print("🌙 Adjusted for low light conditions")
        elif avg_brightness > 200:  # Bright light
            self.object_params[self.object_type]['adaptive_c'] -= 1
            print("☀️ Adjusted for bright light conditions")
        
        if avg_contrast < 30:  # Low contrast
            self.object_params[self.object_type]['bilateral_sigma'] += 25
            print("📊 Adjusted for low contrast")
        
        self.is_calibrated = True
        print("✅ Calibration completed")
        return True
    
    def save_statistics(self):
        """Save detection statistics to file"""
        if self.stats['total_frames'] > 0:
            stats_data = {
                'timestamp': datetime.now().isoformat(),
                'object_type': self.object_type,
                'total_frames': self.stats['total_frames'],
                'total_objects': self.stats['total_objects'],
                'avg_objects_per_frame': self.stats['total_objects'] / self.stats['total_frames'],
                'avg_confidence': self.stats['avg_confidence'],
                'avg_fps': np.mean(self.stats['fps_history']) if self.stats['fps_history'] else 0
            }
            
            filename = f"detection_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(stats_data, f, indent=2)
            
            print(f"📊 Statistics saved to {filename}")
    
    def run_camera_loop(self):
        """Main camera loop with enhanced features"""
        if not self.start_camera():
            return
        
        self.is_running = True
        print("\n🎥 Advanced Camera Counter Started!")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 'r' for rice detection")
        print("- Press 'c' for coin detection")
        print("- Press 'h' for hair detection")
        print("- Press 'g' for general detection")
        print("- Press '1-9' to adjust confidence threshold")
        print("- Press 's' to save current frame")
        print("- Press 'i' for instructions")
        print("- Press 'a' for adaptive mode")
        print("- Press 'm' for multi-scale mode")
        print("- Press 'f' for fixed mode")
        print("- Press 'k' to calibrate detection")
        print("- Press 't' to show statistics")
        
        frame_count = 0
        start_time = time.time()
        
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                print("❌ Error reading frame")
                break
            
            # Process frame
            processed_frame, count, confidence = self.detect_objects_advanced(frame)
            
            # Add overlay
            final_frame = self.add_enhanced_overlay(processed_frame, count, confidence)
            
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
            cv2.imshow('Advanced Camera Counter', final_frame)
            
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
            elif key == ord('a'):
                self.detection_mode = "adaptive"
                print("🔄 Switched to adaptive mode")
            elif key == ord('m'):
                self.detection_mode = "multi_scale"
                print("🔍 Switched to multi-scale mode")
            elif key == ord('f'):
                self.detection_mode = "fixed"
                print("📏 Switched to fixed mode")
            elif key == ord('k'):
                self.calibrate_detection()
            elif key == ord('t'):
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
        print("\n📊 Current Statistics:")
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
        filename = f"advanced_capture_{timestamp}_count{count}_conf{confidence:.3f}.jpg"
        
        try:
            cv2.imwrite(filename, frame)
            print(f"💾 Frame saved: {filename}")
        except Exception as e:
            print(f"❌ Error saving frame: {e}")
    
    def show_instructions(self):
        """Show detailed instructions"""
        print("\n📖 Advanced Instructions:")
        print("1. Point your camera at objects to count")
        print("2. Detection Modes:")
        print("   - Adaptive: Automatic parameter adjustment")
        print("   - Multi-scale: Detects objects at different scales")
        print("   - Fixed: Uses predefined parameters")
        print("3. Object Types:")
        print("   - Rice: Small, round grains (enhanced for rice)")
        print("   - Coins: Circular objects (high circularity)")
        print("   - Hair: Thin strands (low circularity)")
        print("   - General: Any small objects")
        print("4. Press 'k' to calibrate for current lighting")
        print("5. Press 't' to view real-time statistics")
        print("6. Adjust confidence threshold (1-9) for sensitivity")
        print("7. Color coding: Green=high, Yellow=medium, Orange=low, Red=very low")

def main():
    """Main function"""
    print("🔢 Advanced Camera Counter")
    print("=" * 50)
    
    # Create camera counter
    camera_counter = AdvancedCameraCounter()
    
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
