"""
Specialized Rice Detection Module
Advanced algorithms specifically optimized for rice grain detection.
"""

import cv2
import numpy as np
import time

class RiceDetector:
    """Specialized detector for rice grains with advanced algorithms"""
    
    def __init__(self):
        """Initialize rice detector with specialized parameters"""
        self.rice_params = {
            'min_area': 8,
            'max_area': 800,
            'min_circularity': 0.25,
            'max_circularity': 0.95,
            'min_aspect_ratio': 0.2,
            'max_aspect_ratio': 5.0,
            'blur_kernel': (3, 3),
            'bilateral_d': 9,
            'bilateral_sigma': 75,
            'adaptive_block': 11,
            'adaptive_c': 2,
            'morph_kernel': (2, 2),
            'erosion_iterations': 1,
            'dilation_iterations': 1
        }
        
        # Rice-specific color ranges (HSV)
        self.rice_color_ranges = [
            # Light rice (white/yellow)
            {'lower': np.array([0, 0, 180]), 'upper': np.array([180, 30, 255])},
            # Brown rice
            {'lower': np.array([10, 50, 50]), 'upper': np.array([20, 255, 255])},
            # Golden rice
            {'lower': np.array([15, 50, 50]), 'upper': np.array([25, 255, 255])}
        ]
    
    def detect_rice_advanced(self, frame):
        """Advanced rice detection using multiple algorithms"""
        try:
            # Method 1: Color-based detection
            color_contours = self._detect_by_color(frame)
            
            # Method 2: Shape-based detection
            shape_contours = self._detect_by_shape(frame)
            
            # Method 3: Texture-based detection
            texture_contours = self._detect_by_texture(frame)
            
            # Combine all detections
            all_contours = color_contours + shape_contours + texture_contours
            
            # Filter and merge overlapping detections
            valid_contours = self._filter_rice_contours(all_contours)
            
            # Create visualization
            processed_frame = frame.copy()
            for contour in valid_contours:
                confidence = self._calculate_rice_confidence(contour)
                color = self._get_rice_confidence_color(confidence)
                
                # Draw contour
                cv2.drawContours(processed_frame, [contour], -1, color, 2)
                
                # Add bounding box and confidence
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 1)
                cv2.putText(processed_frame, f"{confidence:.2f}", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Calculate overall confidence
            confidence = self._calculate_overall_rice_confidence(valid_contours)
            
            return processed_frame, len(valid_contours), confidence
            
        except Exception as e:
            print(f"Error in rice detection: {e}")
            return frame, 0, 0.0
    
    def _detect_by_color(self, frame):
        """Detect rice grains using color-based approach"""
        contours = []
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for color_range in self.rice_color_ranges:
            # Create mask for current color range
            mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            color_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(color_contours)
        
        return contours
    
    def _detect_by_shape(self, frame):
        """Detect rice grains using shape-based approach"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter for edge preservation
        bilateral = cv2.bilateralFilter(
            gray, self.rice_params['bilateral_d'], 
            self.rice_params['bilateral_sigma'], 
            self.rice_params['bilateral_sigma']
        )
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(bilateral, self.rice_params['blur_kernel'], 0)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            self.rice_params['adaptive_block'], self.rice_params['adaptive_c']
        )
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.rice_params['morph_kernel'])
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Additional erosion and dilation for rice grains
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.erode(cleaned, kernel_erode, iterations=self.rice_params['erosion_iterations'])
        cleaned = cv2.dilate(cleaned, kernel_erode, iterations=self.rice_params['dilation_iterations'])
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours
    
    def _detect_by_texture(self, frame):
        """Detect rice grains using texture-based approach"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gabor filter for texture detection
        kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 2*np.pi, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        
        # Apply threshold
        _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours
    
    def _filter_rice_contours(self, contours):
        """Filter contours specifically for rice grains"""
        valid_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.rice_params['min_area'] <= area <= self.rice_params['max_area']:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if (self.rice_params['min_circularity'] <= circularity <= self.rice_params['max_circularity']):
                        # Check aspect ratio for rice grains (can be elongated)
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = max(w, h) / max(1, min(w, h))
                        
                        if (self.rice_params['min_aspect_ratio'] <= aspect_ratio <= self.rice_params['max_aspect_ratio']):
                            # Check for overlap with existing contours
                            is_overlapping = False
                            for existing in valid_contours:
                                if self._contours_overlap_rice(contour, existing):
                                    is_overlapping = True
                                    break
                            
                            if not is_overlapping:
                                valid_contours.append(contour)
        
        return valid_contours
    
    def _contours_overlap_rice(self, contour1, contour2, threshold=0.3):
        """Check if two rice contours overlap (lower threshold for rice)"""
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
        
        # Check if overlap exceeds threshold (lower for rice)
        overlap_ratio = intersection_area / min(area1, area2)
        return overlap_ratio > threshold
    
    def _calculate_rice_confidence(self, contour):
        """Calculate confidence specifically for rice grains"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Get bounding rectangle for aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / max(1, min(w, h))
        
        # Normalize area confidence
        area_confidence = 1.0 - abs(area - (self.rice_params['min_area'] + self.rice_params['max_area']) / 2) / (self.rice_params['max_area'] - self.rice_params['min_area'])
        area_confidence = max(0.0, min(1.0, area_confidence))
        
        # Normalize circularity confidence
        circularity_confidence = (circularity - self.rice_params['min_circularity']) / (self.rice_params['max_circularity'] - self.rice_params['min_circularity'])
        circularity_confidence = max(0.0, min(1.0, circularity_confidence))
        
        # Normalize aspect ratio confidence
        aspect_confidence = 1.0 - abs(aspect_ratio - 2.0) / 3.0  # Optimal around 2:1
        aspect_confidence = max(0.0, min(1.0, aspect_confidence))
        
        # Combine all factors
        confidence = (circularity_confidence * 0.4 + area_confidence * 0.3 + aspect_confidence * 0.3)
        return max(0.0, min(1.0, confidence))
    
    def _calculate_overall_rice_confidence(self, contours):
        """Calculate overall confidence for rice detection"""
        if not contours:
            return 0.0
        
        total_confidence = 0.0
        for contour in contours:
            total_confidence += self._calculate_rice_confidence(contour)
        
        return total_confidence / len(contours)
    
    def _get_rice_confidence_color(self, confidence):
        """Get color based on rice detection confidence"""
        if confidence > 0.8:
            return (0, 255, 0)  # Green for high confidence
        elif confidence > 0.6:
            return (0, 255, 255)  # Yellow for medium confidence
        elif confidence > 0.4:
            return (0, 165, 255)  # Orange for low-medium confidence
        else:
            return (0, 0, 255)  # Red for low confidence
    
    def calibrate_rice_detection(self, frame):
        """Calibrate rice detection parameters based on current scene"""
        print("🔧 Calibrating rice detection...")
        
        # Analyze frame characteristics
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        avg_contrast = np.std(gray)
        
        # Adjust parameters based on lighting
        if avg_brightness < 100:  # Low light
            self.rice_params['adaptive_c'] += 1
            self.rice_params['bilateral_sigma'] += 25
            print("🌙 Adjusted for low light conditions")
        elif avg_brightness > 200:  # Bright light
            self.rice_params['adaptive_c'] -= 1
            self.rice_params['bilateral_sigma'] -= 25
            print("☀️ Adjusted for bright light conditions")
        
        if avg_contrast < 30:  # Low contrast
            self.rice_params['bilateral_sigma'] += 50
            print("📊 Adjusted for low contrast")
        
        print("✅ Rice detection calibration completed")

def test_rice_detector():
    """Test the rice detector with sample image"""
    detector = RiceDetector()
    
    # Create a test image with rice-like objects
    test_image = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Draw some rice-like ellipses
    cv2.ellipse(test_image, (150, 100), (20, 8), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(test_image, (200, 120), (15, 6), 45, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(test_image, (250, 80), (18, 7), -30, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(test_image, (300, 150), (12, 5), 60, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(test_image, (350, 90), (16, 6), -15, 0, 360, (255, 255, 255), -1)
    
    # Add some noise
    noise = np.random.normal(0, 10, test_image.shape).astype(np.uint8)
    test_image = cv2.add(test_image, noise)
    
    # Detect rice
    processed_frame, count, confidence = detector.detect_rice_advanced(test_image)
    
    print(f"🍚 Rice Detection Test Results:")
    print(f"Detected rice grains: {count}")
    print(f"Confidence: {confidence:.3f}")
    
    # Save test result
    cv2.imwrite("rice_detection_test.jpg", processed_frame)
    print("💾 Test result saved as 'rice_detection_test.jpg'")

if __name__ == "__main__":
    test_rice_detector()
