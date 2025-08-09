"""
Image Processing Module for CountAnything AI
Handles image preprocessing, object detection, and counting using OpenCV and YOLOv8.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import os
from typing import Dict, Any, Optional, Tuple

class ImageProcessor:
    """Handles image processing and object counting using YOLOv8 and OpenCV"""
    
    def __init__(self):
        """Initialize the image processor with YOLOv8 model"""
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the YOLOv8 model for object detection"""
        try:
            # Load YOLOv8n model (nano version for speed)
            self.model = YOLO('yolov8n.pt')
            print("YOLOv8 model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            # Fallback to a basic model or handle gracefully
            self.model = None
    
    def preprocess_image(self, image_file) -> np.ndarray:
        """
        Preprocess uploaded image for better object detection
        
        Args:
            image_file: Uploaded file from Streamlit
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Convert uploaded file to PIL Image
            pil_image = Image.open(image_file)
            
            # Convert to numpy array
            image = np.array(pil_image)
            
            # Convert RGB to BGR (OpenCV format)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            raise
    
    def detect_objects_with_threshold(self, image: np.ndarray, object_type: str = "general", 
                                    confidence_threshold: float = 0.3) -> Tuple[int, float, np.ndarray]:
        """
        Detect objects using threshold-based methods for better counting accuracy
        
        Args:
            image: Input image as numpy array
            object_type: Type of object to detect
            confidence_threshold: Confidence threshold for detection
            
        Returns:
            Tuple of (count, confidence, processed_image)
        """
        try:
            # Special handling for rice detection
            if object_type == "rice":
                return self._detect_rice_grains(image, confidence_threshold)
            
            # Get object-specific parameters
            params = self._get_object_params(object_type)
            
            # Apply threshold-based detection
            count, confidence, processed_image = self._threshold_based_detection(
                image, params, confidence_threshold
            )
            
            return count, confidence, processed_image
            
        except Exception as e:
            print(f"Error in threshold-based detection: {e}")
            return self._basic_blob_detection(image)
    
    def _detect_rice_grains(self, image: np.ndarray, confidence_threshold: float) -> Tuple[int, float, np.ndarray]:
        """
        Specialized rice grain detection with improved accuracy
        
        Args:
            image: Input image
            confidence_threshold: Detection confidence threshold
            
        Returns:
            Tuple of (count, confidence, processed_image)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to preserve edges while reducing noise
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive threshold with different parameters for rice
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
        )
        
        # Morphological operations specifically for rice grains
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Remove small noise
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Close small gaps in rice grains
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours for rice grains
        valid_contours = []
        processed_image = image.copy()
        
        # Rice grain parameters
        min_area = 15  # Smaller minimum for rice grains
        max_area = 300  # Smaller maximum for rice grains
        min_circularity = 0.4  # Lower circularity for rice grains
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Check area constraints for rice
            if min_area <= area <= max_area:
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Check circularity constraint for rice
                    if circularity >= min_circularity:
                        # Additional rice-specific checks
                        # Check aspect ratio (rice grains are usually elongated)
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = float(w) / h if h > 0 else 0
                        
                        # Rice grains typically have aspect ratio between 1.0 and 3.0
                        if 0.3 <= aspect_ratio <= 3.0:
                            valid_contours.append(contour)
                            
                            # Draw contour with confidence color
                            color = self._get_confidence_color(circularity)
                            cv2.drawContours(processed_image, [contour], -1, color, 2)
                            
                            # Add label with area
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                cv2.putText(processed_image, f"{area}", (cx-10, cy),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Calculate confidence based on contour quality
        confidence = self._calculate_rice_confidence(valid_contours)
        
        return len(valid_contours), confidence, processed_image
    
    def _calculate_rice_confidence(self, contours: list) -> float:
        """Calculate confidence specifically for rice grains"""
        if not contours:
            return 0.0
        
        total_confidence = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                # Rice-specific confidence calculation
                confidence = min(circularity / 0.4, 1.0)  # Normalize to rice circularity
                total_confidence += confidence
        
        return total_confidence / len(contours)
    
    def _get_object_params(self, object_type: str) -> Dict:
        """Get parameters for different object types"""
        params = {
            'rice': {
                'min_area': 20,
                'max_area': 500,
                'min_circularity': 0.6,
                'color_lower': np.array([0, 0, 0]),
                'color_upper': np.array([180, 255, 255]),
                'blur_kernel': (5, 5),
                'morph_kernel': (3, 3)
            },
            'coins': {
                'min_area': 100,
                'max_area': 5000,
                'min_circularity': 0.7,
                'color_lower': np.array([0, 0, 0]),
                'color_upper': np.array([180, 255, 255]),
                'blur_kernel': (7, 7),
                'morph_kernel': (5, 5)
            },
            'hair': {
                'min_area': 10,
                'max_area': 200,
                'min_circularity': 0.3,
                'color_lower': np.array([0, 0, 0]),
                'color_upper': np.array([180, 100, 100]),
                'blur_kernel': (3, 3),
                'morph_kernel': (2, 2)
            },
            'general': {
                'min_area': 50,
                'max_area': 2000,
                'min_circularity': 0.4,
                'color_lower': np.array([0, 0, 0]),
                'color_upper': np.array([180, 255, 255]),
                'blur_kernel': (5, 5),
                'morph_kernel': (3, 3)
            }
        }
        return params.get(object_type, params['general'])
    
    def _threshold_based_detection(self, image: np.ndarray, params: Dict, 
                                 confidence_threshold: float) -> Tuple[int, float, np.ndarray]:
        """
        Advanced threshold-based object detection
        
        Args:
            image: Input image
            params: Object-specific parameters
            confidence_threshold: Detection confidence threshold
            
        Returns:
            Tuple of (count, confidence, processed_image)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, params['blur_kernel'], 0)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, params['morph_kernel'])
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on parameters
        valid_contours = []
        processed_image = image.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Check area constraints
            if params['min_area'] <= area <= params['max_area']:
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Check circularity constraint
                    if circularity >= params['min_circularity']:
                        valid_contours.append(contour)
                        
                        # Draw contour with confidence color
                        color = self._get_confidence_color(circularity)
                        cv2.drawContours(processed_image, [contour], -1, color, 2)
                        
                        # Add label with area
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.putText(processed_image, f"{area}", (cx-10, cy),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Calculate confidence based on contour quality
        confidence = self._calculate_confidence(valid_contours, params)
        
        return len(valid_contours), confidence, processed_image
    
    def _get_confidence_color(self, circularity: float) -> Tuple[int, int, int]:
        """Get color based on circularity (confidence)"""
        if circularity > 0.8:
            return (0, 255, 0)  # Green for high confidence
        elif circularity > 0.6:
            return (0, 255, 255)  # Yellow for medium confidence
        else:
            return (0, 0, 255)  # Red for low confidence
    
    def _calculate_confidence(self, contours: list, params: Dict) -> float:
        """Calculate overall confidence based on contour quality"""
        if not contours:
            return 0.0
        
        total_confidence = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                # Normalize circularity to confidence
                confidence = min(circularity / params['min_circularity'], 1.0)
                total_confidence += confidence
        
        return total_confidence / len(contours)
    
    def detect_objects(self, image: np.ndarray, object_type: str = "general", 
                      confidence_threshold: float = 0.3) -> Tuple[int, float, np.ndarray]:
        """
        Main detection method with improved threshold-based approach
        
        Args:
            image: Input image as numpy array
            object_type: Type of object to detect
            confidence_threshold: Detection confidence threshold
            
        Returns:
            Tuple of (count, confidence, processed_image)
        """
        # Use threshold-based detection for better accuracy
        return self.detect_objects_with_threshold(image, object_type, confidence_threshold)
    
    def _basic_blob_detection(self, image: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Basic blob detection as fallback
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (count, confidence, processed_image)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to create binary image
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            min_area = 50
            max_area = 10000
            valid_contours = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    valid_contours.append(contour)
            
            # Draw contours on image
            processed_image = image.copy()
            cv2.drawContours(processed_image, valid_contours, -1, (0, 255, 0), 2)
            
            count = len(valid_contours)
            confidence = 0.5  # Default confidence for basic detection
            
            return count, confidence, processed_image
            
        except Exception as e:
            print(f"Error in basic blob detection: {e}")
            return 0, 0.0, image
    
    def process_image(self, image_file, object_type: str = "general", 
                     confidence_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Main method to process an uploaded image and count objects
        
        Args:
            image_file: Uploaded file from Streamlit
            object_type: Type of object to detect
            confidence_threshold: Detection confidence threshold
            
        Returns:
            Dictionary with count, confidence, and processed image
        """
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image_file)
            
            # Detect objects with threshold-based method
            count, confidence, result_image = self.detect_objects(
                processed_image, object_type, confidence_threshold
            )
            
            # Convert result image back to RGB for Streamlit
            if result_image is not None:
                result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                result_pil = Image.fromarray(result_image_rgb)
            else:
                result_pil = None
            
            return {
                'count': count,
                'confidence': confidence,
                'processed_image': result_pil
            }
            
        except Exception as e:
            print(f"Error processing image: {e}")
            raise
    
    def get_contextual_insight(self, count: int, object_type: str = "objects") -> str:
        """
        Generate contextual insights based on the count
        
        Args:
            count: Number of objects detected
            object_type: Type of objects being counted
            
        Returns:
            Contextual insight string
        """
        if count == 0:
            return "No objects detected. Try adjusting the confidence threshold or image quality."
        elif count < 10:
            return f"Found {count} {object_type}. This is a small quantity."
        elif count < 100:
            return f"Found {count} {object_type}. This is a moderate quantity."
        elif count < 1000:
            return f"Found {count} {object_type}. This is a large quantity."
        else:
            return f"Found {count} {object_type}. This is a very large quantity!"
