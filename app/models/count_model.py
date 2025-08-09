"""
YOLOv8 Model Integration for CountAnything AI
Specialized model handling for different object types (rice, coins, hair, etc.)
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
import os

class CountModel:
    """Specialized YOLOv8 model for object counting"""
    
    def __init__(self, model_type: str = "yolov8n"):
        """
        Initialize the counting model
        
        Args:
            model_type: Type of YOLOv8 model to use (nano, small, medium, large)
        """
        self.model = None
        self.model_type = model_type
        self.load_model()
        
        # Define object types and their characteristics
        self.object_types = {
            'rice': {'min_size': 5, 'max_size': 50, 'color_range': [(0, 0, 0), (255, 255, 255)]},
            'coins': {'min_size': 20, 'max_size': 200, 'color_range': [(0, 0, 0), (255, 255, 255)]},
            'hair': {'min_size': 2, 'max_size': 20, 'color_range': [(0, 0, 0), (100, 100, 100)]},
            'general': {'min_size': 10, 'max_size': 100, 'color_range': [(0, 0, 0), (255, 255, 255)]}
        }
    
    def load_model(self):
        """Load the YOLOv8 model"""
        try:
            model_path = f"{self.model_type}.pt"
            self.model = YOLO(model_path)
            print(f"YOLOv8 {self.model_type} model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            self.model = None
    
    def detect_specific_objects(self, image: np.ndarray, object_type: str = "general") -> Tuple[int, float, np.ndarray]:
        """
        Detect specific types of objects with optimized parameters
        
        Args:
            image: Input image as numpy array
            object_type: Type of object to detect (rice, coins, hair, general)
            
        Returns:
            Tuple of (count, confidence, processed_image)
        """
        if self.model is None:
            return self._fallback_detection(image, object_type)
        
        try:
            # Get object-specific parameters
            params = self.object_types.get(object_type, self.object_types['general'])
            
            # Run detection with optimized confidence threshold
            results = self.model(image, conf=0.3, iou=0.5)
            
            count = 0
            confidence = 0.0
            processed_image = image.copy()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    # Filter boxes by size and other criteria
                    valid_boxes = self._filter_boxes_by_type(boxes, params)
                    count += len(valid_boxes)
                    
                    if len(valid_boxes) > 0:
                        confidences = [box.conf[0].cpu().numpy() for box in valid_boxes]
                        confidence = float(np.mean(confidences))
                    
                    # Draw filtered boxes
                    for box in valid_boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        # Draw rectangle with object type color
                        color = self._get_object_color(object_type)
                        cv2.rectangle(processed_image, 
                                    (int(x1), int(y1)), 
                                    (int(x2), int(y2)), 
                                    color, 2)
                        
                        # Add label
                        label = f"{object_type}: {conf:.2f}"
                        cv2.putText(processed_image, 
                                  label, 
                                  (int(x1), int(y1) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, 
                                  color, 
                                  2)
            
            return count, confidence, processed_image
            
        except Exception as e:
            print(f"Error in specific object detection: {e}")
            return self._fallback_detection(image, object_type)
    
    def _filter_boxes_by_type(self, boxes, params: Dict) -> List:
        """
        Filter detection boxes based on object type parameters
        
        Args:
            boxes: YOLOv8 detection boxes
            params: Object type parameters
            
        Returns:
            Filtered list of valid boxes
        """
        valid_boxes = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Calculate box size
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Check if box size is within valid range
            if params['min_size'] <= area <= params['max_size']:
                valid_boxes.append(box)
        
        return valid_boxes
    
    def _get_object_color(self, object_type: str) -> Tuple[int, int, int]:
        """
        Get color for different object types
        
        Args:
            object_type: Type of object
            
        Returns:
            BGR color tuple
        """
        colors = {
            'rice': (0, 255, 0),      # Green
            'coins': (0, 0, 255),     # Red
            'hair': (255, 0, 0),      # Blue
            'general': (0, 255, 255)  # Yellow
        }
        return colors.get(object_type, (0, 255, 255))
    
    def _fallback_detection(self, image: np.ndarray, object_type: str) -> Tuple[int, float, np.ndarray]:
        """
        Fallback detection method when YOLOv8 is not available
        
        Args:
            image: Input image
            object_type: Type of object to detect
            
        Returns:
            Tuple of (count, confidence, processed_image)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Morphological operations to clean up
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours based on object type
            params = self.object_types.get(object_type, self.object_types['general'])
            valid_contours = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if params['min_size'] <= area <= params['max_size']:
                    valid_contours.append(contour)
            
            # Draw contours
            processed_image = image.copy()
            color = self._get_object_color(object_type)
            cv2.drawContours(processed_image, valid_contours, -1, color, 2)
            
            count = len(valid_contours)
            confidence = 0.4  # Lower confidence for fallback method
            
            return count, confidence, processed_image
            
        except Exception as e:
            print(f"Error in fallback detection: {e}")
            return 0, 0.0, image
    
    def get_detection_summary(self, count: int, object_type: str, confidence: float) -> str:
        """
        Generate a summary of the detection results
        
        Args:
            count: Number of objects detected
            object_type: Type of objects
            confidence: Detection confidence
            
        Returns:
            Summary string
        """
        if count == 0:
            return f"No {object_type} detected. Confidence: {confidence:.2f}"
        elif confidence > 0.7:
            return f"High confidence detection: {count} {object_type} found (confidence: {confidence:.2f})"
        elif confidence > 0.4:
            return f"Moderate confidence detection: {count} {object_type} found (confidence: {confidence:.2f})"
        else:
            return f"Low confidence detection: {count} {object_type} found (confidence: {confidence:.2f})"
