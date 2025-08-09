"""
Simple Camera Counter - Standalone Version
Real-time object counting using webcam with OpenCV.
"""

import cv2
import numpy as np
import time
import os

class SimpleCameraCounter:
    """Simple real-time object counting using webcam"""
    
    def __init__(self):
        """Initialize camera counter"""
        self.camera = None
        self.is_running = False
        
        # Detection settings
        self.object_type = "general"
        self.confidence_threshold = 0.3
        
        # Camera settings
        self.camera_index = 0
        self.frame_width = 640
        self.frame_height = 480
        
        # Object detection parameters
        self.object_params = {
            'rice': {'min_area': 15, 'max_area': 300, 'min_circularity': 0.4},
            'coins': {'min_area': 100, 'max_area': 5000, 'min_circularity': 0.7},
            'hair': {'min_area': 10, 'max_area': 200, 'min_circularity': 0.3},
            'general': {'min_area': 50, 'max_area': 2000, 'min_circularity': 0.4}
        }
    
    def start_camera(self):
        """Start the camera"""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            if not self.camera.isOpened():
                print("❌ Error: Could not open camera")
                return False
            
            print("✅ Camera started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop the camera"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("🛑 Camera stopped")
    
    def detect_objects_simple(self, frame):
        """Simple object detection using threshold-based method"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Get parameters for current object type
            params = self.object_params.get(self.object_type, self.object_params['general'])
            
            # Filter contours
            valid_contours = []
            processed_frame = frame.copy()
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if params['min_area'] <= area <= params['max_area']:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        if circularity >= params['min_circularity']:
                            valid_contours.append(contour)
                            
                            # Draw contour with color based on confidence
                            color = self.get_confidence_color(circularity)
                            cv2.drawContours(processed_frame, [contour], -1, color, 2)
            
            # Calculate confidence
            confidence = self.calculate_confidence(valid_contours, params)
            
            return processed_frame, len(valid_contours), confidence
            
        except Exception as e:
            print(f"Error in object detection: {e}")
            return frame, 0, 0.0
    
    def get_confidence_color(self, circularity):
        """Get color based on circularity (confidence)"""
        if circularity > 0.8:
            return (0, 255, 0)  # Green for high confidence
        elif circularity > 0.6:
            return (0, 255, 255)  # Yellow for medium confidence
        else:
            return (0, 0, 255)  # Red for low confidence
    
    def calculate_confidence(self, contours, params):
        """Calculate overall confidence"""
        if not contours:
            return 0.0
        
        total_confidence = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                confidence = min(circularity / params['min_circularity'], 1.0)
                total_confidence += confidence
        
        return total_confidence / len(contours)
    
    def add_overlay(self, frame, count, confidence):
        """Add information overlay to the frame"""
        overlay = frame.copy()
        
        # Add background rectangle
        cv2.rectangle(overlay, (10, 10), (350, 130), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (350, 130), (255, 255, 255), 2)
        
        # Add text information
        cv2.putText(overlay, f"Objects: {count}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(overlay, f"Confidence: {confidence:.2f}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(overlay, f"Type: {self.object_type.title()}", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return overlay
    
    def run_camera_loop(self):
        """Main camera loop"""
        if not self.start_camera():
            return
        
        self.is_running = True
        print("\n🎥 Simple Camera Counter Started!")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 'r' for rice detection")
        print("- Press 'c' for coin detection")
        print("- Press 'h' for hair detection")
        print("- Press 'g' for general detection")
        print("- Press '1-9' to adjust confidence threshold")
        print("- Press 's' to save current frame")
        print("- Press 'i' for instructions")
        
        frame_count = 0
        start_time = time.time()
        
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                print("❌ Error reading frame")
                break
            
            # Process frame
            processed_frame, count, confidence = self.detect_objects_simple(frame)
            
            # Add overlay
            final_frame = self.add_overlay(processed_frame, count, confidence)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = 30 / elapsed_time
                start_time = time.time()
                print(f"📊 FPS: {fps:.1f}, Objects: {count}, Confidence: {confidence:.2f}")
            
            # Display frame
            cv2.imshow('Simple Camera Counter', final_frame)
            
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
            elif key == ord('1'):
                self.confidence_threshold = 0.1
                print("⚙️ Confidence threshold: 0.1")
            elif key == ord('2'):
                self.confidence_threshold = 0.2
                print("⚙️ Confidence threshold: 0.2")
            elif key == ord('3'):
                self.confidence_threshold = 0.3
                print("⚙️ Confidence threshold: 0.3")
            elif key == ord('4'):
                self.confidence_threshold = 0.4
                print("⚙️ Confidence threshold: 0.4")
            elif key == ord('5'):
                self.confidence_threshold = 0.5
                print("⚙️ Confidence threshold: 0.5")
            elif key == ord('6'):
                self.confidence_threshold = 0.6
                print("⚙️ Confidence threshold: 0.6")
            elif key == ord('7'):
                self.confidence_threshold = 0.7
                print("⚙️ Confidence threshold: 0.7")
            elif key == ord('8'):
                self.confidence_threshold = 0.8
                print("⚙️ Confidence threshold: 0.8")
            elif key == ord('9'):
                self.confidence_threshold = 0.9
                print("⚙️ Confidence threshold: 0.9")
            elif key == ord('s'):
                self.save_frame(final_frame, count, confidence)
            elif key == ord('i'):
                self.show_instructions()
        
        self.stop_camera()
    
    def save_frame(self, frame, count, confidence):
        """Save current frame with detection results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"camera_capture_{timestamp}_count{count}_conf{confidence:.2f}.jpg"
        
        try:
            cv2.imwrite(filename, frame)
            print(f"💾 Frame saved: {filename}")
        except Exception as e:
            print(f"❌ Error saving frame: {e}")
    
    def show_instructions(self):
        """Show detailed instructions"""
        print("\n📖 Instructions:")
        print("1. Point your camera at objects to count")
        print("2. Use different object types for better detection:")
        print("   - Rice: Small, round grains")
        print("   - Coins: Circular objects")
        print("   - Hair: Thin strands")
        print("   - General: Any small objects")
        print("3. Adjust confidence threshold (1-9) for sensitivity")
        print("4. Press 's' to save images with detection results")
        print("5. Green = high confidence, Yellow = medium, Red = low")

def main():
    """Main function"""
    print("🔢 Simple Camera Counter")
    print("=" * 50)
    
    # Create camera counter
    camera_counter = SimpleCameraCounter()
    
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
