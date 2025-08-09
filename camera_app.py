"""
CountAnything AI - Camera Application
Real-time object counting using webcam with OpenCV and threshold-based detection.
"""

import cv2
import numpy as np
import time
from pathlib import Path
import sys

# Add app directory to path
sys.path.append(str(Path(__file__).parent / "app"))

from utils.image_processor import ImageProcessor

class CameraCounter:
    """Real-time object counting using webcam"""
    
    def __init__(self):
        """Initialize camera counter"""
        self.image_processor = ImageProcessor()
        self.camera = None
        self.is_running = False
        
        # Detection settings
        self.object_type = "general"
        self.confidence_threshold = 0.3
        self.show_confidence = True
        self.show_area = True
        
        # Camera settings
        self.camera_index = 0
        self.frame_width = 640
        self.frame_height = 480
        
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
    
    def process_frame(self, frame):
        """Process a single frame for object detection"""
        try:
            # Convert frame to BGR if needed
            if len(frame.shape) == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            # Detect objects in the frame
            count, confidence, processed_frame = self.image_processor.detect_objects(
                frame_bgr, self.object_type, self.confidence_threshold
            )
            
            # Add information overlay
            overlay_frame = self.add_overlay(processed_frame, count, confidence)
            
            return overlay_frame, count, confidence
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame, 0, 0.0
    
    def add_overlay(self, frame, count, confidence):
        """Add information overlay to the frame"""
        # Create overlay
        overlay = frame.copy()
        
        # Add background rectangle for text
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (300, 120), (255, 255, 255), 2)
        
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
        print("\n🎥 Camera Counter Started!")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 'r' for rice detection")
        print("- Press 'c' for coin detection")
        print("- Press 'h' for hair detection")
        print("- Press 'g' for general detection")
        print("- Press '1-9' to adjust confidence threshold")
        print("- Press 's' to save current frame")
        print("- Press 'i' to toggle information display")
        
        frame_count = 0
        start_time = time.time()
        
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                print("❌ Error reading frame")
                break
            
            # Process frame
            processed_frame, count, confidence = self.process_frame(frame)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = 30 / elapsed_time
                start_time = time.time()
                print(f"📊 FPS: {fps:.1f}, Objects: {count}, Confidence: {confidence:.2f}")
            
            # Display frame
            cv2.imshow('CountAnything AI - Camera', processed_frame)
            
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
                self.save_frame(processed_frame, count, confidence)
            elif key == ord('i'):
                self.show_confidence = not self.show_confidence
                print(f"ℹ️ Information display: {'ON' if self.show_confidence else 'OFF'}")
        
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

def main():
    """Main function"""
    print("🔢 CountAnything AI - Camera Application")
    print("=" * 50)
    
    # Create camera counter
    camera_counter = CameraCounter()
    
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
