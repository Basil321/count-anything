"""
Screenshot Helper for CountAnything AI
Run this to get guided instructions for taking screenshots
"""

import time

def print_banner():
    print("🎯" + "="*60 + "🎯")
    print("    COUNTANYTHING AI - SCREENSHOT GUIDE")
    print("🎯" + "="*60 + "🎯")

def screenshot_guide():
    print_banner()
    
    print("\n📸 SCREENSHOT INSTRUCTIONS:")
    print("Follow these steps to capture all required screenshots:\n")
    
    screenshots = [
        {
            "name": "web_interface.png",
            "app": "Streamlit Web Interface",
            "command": "streamlit run app/main.py",
            "instructions": [
                "1. Wait for Streamlit to open in browser",
                "2. Upload a sample image (use any image with objects)",
                "3. Select 'Rice' as object type",
                "4. Adjust confidence slider to 0.3",
                "5. Click 'Count Objects'", 
                "6. Take screenshot showing results with count and confidence",
                "7. Save as 'screenshots/web_interface.png'"
            ]
        },
        {
            "name": "camera_counter.png", 
            "app": "Ultimate Camera Counter",
            "command": "python ultimate_camera_counter.py",
            "instructions": [
                "1. Run the command and wait for camera to start",
                "2. Press '3' to switch to Neon theme",
                "3. Press 'r' to switch to rice detection",
                "4. Point camera at some objects (coins, rice, small items)",
                "5. Press 'b' to ensure bounding boxes are on",
                "6. Press 'a' to enable animations",
                "7. Take screenshot showing detection with neon overlay",
                "8. Save as 'screenshots/camera_counter.png'"
            ]
        },
        {
            "name": "rice_detection.png",
            "app": "Rice Detector Specialized",
            "command": "python rice_detector.py",
            "instructions": [
                "1. Run the rice detector test",
                "2. It will create 'rice_detection_test.jpg'", 
                "3. Open this file and take screenshot",
                "4. Or use camera mode with rice objects",
                "5. Save as 'screenshots/rice_detection.png'"
            ]
        },
        {
            "name": "ultimate_interface.png",
            "app": "Ultimate Interface with Stats",
            "command": "python ultimate_camera_counter.py",
            "instructions": [
                "1. Run ultimate camera counter",
                "2. Press '4' to switch to cyberpunk theme",
                "3. Press 'x' to show statistics overlay",
                "4. Press 'o' to ensure overlay is visible",
                "5. Let it run for a minute to build statistics",
                "6. Take screenshot showing full interface with stats",
                "7. Save as 'screenshots/ultimate_interface.png'"
            ]
        }
    ]
    
    for i, shot in enumerate(screenshots, 1):
        print(f"\n📸 SCREENSHOT {i}: {shot['name']}")
        print(f"   App: {shot['app']}")
        print(f"   Command: {shot['command']}")
        print("   Steps:")
        for instruction in shot['instructions']:
            print(f"      {instruction}")
        
        if i < len(screenshots):
            input(f"\n   Press Enter when ready for Screenshot {i+1}...")
    
    print("\n🎨 DIAGRAM CREATION:")
    print("For the diagrams, you can:")
    print("1. Create simple flowcharts using draw.io or Canva")
    print("2. Show the flow: Camera/Upload → Detection Algorithms → Count Results")
    print("3. Include the multiple algorithms: Threshold, Gabor, Color, Edge, Hough, YOLO")
    
    print(f"\n✅ All screenshots should be saved in: D:\\useless\\screenshots\\")
    print("   Make sure they show the actual detection working!")

def quick_test():
    print("\n🚀 QUICK TEST MODE:")
    print("This will run a quick test of each application...")
    
    apps = [
        ("Web Interface", "streamlit run app/main.py --server.headless true"),
        ("Simple Camera", "python simple_camera_counter.py"),
        ("Ultimate Camera", "python ultimate_camera_counter.py")
    ]
    
    for name, cmd in apps:
        print(f"\n✅ {name} - Command: {cmd}")
        print("   (Run this manually and take screenshots)")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Full Screenshot Guide")
    print("2. Quick Test Commands")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        screenshot_guide()
    elif choice == "2":
        quick_test()
    else:
        print("Invalid choice. Running full guide...")
        screenshot_guide()
    
    print(f"\n🎯 Good luck with your screenshots!")
    print("Remember: Show the apps actually detecting and counting objects!")
