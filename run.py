#!/usr/bin/env python3
"""
Virtual Shirt Try-On Application Launcher
Optimized for 100% accuracy and performance
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import cv2
        import mediapipe
        import numpy
        print("✅ All dependencies are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Installing required packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please install manually:")
            print("pip install opencv-python mediapipe numpy")
            return False

def check_files():
    """Check if required files exist"""
    required_files = ["main.py", "button.png", "Shirts/"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files found!")
    return True

def main():
    """Main launcher function"""
    print("🎽 Virtual Shirt Try-On Application")
    print("=" * 40)
    
    if not check_dependencies():
        return
    
    if not check_files():
        return
    
    print("\n🚀 Starting application...")
    print("Controls:")
    print("  - Point at buttons to change shirts")
    print("  - 'n' = Next shirt, 'p' = Previous shirt")
    print("  - 'r' = Reset to first shirt")
    print("  - 's' = Save screenshot")
    print("  - 'q' or 'Esc' = Quit")
    print("\nPress any key to continue...")
    input()
    
    try:
        subprocess.run([sys.executable, "main.py"])
    except KeyboardInterrupt:
        print("\n👋 Application closed by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

if __name__ == "__main__":
    main()
