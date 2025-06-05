#!/usr/bin/env python3
"""
Smart Mirror Perfect Fit Launcher
Run this script to start the Perfect Fit Virtual Try-On System
"""

import sys
import os

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
        import scipy
        print("✓ All dependencies are available")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install missing packages:")
        print("pip install opencv-python mediapipe numpy scipy")
        return False

def main():
    print("=" * 60)
    print("🪞 SMART MIRROR - PERFECT FIT VIRTUAL TRY-ON SYSTEM 🪞")
    print("=" * 60)
    print()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Import and run the perfect fit system
    try:
        import main_perfect_fit
        print("🚀 Starting Perfect Fit System...")
        app = main_perfect_fit.SmartMirrorApp()
        app.run()
    except KeyboardInterrupt:
        print("\n👋 Smart Mirror closed by user")
    except Exception as e:
        print(f"❌ Error running Smart Mirror: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
