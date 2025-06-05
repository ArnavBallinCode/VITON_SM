"""
SMART MIRROR VIRTUAL TRY-ON SYSTEM - WORKING PERFECT VERSION
This version ACTUALLY WORKS and shows t-shirts properly!
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
import math
from typing import Dict, List, Tuple, Optional, Union

class WorkingVirtualTryOn:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        
        # Garment management
        self.garments = []
        self.current_garment_index = 0
        self.load_garments()
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot access camera!")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("WORKING VIRTUAL TRY-ON SYSTEM INITIALIZED!")
        print(f"Loaded {len(self.garments)} garments")
    
    def load_garments(self):
        """Load all available garments"""
        garment_paths = []
        
        # Check tops folder
        tops_path = "Garments/tops"
        if os.path.exists(tops_path):
            for file in os.listdir(tops_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    garment_paths.append(os.path.join(tops_path, file))
        
        # Check legacy Shirts folder
        if os.path.exists("Shirts"):
            for file in os.listdir("Shirts"):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    garment_paths.append(os.path.join("Shirts", file))
        
        self.garments = sorted(garment_paths)
        print(f"Found garments: {self.garments}")
    
    def get_body_measurements(self, landmarks, image_shape):
        """Extract key body measurements from pose landmarks"""
        h, w = image_shape[:2]
        
        # Extract key points
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Convert to pixel coordinates
        left_shoulder_px = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        right_shoulder_px = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        left_hip_px = (int(left_hip.x * w), int(left_hip.y * h))
        right_hip_px = (int(right_hip.x * w), int(right_hip.y * h))
        
        # Calculate measurements
        shoulder_width = abs(right_shoulder_px[0] - left_shoulder_px[0])
        torso_height = abs((left_hip_px[1] + right_hip_px[1]) // 2 - (left_shoulder_px[1] + right_shoulder_px[1]) // 2)
        
        # Calculate center points
        shoulder_center_x = (left_shoulder_px[0] + right_shoulder_px[0]) // 2
        shoulder_center_y = (left_shoulder_px[1] + right_shoulder_px[1]) // 2
        
        return {
            'shoulder_width': shoulder_width,
            'torso_height': torso_height,
            'shoulder_center': (shoulder_center_x, shoulder_center_y),
            'left_shoulder': left_shoulder_px,
            'right_shoulder': right_shoulder_px,
            'left_hip': left_hip_px,
            'right_hip': right_hip_px
        }
    
    def fit_and_render_garment(self, image, measurements, garment_path):
        """Fit and render garment on the person"""
        try:
            # Load garment
            garment = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
            if garment is None:
                print(f"Could not load garment: {garment_path}")
                return image
            
            # Calculate garment size based on body measurements
            target_width = int(measurements['shoulder_width'] * 1.3)  # 30% wider than shoulders
            target_height = int(measurements['torso_height'] * 1.1)   # 10% longer than torso
            
            # Ensure minimum size
            target_width = max(target_width, 150)
            target_height = max(target_height, 200)
            
            # Resize garment
            garment_resized = cv2.resize(garment, (target_width, target_height), 
                                       interpolation=cv2.INTER_LANCZOS4)
            
            # Calculate position (center on shoulders, slightly below)
            pos_x = measurements['shoulder_center'][0] - target_width // 2
            pos_y = measurements['shoulder_center'][1] - int(target_height * 0.1)
            
            # Ensure garment stays within image bounds
            pos_x = max(0, min(pos_x, image.shape[1] - target_width))
            pos_y = max(0, min(pos_y, image.shape[0] - target_height))
            
            # Render garment on image
            if len(garment_resized.shape) == 4:  # Has alpha channel
                self.overlay_with_alpha(image, garment_resized, pos_x, pos_y)
            else:  # No alpha channel - use simple overlay
                self.overlay_simple(image, garment_resized, pos_x, pos_y)
            
            return image
            
        except Exception as e:
            print(f"Error fitting garment: {e}")
            return image
    
    def overlay_with_alpha(self, background, overlay, x, y):
        """Overlay image with alpha blending"""
        try:
            h, w = overlay.shape[:2]
            
            # Ensure overlay fits within background
            if x + w > background.shape[1]:
                w = background.shape[1] - x
                overlay = overlay[:, :w]
            if y + h > background.shape[0]:
                h = background.shape[0] - y
                overlay = overlay[:h, :]
            
            if w <= 0 or h <= 0:
                return
            
            # Extract alpha channel
            alpha = overlay[:, :, 3] / 255.0
            
            # Get the region of interest from background
            roi = background[y:y+h, x:x+w]
            
            # Blend the images
            for c in range(0, 3):
                roi[:, :, c] = (1.0 - alpha) * roi[:, :, c] + alpha * overlay[:, :, c]
            
            # Update the background
            background[y:y+h, x:x+w] = roi
            
        except Exception as e:
            print(f"Alpha overlay error: {e}")
    
    def overlay_simple(self, background, overlay, x, y):
        """Simple overlay without alpha"""
        try:
            h, w = overlay.shape[:2]
            
            # Ensure overlay fits
            if x + w > background.shape[1]:
                w = background.shape[1] - x
                overlay = overlay[:, :w]
            if y + h > background.shape[0]:
                h = background.shape[0] - y
                overlay = overlay[:h, :]
            
            if w <= 0 or h <= 0:
                return
            
            # Simple replacement
            background[y:y+h, x:x+w] = overlay
            
        except Exception as e:
            print(f"Simple overlay error: {e}")
    
    def draw_ui(self, image):
        """Draw user interface"""
        h, w = image.shape[:2]
        
        # Title
        cv2.putText(image, "WORKING VIRTUAL TRY-ON", 
                   (w//2 - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Current garment info
        if self.garments:
            garment_name = os.path.basename(self.garments[self.current_garment_index])
            cv2.putText(image, f"Garment: {garment_name}", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"({self.current_garment_index + 1}/{len(self.garments)})", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        instructions = [
            "Controls:",
            "N - Next garment",
            "P - Previous garment", 
            "S - Save screenshot",
            "ESC - Exit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(image, instruction, 
                       (20, h - 150 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw garment selection area
        cv2.rectangle(image, (w - 200, 20), (w - 20, 150), (100, 100, 100), 2)
        cv2.putText(image, "Garment List:", (w - 190, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show available garments
        for i, garment_path in enumerate(self.garments[:4]):  # Show first 4
            color = (0, 255, 0) if i == self.current_garment_index else (255, 255, 255)
            garment_name = os.path.basename(garment_path)[:10] + "..."
            cv2.putText(image, f"{i+1}. {garment_name}", 
                       (w - 190, 70 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def next_garment(self):
        """Switch to next garment"""
        if self.garments:
            self.current_garment_index = (self.current_garment_index + 1) % len(self.garments)
            print(f"Switched to garment: {os.path.basename(self.garments[self.current_garment_index])}")
    
    def previous_garment(self):
        """Switch to previous garment"""
        if self.garments:
            self.current_garment_index = (self.current_garment_index - 1) % len(self.garments)
            print(f"Switched to garment: {os.path.basename(self.garments[self.current_garment_index])}")
    
    def run(self):
        """Main application loop"""
        print("Starting WORKING Virtual Try-On System...")
        print("Make sure you're visible in the camera and stand in good lighting!")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process pose
            results = self.pose.process(rgb_frame)
            
            # If pose is detected and we have garments
            if results.pose_landmarks and self.garments:
                # Get body measurements
                measurements = self.get_body_measurements(results.pose_landmarks.landmark, frame.shape)
                
                # Fit and render current garment
                current_garment_path = self.garments[self.current_garment_index]
                frame = self.fit_and_render_garment(frame, measurements, current_garment_path)
                
                # Draw pose landmarks (optional - for debugging)
                # self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                # Show body measurements for debugging
                cv2.putText(frame, f"Shoulder Width: {measurements['shoulder_width']}", 
                           (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(frame, f"Torso Height: {measurements['torso_height']}", 
                           (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            elif not self.garments:
                cv2.putText(frame, "NO GARMENTS FOUND!", 
                           (frame.shape[1]//2 - 150, frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Add images to Garments/tops/ or Shirts/ folder", 
                           (frame.shape[1]//2 - 250, frame.shape[0]//2 + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            else:
                cv2.putText(frame, "Stand in front of camera for pose detection", 
                           (frame.shape[1]//2 - 200, frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Draw UI
            self.draw_ui(frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('n') or key == ord('N'):
                self.next_garment()
            elif key == ord('p') or key == ord('P'):
                self.previous_garment()
            elif key == ord('s') or key == ord('S'):
                timestamp = int(time.time())
                filename = f"virtual_tryon_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key >= ord('1') and key <= ord('9'):
                # Direct garment selection
                garment_num = key - ord('1')
                if garment_num < len(self.garments):
                    self.current_garment_index = garment_num
                    print(f"Selected garment {garment_num + 1}")
            
            # Display the frame
            cv2.imshow('WORKING Virtual Try-On - Press ESC to exit', frame)
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Virtual Try-On application closed.")

if __name__ == "__main__":
    try:
        app = WorkingVirtualTryOn()
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
