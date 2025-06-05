#!/usr/bin/env python3
"""
BASIC WORKING VIRTUAL TRY-ON
============================
Super simple, just overlay garments correctly.
"""

import cv2
import numpy as np
import mediapipe as mp
import os

def remove_white_background(image):
    """Remove white background from garment image."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create mask - white pixels become 0 (transparent), others become 255 (opaque)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Apply some smoothing
    mask = cv2.medianBlur(mask, 5)
    
    return mask

def main():
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    
    # Load garments
    garments = []
    for folder in ['Garments/tops/', 'Shirts/']:
        if os.path.exists(folder):
            for file in ['1.png', '3.png', '4.png']:
                path = os.path.join(folder, file)
                if os.path.exists(path):
                    img = cv2.imread(path)
                    if img is not None:
                        garments.append(img)
                        print(f"Loaded: {path}")
    
    if not garments:
        print("No garments found!")
        return
    
    # Camera
    cap = cv2.VideoCapture(0)
    current_garment = 0
    
    print("Controls: N=Next, P=Previous, ESC=Exit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)  # Mirror
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w = frame.shape[:2]
            
            # Get shoulder positions
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Convert to pixel coordinates
            left_x = int(left_shoulder.x * w)
            left_y = int(left_shoulder.y * h)
            right_x = int(right_shoulder.x * w)
            right_y = int(right_shoulder.y * h)
            
            # Calculate garment position and size
            shoulder_width = abs(right_x - left_x)
            center_x = (left_x + right_x) // 2
            center_y = (left_y + right_y) // 2
            
            # Size garment based on shoulder width
            garment_width = int(shoulder_width * 2.0)  # Make it 2x shoulder width
            garment_height = int(garment_width * 1.2)  # Maintain aspect ratio
            
            # Get current garment and resize it
            current_img = garments[current_garment]
            resized_garment = cv2.resize(current_img, (garment_width, garment_height))
            
            # Remove background
            mask = remove_white_background(resized_garment)
            
            # Position garment
            start_x = center_x - garment_width // 2
            start_y = center_y - garment_height // 4  # Position slightly below shoulders
            
            # Make sure garment fits in frame
            if (start_x >= 0 and start_y >= 0 and 
                start_x + garment_width <= w and start_y + garment_height <= h):
                
                # Extract the region from frame
                roi = frame[start_y:start_y + garment_height, start_x:start_x + garment_width]
                
                # Apply garment with mask
                for i in range(garment_height):
                    for j in range(garment_width):
                        if mask[i, j] > 0:  # If not background
                            frame[start_y + i, start_x + j] = resized_garment[i, j]
            
            # Draw shoulder points for reference
            cv2.circle(frame, (left_x, left_y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (right_x, right_y), 5, (0, 255, 0), -1)
        
        # Show current garment info
        cv2.putText(frame, f"Garment: {current_garment + 1}/{len(garments)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Basic Virtual Try-On', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('n'):
            current_garment = (current_garment + 1) % len(garments)
            print(f"Switched to garment {current_garment + 1}")
        elif key == ord('p'):
            current_garment = (current_garment - 1) % len(garments)
            print(f"Switched to garment {current_garment + 1}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
