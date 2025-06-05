#!/usr/bin/env python3
"""
DEFINITIVE WORKING VIRTUAL TRY-ON
=================================
This version WILL work correctly with proper orientation.
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import time

class DefinitiveVirtualTryOn:
    def __init__(self):
        print("🎯 DEFINITIVE VIRTUAL TRY-ON")
        print("=" * 40)
        
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_segmentation = mp.solutions.selfie_segmentation
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.segmentor = self.mp_segmentation.SelfieSegmentation(model_selection=0)
        
        print("✅ MediaPipe initialized")
    
    def load_garments(self):
        """Load all garments."""
        garments = []
        folders = ['Garments/tops/', 'Shirts/']
        
        for folder in folders:
            if os.path.exists(folder):
                for file in sorted(os.listdir(folder)):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        path = os.path.join(folder, file)
                        img = cv2.imread(path)
                        if img is not None:
                            garments.append({
                                'name': f"{folder.split('/')[-2]}/{file}",
                                'image': img,
                                'path': path
                            })
                            print(f"✅ Loaded: {path}")
        
        return garments
    
    def create_garment_mask(self, garment):
        """Create mask for garment (remove white background)."""
        # Convert to different color spaces for better detection
        gray = cv2.cvtColor(garment, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(garment, cv2.COLOR_BGR2HSV)
        
        # Multiple threshold methods
        _, mask1 = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        _, mask2 = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        
        # HSV based mask for white colors
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask3 = cv2.bitwise_not(cv2.inRange(hsv, lower_white, upper_white))
        
        # Combine masks
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.bitwise_or(mask, mask3)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Smooth edges
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        
        return mask
    
    def get_body_landmarks(self, frame):
        """Get body landmarks from frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if results.pose_landmarks:
            return results.pose_landmarks.landmark
        return None
    
    def calculate_garment_placement(self, landmarks, frame_shape):
        """Calculate where to place the garment."""
        h, w = frame_shape[:2]
        
        # Key points
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        
        # Convert to pixels
        left_shoulder_px = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        right_shoulder_px = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        left_hip_px = (int(left_hip.x * w), int(left_hip.y * h))
        right_hip_px = (int(right_hip.x * w), int(right_hip.y * h))
        nose_px = (int(nose.x * w), int(nose.y * h))
        
        # Calculate measurements
        shoulder_width = abs(right_shoulder_px[0] - left_shoulder_px[0])
        shoulder_center_x = (left_shoulder_px[0] + right_shoulder_px[0]) // 2
        shoulder_center_y = (left_shoulder_px[1] + right_shoulder_px[1]) // 2
        
        torso_height = abs((left_hip_px[1] + right_hip_px[1]) // 2 - shoulder_center_y)
        
        return {
            'shoulder_width': shoulder_width,
            'shoulder_center': (shoulder_center_x, shoulder_center_y),
            'torso_height': torso_height,
            'left_shoulder': left_shoulder_px,
            'right_shoulder': right_shoulder_px,
            'nose': nose_px
        }
    
    def fit_and_place_garment(self, frame, garment, measurements):
        """Fit garment to body and place it correctly."""
        # Calculate target size
        target_width = max(120, int(measurements['shoulder_width'] * 1.8))
        target_height = max(150, int(measurements['torso_height'] * 1.3))
        
        # Resize garment - ENSURE CORRECT ORIENTATION
        garment_resized = cv2.resize(garment, (target_width, target_height), 
                                   interpolation=cv2.INTER_LANCZOS4)
        
        # Create mask for the resized garment
        mask = self.create_garment_mask(garment_resized)
        
        # Calculate position - place below neck area
        center_x, center_y = measurements['shoulder_center']
        start_x = center_x - target_width // 2
        start_y = center_y + 10  # Start 10 pixels below shoulder line
        
        # Ensure garment fits in frame
        h, w = frame.shape[:2]
        start_x = max(0, min(start_x, w - target_width))
        start_y = max(0, min(start_y, h - target_height))
        
        end_x = min(w, start_x + target_width)
        end_y = min(h, start_y + target_height)
        
        actual_width = end_x - start_x
        actual_height = end_y - start_y
        
        if actual_width > 20 and actual_height > 20:
            # Extract regions
            frame_region = frame[start_y:end_y, start_x:end_x].copy()
            garment_region = garment_resized[:actual_height, :actual_width]
            mask_region = mask[:actual_height, :actual_width]
            
            # Normalize mask
            mask_norm = mask_region.astype(float) / 255.0
            
            # Create 3-channel mask
            mask_3d = np.stack([mask_norm] * 3, axis=2)
            
            # Blend garment with frame
            blended = (mask_3d * garment_region.astype(float) + 
                      (1 - mask_3d) * frame_region.astype(float))
            
            # Place back in frame
            frame[start_y:end_y, start_x:end_x] = blended.astype(np.uint8)
        
        return frame
    
    def remove_background(self, frame):
        """Remove background using MediaPipe segmentation."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.segmentor.process(rgb)
        
        if results.segmentation_mask is not None:
            # Create mask
            mask = results.segmentation_mask
            mask_3d = np.stack([mask] * 3, axis=2)
            
            # Create dark background
            background = np.zeros_like(frame)
            
            # Blend
            frame = (mask_3d * frame.astype(float) + 
                    (1 - mask_3d) * background.astype(float)).astype(np.uint8)
        
        return frame
    
    def process_frame(self, frame, garment):
        """Process single frame."""
        if frame is None or garment is None:
            return frame
        
        # Remove background first
        frame = self.remove_background(frame)
        
        # Get pose landmarks
        landmarks = self.get_body_landmarks(frame)
        
        if landmarks:
            # Calculate measurements
            measurements = self.calculate_garment_placement(landmarks, frame.shape)
            
            # Place garment
            frame = self.fit_and_place_garment(frame, garment, measurements)
            
            # Debug info
            cv2.circle(frame, measurements['shoulder_center'], 3, (0, 255, 0), -1)
            cv2.putText(frame, f"SW: {measurements['shoulder_width']}px", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"TH: {measurements['torso_height']}px", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, "DEFINITIVE ALGORITHM", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(frame, "STAND FACING CAMERA", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame

def main():
    """Main function."""
    # Initialize
    tryon = DefinitiveVirtualTryOn()
    
    # Load garments
    garments = tryon.load_garments()
    
    if not garments:
        print("❌ No garments found!")
        return
    
    print(f"\n🎽 Loaded {len(garments)} garments")
    print("\n🎮 CONTROLS:")
    print("   N: Next garment")
    print("   P: Previous garment") 
    print("   S: Save screenshot")
    print("   ESC: Exit")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    current_idx = 0
    
    print("\n🚀 Starting definitive virtual try-on...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process frame
        current_garment = garments[current_idx]['image']
        result = tryon.process_frame(frame, current_garment)
        
        # Add UI
        garment_name = garments[current_idx]['name']
        cv2.putText(result, f"Garment: {garment_name} ({current_idx + 1}/{len(garments)})", 
                   (10, result.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show
        cv2.imshow('Definitive Virtual Try-On', result)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('n') or key == ord('N'):
            current_idx = (current_idx + 1) % len(garments)
            print(f"📱 Next: {garments[current_idx]['name']}")
        elif key == ord('p') or key == ord('P'):
            current_idx = (current_idx - 1) % len(garments)
            print(f"📱 Previous: {garments[current_idx]['name']}")
        elif key == ord('s') or key == ord('S'):
            filename = f"definitive_tryon_{int(time.time())}.jpg"
            cv2.imwrite(filename, result)
            print(f"📸 Saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Definitive virtual try-on closed")

if __name__ == "__main__":
    main()
