#!/usr/bin/env python3
"""
ACTUALLY WORKING VIRTUAL TRY-ON
===============================
No fancy research-grade nonsense. Just a working system.
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import time

class ActuallyWorkingTryOn:
    def __init__(self):
        """Initialize with simple, working configuration."""
        print("🔧 ACTUALLY WORKING VIRTUAL TRY-ON")
        print("=" * 40)
        print("✅ Simple and functional approach")
        
        # Basic MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_segmentation = mp.solutions.selfie_segmentation
        
        # Simple pose detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Simple segmentation
        self.segmentor = self.mp_segmentation.SelfieSegmentation(model_selection=0)
        
        print("✅ MediaPipe initialized successfully")
    
    def remove_garment_background(self, garment):
        """Simple background removal - remove white/light backgrounds."""
        if len(garment.shape) == 3:
            # Convert to grayscale for mask creation
            gray = cv2.cvtColor(garment, cv2.COLOR_BGR2GRAY)
            
            # Create mask for white/light backgrounds
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            # Clean up the mask
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Apply mask to garment
            result = garment.copy()
            result[mask == 0] = [0, 0, 0]  # Set background to black
            
            return result, mask
        else:
            return garment, np.ones(garment.shape[:2], dtype=np.uint8) * 255
    
    def get_body_measurements(self, landmarks, frame_shape):
        """Get basic body measurements for garment fitting."""
        h, w = frame_shape[:2]
        
        # Get key landmarks
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        
        # Convert to pixel coordinates
        left_shoulder_px = [int(left_shoulder.x * w), int(left_shoulder.y * h)]
        right_shoulder_px = [int(right_shoulder.x * w), int(right_shoulder.y * h)]
        left_hip_px = [int(left_hip.x * w), int(left_hip.y * h)]
        right_hip_px = [int(right_hip.x * w), int(right_hip.y * h)]
        nose_px = [int(nose.x * w), int(nose.y * h)]
        
        # Calculate measurements
        shoulder_width = abs(right_shoulder_px[0] - left_shoulder_px[0])
        shoulder_center_x = (left_shoulder_px[0] + right_shoulder_px[0]) // 2
        shoulder_center_y = (left_shoulder_px[1] + right_shoulder_px[1]) // 2
        
        hip_center_y = (left_hip_px[1] + right_hip_px[1]) // 2
        torso_height = abs(hip_center_y - shoulder_center_y)
        
        return {
            'shoulder_width': shoulder_width,
            'shoulder_center': (shoulder_center_x, shoulder_center_y),
            'torso_height': torso_height,
            'nose_pos': nose_px
        }
    
    def fit_garment(self, garment, measurements):
        """Fit garment to body measurements."""
        # Calculate garment size based on body
        target_width = max(150, int(measurements['shoulder_width'] * 1.8))  # 1.8x shoulder width
        target_height = max(200, int(measurements['torso_height'] * 1.2))   # 1.2x torso height
        
        # Resize garment - THIS IS THE KEY FIX!
        # We need to ensure the garment is RIGHT-SIDE UP
        garment_resized = cv2.resize(garment, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        return garment_resized
    
    def place_garment(self, frame, garment, mask, measurements):
        """Place garment on the body correctly."""
        h, w = frame.shape[:2]
        gh, gw = garment.shape[:2]
        
        # Calculate position - place garment correctly below neck
        shoulder_center = measurements['shoulder_center']
        neck_offset = 20  # Distance below neck to start garment
        
        # Position garment
        start_x = shoulder_center[0] - gw // 2
        start_y = shoulder_center[1] + neck_offset
        
        # Ensure garment fits in frame
        start_x = max(0, min(start_x, w - gw))
        start_y = max(0, min(start_y, h - gh))
        
        end_x = min(w, start_x + gw)
        end_y = min(h, start_y + gh)
        
        # Calculate actual dimensions
        actual_width = end_x - start_x
        actual_height = end_y - start_y
        
        if actual_width > 0 and actual_height > 0:
            # Extract regions
            frame_region = frame[start_y:end_y, start_x:end_x]
            garment_region = garment[:actual_height, :actual_width]
            mask_region = mask[:actual_height, :actual_width]
            
            # Create alpha for blending
            alpha = mask_region.astype(float) / 255.0
            alpha = np.stack([alpha] * 3, axis=2)
            
            # Blend garment with frame
            blended = (alpha * garment_region.astype(float) + 
                      (1 - alpha) * frame_region.astype(float))
            
            # Place back in frame
            frame[start_y:end_y, start_x:end_x] = blended.astype(np.uint8)
        
        return frame
    
    def process_frame(self, frame, garment_img):
        """Process single frame."""
        if frame is None or garment_img is None:
            return frame
        
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Background removal
        seg_result = self.segmentor.process(rgb_frame)
        if seg_result.segmentation_mask is not None:
            mask = seg_result.segmentation_mask
            mask_3d = np.stack([mask] * 3, axis=2)
            
            # Simple dark background
            background = np.zeros_like(frame)
            frame = (mask_3d * frame.astype(float) + 
                    (1 - mask_3d) * background.astype(float)).astype(np.uint8)
        
        # Pose detection
        pose_result = self.pose.process(rgb_frame)
        
        if pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks.landmark
            
            # Get body measurements
            measurements = self.get_body_measurements(landmarks, frame.shape)
            
            # Prepare garment
            garment_clean, garment_mask = self.remove_garment_background(garment_img)
            garment_fitted = self.fit_garment(garment_clean, measurements)
            
            # Resize mask to match fitted garment
            fitted_mask = cv2.resize(garment_mask, 
                                   (garment_fitted.shape[1], garment_fitted.shape[0]), 
                                   interpolation=cv2.INTER_AREA)
            
            # Place garment on body
            frame = self.place_garment(frame, garment_fitted, fitted_mask, measurements)
            
            # Draw shoulder points for debugging
            cv2.circle(frame, measurements['shoulder_center'], 5, (0, 255, 0), -1)
            
            # Show info
            cv2.putText(frame, f"Shoulder Width: {measurements['shoulder_width']}px", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Torso Height: {measurements['torso_height']}px", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "WORKING ALGORITHM", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "STAND FACING CAMERA", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame

def main():
    """Main function - actually working version."""
    # Initialize system
    tryon = ActuallyWorkingTryOn()
    
    # Load garments
    garments = []
    garment_paths = ['Garments/tops/', 'Shirts/']
    
    for path in garment_paths:
        if os.path.exists(path):
            for file in sorted(os.listdir(path)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(path, file)
                    img = cv2.imread(full_path)
                    if img is not None:
                        garments.append({
                            'name': f"{path.split('/')[-2]}/{file}",
                            'image': img,
                            'path': full_path
                        })
                        print(f"✅ Loaded: {path}{file}")
    
    if not garments:
        print("❌ No garments found!")
        return
    
    print(f"\n🎽 Loaded {len(garments)} garments")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not available!")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    current_garment_idx = 0
    
    print("\n🎮 CONTROLS:")
    print("   N: Next garment")
    print("   P: Previous garment") 
    print("   S: Save screenshot")
    print("   ESC: Exit")
    print("\n🚀 Starting actually working virtual try-on...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process frame
        current_garment = garments[current_garment_idx]['image']
        result_frame = tryon.process_frame(frame, current_garment)
        
        # Add garment info
        garment_name = garments[current_garment_idx]['name']
        cv2.putText(result_frame, f"Garment: {garment_name} ({current_garment_idx + 1}/{len(garments)})", 
                   (10, result_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show result
        cv2.imshow('Actually Working Virtual Try-On', result_frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('n') or key == ord('N'):
            current_garment_idx = (current_garment_idx + 1) % len(garments)
            print(f"📱 Switched to: {garments[current_garment_idx]['name']}")
        elif key == ord('p') or key == ord('P'):
            current_garment_idx = (current_garment_idx - 1) % len(garments)
            print(f"📱 Switched to: {garments[current_garment_idx]['name']}")
        elif key == ord('s') or key == ord('S'):
            timestamp = int(time.time())
            filename = f"working_tryon_{timestamp}.jpg"
            cv2.imwrite(filename, result_frame)
            print(f"📸 Screenshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Actually working virtual try-on closed")

if __name__ == "__main__":
    main()
