#!/usr/bin/env python3
"""
SUPER ACCURATE PERFECT FIT VIRTUAL TRY-ON
=========================================
Using ALL possible body points for perfect garment fitting.
Based on the working simple_perfect_tryon.py but with maximum accuracy.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
from scipy.spatial.distance import euclidean
from scipy import ndimage

class SuperAccurateFit:
    def __init__(self):
        print("🎯 SUPER ACCURATE PERFECT FIT VIRTUAL TRY-ON")
        print("=" * 55)
        print("🔬 Using ALL 33 MediaPipe pose landmarks")
        print("📐 Maximum precision body measurement")
        print("🎽 Perfect garment fitting algorithm")
        
        # MediaPipe setup with maximum accuracy
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_segmentation = mp.solutions.selfie_segmentation
        
        # Maximum accuracy pose detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Highest accuracy
            enable_segmentation=True,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # High-quality background removal
        self.segmentor = self.mp_segmentation.SelfieSegmentation(model_selection=1)
        
        # ALL 33 pose landmarks for maximum accuracy
        self.all_landmarks = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_EYE_INNER,
            self.mp_pose.PoseLandmark.LEFT_EYE,
            self.mp_pose.PoseLandmark.LEFT_EYE_OUTER,
            self.mp_pose.PoseLandmark.RIGHT_EYE_INNER,
            self.mp_pose.PoseLandmark.RIGHT_EYE,
            self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
            self.mp_pose.PoseLandmark.LEFT_EAR,
            self.mp_pose.PoseLandmark.RIGHT_EAR,
            self.mp_pose.PoseLandmark.MOUTH_LEFT,
            self.mp_pose.PoseLandmark.MOUTH_RIGHT,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_PINKY,
            self.mp_pose.PoseLandmark.RIGHT_PINKY,
            self.mp_pose.PoseLandmark.LEFT_INDEX,
            self.mp_pose.PoseLandmark.RIGHT_INDEX,
            self.mp_pose.PoseLandmark.LEFT_THUMB,
            self.mp_pose.PoseLandmark.RIGHT_THUMB,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            self.mp_pose.PoseLandmark.LEFT_HEEL,
            self.mp_pose.PoseLandmark.RIGHT_HEEL,
            self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
            self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
        ]
        
        print("✅ Maximum accuracy pose model initialized")
        print("🎯 33 landmark tracking enabled")
    
    def extract_all_body_measurements(self, landmarks, frame_shape):
        """Extract comprehensive body measurements using ALL landmarks."""
        h, w = frame_shape[:2]
        
        # Convert ALL landmarks to pixel coordinates
        points = {}
        for landmark_id in self.all_landmarks:
            landmark = landmarks[landmark_id]
            if landmark.visibility > 0.3:  # Only use visible landmarks
                points[landmark_id] = {
                    'x': int(landmark.x * w),
                    'y': int(landmark.y * h),
                    'visibility': landmark.visibility,
                    'z': landmark.z
                }
        
        # Core body measurements with maximum precision
        measurements = {}
        
        # 1. SHOULDER MEASUREMENTS (most critical)
        if (self.mp_pose.PoseLandmark.LEFT_SHOULDER in points and 
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER in points):
            
            left_shoulder = points[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            measurements['shoulder_width'] = abs(right_shoulder['x'] - left_shoulder['x'])
            measurements['shoulder_center_x'] = (left_shoulder['x'] + right_shoulder['x']) // 2
            measurements['shoulder_center_y'] = (left_shoulder['y'] + right_shoulder['y']) // 2
            measurements['shoulder_angle'] = np.arctan2(
                right_shoulder['y'] - left_shoulder['y'],
                right_shoulder['x'] - left_shoulder['x']
            )
        
        # 2. TORSO MEASUREMENTS
        if (self.mp_pose.PoseLandmark.LEFT_HIP in points and 
            self.mp_pose.PoseLandmark.RIGHT_HIP in points):
            
            left_hip = points[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = points[self.mp_pose.PoseLandmark.RIGHT_HIP]
            
            measurements['hip_width'] = abs(right_hip['x'] - left_hip['x'])
            measurements['hip_center_x'] = (left_hip['x'] + right_hip['x']) // 2
            measurements['hip_center_y'] = (left_hip['y'] + right_hip['y']) // 2
            
            # Torso height (shoulder to hip)
            if 'shoulder_center_y' in measurements:
                measurements['torso_height'] = abs(measurements['hip_center_y'] - measurements['shoulder_center_y'])
        
        # 3. ARM MEASUREMENTS (for sleeve fitting)
        arm_measurements = {}
        for side in ['LEFT', 'RIGHT']:
            shoulder_key = getattr(self.mp_pose.PoseLandmark, f'{side}_SHOULDER')
            elbow_key = getattr(self.mp_pose.PoseLandmark, f'{side}_ELBOW')
            wrist_key = getattr(self.mp_pose.PoseLandmark, f'{side}_WRIST')
            
            if all(key in points for key in [shoulder_key, elbow_key, wrist_key]):
                shoulder = points[shoulder_key]
                elbow = points[elbow_key]
                wrist = points[wrist_key]
                
                # Upper arm length
                upper_arm = euclidean([shoulder['x'], shoulder['y']], [elbow['x'], elbow['y']])
                # Forearm length
                forearm = euclidean([elbow['x'], elbow['y']], [wrist['x'], wrist['y']])
                # Total arm length
                total_arm = upper_arm + forearm
                
                arm_measurements[f'{side.lower()}_arm_length'] = total_arm
                arm_measurements[f'{side.lower()}_upper_arm'] = upper_arm
                arm_measurements[f'{side.lower()}_forearm'] = forearm
                
                # Arm angle (for sleeve rotation)
                arm_measurements[f'{side.lower()}_arm_angle'] = np.arctan2(
                    elbow['y'] - shoulder['y'],
                    elbow['x'] - shoulder['x']
                )
        
        measurements.update(arm_measurements)
        
        # 4. NECK POSITION (for collar fitting)
        if self.mp_pose.PoseLandmark.NOSE in points:
            nose = points[self.mp_pose.PoseLandmark.NOSE]
            measurements['neck_x'] = nose['x']
            measurements['neck_y'] = nose['y'] + 20  # Estimate neck position below nose
        
        # 5. CHEST WIDTH (using elbow positions when arms are down)
        if (self.mp_pose.PoseLandmark.LEFT_ELBOW in points and 
            self.mp_pose.PoseLandmark.RIGHT_ELBOW in points):
            
            left_elbow = points[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = points[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            
            # Only use elbow width if arms are roughly at sides
            elbow_shoulder_ratio = abs(left_elbow['y'] - measurements.get('shoulder_center_y', 0))
            if elbow_shoulder_ratio < 100:  # Arms roughly at sides
                measurements['chest_width'] = abs(right_elbow['x'] - left_elbow['x']) * 0.8
        
        return measurements
    
    def remove_garment_background_advanced(self, garment):
        """Advanced background removal for garments."""
        # Handle RGBA images by converting to BGR
        if len(garment.shape) == 3 and garment.shape[2] == 4:
            # Convert RGBA to BGR
            garment = cv2.cvtColor(garment, cv2.COLOR_RGBA2BGR)
        elif len(garment.shape) != 3:
            return garment, np.ones(garment.shape[:2], dtype=np.uint8) * 255
        
        # Multiple color space analysis
        gray = cv2.cvtColor(garment, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(garment, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(garment, cv2.COLOR_BGR2LAB)
        
        # Method 1: White background detection
        _, mask1 = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Method 2: HSV white detection
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask2 = cv2.bitwise_not(cv2.inRange(hsv, lower_white, upper_white))
        
        # Method 3: LAB lightness detection
        l_channel = lab[:, :, 0]
        _, mask3 = cv2.threshold(l_channel, 220, 255, cv2.THRESH_BINARY_INV)
        
        # Method 4: Edge-based detection
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask4 = np.zeros(gray.shape, dtype=np.uint8)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.fillPoly(mask4, [largest_contour], 255)
        
        # Combine all masks
        combined_mask = cv2.bitwise_or(mask1, mask2)
        combined_mask = cv2.bitwise_or(combined_mask, mask3)
        combined_mask = cv2.bitwise_or(combined_mask, mask4)
        
        # Morphological operations for cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Gaussian blur for smooth edges
        combined_mask = cv2.GaussianBlur(combined_mask, (3, 3), 0)
        
        return garment, combined_mask
    
    def calculate_perfect_garment_fit(self, measurements, garment_shape):
        """Calculate perfect garment fit using comprehensive measurements."""
        gh, gw = garment_shape[:2]
        
        # Base measurements
        shoulder_width = measurements.get('shoulder_width', 150)
        torso_height = measurements.get('torso_height', 200)
        
        # Advanced sizing algorithm
        # 1. Width calculation using multiple reference points
        width_factors = []
        
        # Shoulder-based width (primary)
        width_factors.append(shoulder_width * 1.8)
        
        # Chest-based width (if available)
        if 'chest_width' in measurements:
            width_factors.append(measurements['chest_width'] * 1.2)
        
        # Hip-based width (for loose fit)
        if 'hip_width' in measurements:
            width_factors.append(measurements['hip_width'] * 1.1)
        
        # Use median of available width measurements
        target_width = int(np.median(width_factors)) if width_factors else 200
        target_width = max(120, min(target_width, 400))  # Reasonable bounds
        
        # 2. Height calculation
        target_height = max(150, int(torso_height * 1.3))
        target_height = max(target_height, int(target_width * 1.2))  # Maintain proportion
        
        # 3. Position calculation
        center_x = measurements.get('shoulder_center_x', 320)
        center_y = measurements.get('shoulder_center_y', 240)
        
        # Position garment slightly below shoulders
        neck_offset = 15  # Distance below neck/shoulders
        start_x = center_x - target_width // 2
        start_y = center_y + neck_offset
        
        return {
            'width': target_width,
            'height': target_height,
            'x': start_x,
            'y': start_y,
            'center_x': center_x,
            'center_y': center_y
        }
    
    def apply_garment_rotation_and_perspective(self, garment, measurements):
        """Apply rotation and perspective correction based on body pose."""
        # FIXED: Disable problematic rotation that was causing upside-down garments
        # The rotation algorithm was causing garments to flip orientation
        # For now, return garment without rotation to ensure correct orientation
        
        # Get shoulder angle for garment rotation
        shoulder_angle = measurements.get('shoulder_angle', 0)
        
        # CONSERVATIVE rotation - only apply very small corrections to avoid flipping
        if abs(shoulder_angle) > 0.2 and abs(shoulder_angle) < 0.3:  # Very narrow range: ~11-17 degrees
            # Apply minimal rotation with careful bounds checking
            center = (garment.shape[1] // 2, garment.shape[0] // 2)
            # Limit rotation to prevent upside-down garments
            safe_angle = np.clip(np.degrees(shoulder_angle), -15, 15)  # Max ±15 degrees
            rotation_matrix = cv2.getRotationMatrix2D(center, safe_angle, 1.0)
            garment = cv2.warpAffine(garment, rotation_matrix, (garment.shape[1], garment.shape[0]))
        
        # Most of the time, return garment as-is to ensure correct orientation
        return garment
    
    def blend_garment_advanced(self, frame, garment, mask, fit_params):
        """Advanced garment blending with perfect positioning."""
        h, w = frame.shape[:2]
        
        # Extract parameters
        start_x = max(0, min(fit_params['x'], w - fit_params['width']))
        start_y = max(0, min(fit_params['y'], h - fit_params['height']))
        end_x = min(w, start_x + fit_params['width'])
        end_y = min(h, start_y + fit_params['height'])
        
        actual_width = end_x - start_x
        actual_height = end_y - start_y
        
        if actual_width > 20 and actual_height > 20:
            # Resize garment and mask to fit
            garment_fitted = cv2.resize(garment, (actual_width, actual_height), 
                                      interpolation=cv2.INTER_LANCZOS4)
            
            # Ensure garment is 3-channel BGR
            if len(garment_fitted.shape) == 3 and garment_fitted.shape[2] == 4:
                garment_fitted = cv2.cvtColor(garment_fitted, cv2.COLOR_RGBA2BGR)
            elif len(garment_fitted.shape) == 2:
                garment_fitted = cv2.cvtColor(garment_fitted, cv2.COLOR_GRAY2BGR)
            
            mask_fitted = cv2.resize(mask, (actual_width, actual_height), 
                                   interpolation=cv2.INTER_LANCZOS4)
            
            # Extract frame region
            frame_region = frame[start_y:end_y, start_x:end_x].copy()
            
            # Normalize mask
            mask_norm = mask_fitted.astype(float) / 255.0
            mask_3d = np.stack([mask_norm] * 3, axis=2)
            
            # Advanced blending with edge feathering
            # Apply slight gaussian blur to mask edges for smooth blending
            mask_blurred = cv2.GaussianBlur(mask_norm, (3, 3), 0)
            mask_3d_blurred = np.stack([mask_blurred] * 3, axis=2)
            
            # Blend with improved alpha
            blended = (mask_3d_blurred * garment_fitted.astype(float) + 
                      (1 - mask_3d_blurred) * frame_region.astype(float))
            
            # Place back in frame
            frame[start_y:end_y, start_x:end_x] = blended.astype(np.uint8)
        
        return frame
    
    def process_frame_super_accurate(self, frame, garment_img):
        """Process frame with super accurate fitting."""
        if frame is None or garment_img is None:
            return frame
        
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Background removal
        seg_result = self.segmentor.process(rgb_frame)
        if seg_result.segmentation_mask is not None:
            mask = seg_result.segmentation_mask
            condition = np.stack((mask,) * 3, axis=-1) > 0.5
            background = np.zeros(frame.shape, dtype=np.uint8)
            frame = np.where(condition, frame, background)
        
        # Pose detection
        pose_result = self.pose.process(rgb_frame)
        
        if pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks.landmark
            
            # Extract ALL body measurements
            measurements = self.extract_all_body_measurements(landmarks, frame.shape)
            
            if measurements:  # Only proceed if we have measurements
                # Advanced garment processing
                garment_clean, garment_mask = self.remove_garment_background_advanced(garment_img)
                
                # Apply rotation based on body pose
                garment_rotated = self.apply_garment_rotation_and_perspective(garment_clean, measurements)
                
                # Calculate perfect fit
                fit_params = self.calculate_perfect_garment_fit(measurements, garment_rotated.shape)
                
                # Blend garment
                frame = self.blend_garment_advanced(frame, garment_rotated, garment_mask, fit_params)
                
                # Debug visualization
                if 'shoulder_center_x' in measurements and 'shoulder_center_y' in measurements:
                    cv2.circle(frame, (measurements['shoulder_center_x'], measurements['shoulder_center_y']), 
                             3, (0, 255, 0), -1)
                
                # Show measurements
                y_offset = 30
                for key, value in list(measurements.items())[:5]:  # Show first 5 measurements
                    if isinstance(value, (int, float)):
                        cv2.putText(frame, f"{key}: {value:.1f}", (10, y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        y_offset += 20
                
                cv2.putText(frame, "SUPER ACCURATE FIT", (10, h - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "POSITION YOURSELF IN FRONT OF CAMERA", (w//4, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return frame

def main():
    """Main function with super accurate fitting."""
    # Initialize
    fitter = SuperAccurateFit()
    
    # Load garments (using the working method from simple_perfect_tryon.py)
    garments = []
    garment_paths = ["Garments/tops/", "Shirts/"]
    for path in garment_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    garments.append(os.path.join(path, file))
    
    if not garments:
        print("ERROR: No garments found!")
        return
    
    print(f"\n🎽 Found {len(garments)} garments:")
    for i, garment in enumerate(garments):
        print(f"  {i+1}. {os.path.basename(garment)}")
    
    # Initialize camera (same settings as working version)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot access camera!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    current_garment = 0
    
    print("\n🎮 CONTROLS:")
    print("   N: Next garment")
    print("   P: Previous garment") 
    print("   S: Save screenshot")
    print("   ESC: Exit")
    print("\n🚀 Starting super accurate virtual try-on...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror effect
        frame = cv2.flip(frame, 1)
        
        # Load current garment
        garment_path = garments[current_garment]
        garment_img = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
        
        if garment_img is not None:
            # Process with super accurate fitting
            result_frame = fitter.process_frame_super_accurate(frame, garment_img)
        else:
            result_frame = frame
        
        # Add UI
        garment_name = os.path.basename(garments[current_garment])
        cv2.putText(result_frame, f"Current: {garment_name}", (20, result_frame.shape[0] - 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_frame, f"Garment {current_garment + 1} of {len(garments)}", 
                   (20, result_frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result_frame, "N=Next P=Prev S=Save ESC=Exit", 
                   (20, result_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('SUPER ACCURATE Perfect Fit Virtual Try-On', result_frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in [ord('n'), ord('N')]:
            current_garment = (current_garment + 1) % len(garments)
            print(f"📱 Switched to: {os.path.basename(garments[current_garment])}")
        elif key in [ord('p'), ord('P')]:
            current_garment = (current_garment - 1) % len(garments)
            print(f"📱 Switched to: {os.path.basename(garments[current_garment])}")
        elif key in [ord('s'), ord('S')]:
            # Ensure Screenshots directory exists
            os.makedirs("Screenshots", exist_ok=True)
            filename = f"Screenshots/super_accurate_fit_{int(time.time())}.jpg"
            cv2.imwrite(filename, result_frame)
            print(f"📸 Screenshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    fitter.pose.close()
    fitter.segmentor.close()
    print("👋 Super accurate virtual try-on closed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚡ System interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
