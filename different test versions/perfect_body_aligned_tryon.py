"""
PERFECT BODY-ALIGNED VIRTUAL TRY-ON
Advanced body alignment with background removal and precise fitting!
Features:
- Precise body measurements using 9+ pose landmarks
- Automatic garment rotation to match body angle
- Body taper adjustment for natural fit
- Enhanced background removal algorithms
- Professional alpha blending with edge smoothing
- Real-time body shape adaptation
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

def remove_garment_background(garment_img, threshold=240):
    """
    Advanced background removal for garment images
    Removes white/light backgrounds and enhances transparency
    """
    if garment_img is None:
        return None
    
    # Convert to RGBA if not already
    if len(garment_img.shape) == 3:
        if garment_img.shape[2] == 3:
            # Add alpha channel
            alpha = np.ones((garment_img.shape[0], garment_img.shape[1], 1), dtype=np.uint8) * 255
            garment_img = np.concatenate([garment_img, alpha], axis=2)
    
    # Work with RGBA image
    img_rgba = garment_img.copy()
    
    # Create mask for background removal
    # Method 1: Remove white/light backgrounds
    gray = cv2.cvtColor(img_rgba[:, :, :3], cv2.COLOR_BGR2GRAY)
    white_mask = gray > threshold
    
    # Method 2: Remove similar colored backgrounds (grab cut style)
    # Find dominant background color (usually corners)
    h, w = gray.shape
    corner_samples = [
        img_rgba[0:20, 0:20],          # Top-left
        img_rgba[0:20, w-20:w],        # Top-right  
        img_rgba[h-20:h, 0:20],        # Bottom-left
        img_rgba[h-20:h, w-20:w]       # Bottom-right
    ]
    
    # Get average background color
    bg_colors = []
    for corner in corner_samples:
        if corner.size > 0:
            avg_color = np.mean(corner.reshape(-1, 4), axis=0)[:3]
            bg_colors.append(avg_color)
    
    if bg_colors:
        bg_color = np.mean(bg_colors, axis=0)
        
        # Create mask for similar colors
        color_diff = np.sqrt(np.sum((img_rgba[:, :, :3] - bg_color) ** 2, axis=2))
        color_mask = color_diff < 50  # Threshold for similar colors
        
        # Combine masks
        final_mask = white_mask | color_mask
    else:
        final_mask = white_mask
    
    # Apply mask to alpha channel
    img_rgba[:, :, 3][final_mask] = 0
    
    # Smooth edges for better blending
    alpha_smooth = cv2.GaussianBlur(img_rgba[:, :, 3], (3, 3), 0)
    img_rgba[:, :, 3] = alpha_smooth
    
    # Edge feathering for natural look
    kernel = np.ones((3, 3), np.uint8)
    alpha_eroded = cv2.erode(img_rgba[:, :, 3], kernel, iterations=1)
    alpha_dilated = cv2.dilate(img_rgba[:, :, 3], kernel, iterations=1)
    
    # Create soft edges
    edge_mask = (alpha_dilated > 0) & (alpha_eroded == 0)
    img_rgba[:, :, 3][edge_mask] = img_rgba[:, :, 3][edge_mask] * 0.7
    
    return img_rgba

def calculate_precise_body_measurements(landmarks, w, h):
    """
    Calculate precise body measurements using all relevant pose landmarks
    Returns detailed measurements for perfect garment alignment
    """
    mp_pose = mp.solutions.pose
    
    # Get key landmarks
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    
    # Convert to pixel coordinates
    def to_px(landmark):
        return (int(landmark.x * w), int(landmark.y * h))
    
    left_shoulder_px = to_px(left_shoulder)
    right_shoulder_px = to_px(right_shoulder)
    left_elbow_px = to_px(left_elbow)
    right_elbow_px = to_px(right_elbow)
    left_wrist_px = to_px(left_wrist)
    right_wrist_px = to_px(right_wrist)
    left_hip_px = to_px(left_hip)
    right_hip_px = to_px(right_hip)
    nose_px = to_px(nose)
    
    # Calculate measurements
    shoulder_width = abs(right_shoulder_px[0] - left_shoulder_px[0])
    shoulder_center = ((left_shoulder_px[0] + right_shoulder_px[0]) // 2,
                      (left_shoulder_px[1] + right_shoulder_px[1]) // 2)
    
    hip_width = abs(right_hip_px[0] - left_hip_px[0])
    hip_center = ((left_hip_px[0] + right_hip_px[0]) // 2,
                  (left_hip_px[1] + right_hip_px[1]) // 2)
    
    torso_height = abs(hip_center[1] - shoulder_center[1])
    
    # Calculate body angle (for rotation alignment)
    shoulder_angle = np.arctan2(right_shoulder_px[1] - left_shoulder_px[1], 
                               right_shoulder_px[0] - left_shoulder_px[0])
    
    # Calculate arm span for sleeve fitting
    left_arm_length = np.sqrt((left_wrist_px[0] - left_shoulder_px[0])**2 + 
                             (left_wrist_px[1] - left_shoulder_px[1])**2)
    right_arm_length = np.sqrt((right_wrist_px[0] - right_shoulder_px[0])**2 + 
                              (right_wrist_px[1] - right_shoulder_px[1])**2)
    avg_arm_length = (left_arm_length + right_arm_length) / 2
    
    # Body taper ratio (shoulder to hip)
    taper_ratio = shoulder_width / max(hip_width, 1)
    
    return {
        'shoulder_width': shoulder_width,
        'shoulder_center': shoulder_center,
        'hip_width': hip_width,
        'hip_center': hip_center,
        'torso_height': torso_height,
        'shoulder_angle': shoulder_angle,
        'avg_arm_length': avg_arm_length,
        'taper_ratio': taper_ratio,
        'nose_position': nose_px,
        'body_center': ((shoulder_center[0] + hip_center[0]) // 2,
                       (shoulder_center[1] + hip_center[1]) // 2)
    }

def create_body_aligned_garment(garment_rgba, body_measurements):
    """
    Transform garment to perfectly align with body measurements
    """
    if garment_rgba is None:
        return None
    
    # Get body measurements
    shoulder_width = body_measurements['shoulder_width']
    torso_height = body_measurements['torso_height']
    shoulder_angle = body_measurements['shoulder_angle']
    taper_ratio = body_measurements['taper_ratio']
    
    # Calculate optimal garment dimensions
    garment_width = max(int(shoulder_width * 1.3), 180)  # 30% wider than shoulders
    garment_height = max(int(torso_height * 1.4), 220)   # 40% longer than torso
    
    # Resize garment
    garment_resized = cv2.resize(garment_rgba, (garment_width, garment_height), 
                                interpolation=cv2.INTER_LANCZOS4)
    
    # Apply body-aligned transformations
    h, w = garment_resized.shape[:2]
    
    # 1. Rotation to match shoulder angle
    if abs(shoulder_angle) > 0.05:  # Only rotate if significant angle
        rotation_angle = np.degrees(shoulder_angle)
        # Limit rotation for natural look
        rotation_angle = np.clip(rotation_angle, -15, 15)
        
        # Rotate around center
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        garment_rotated = cv2.warpAffine(garment_resized, rotation_matrix, (w, h), 
                                        flags=cv2.INTER_LANCZOS4, 
                                        borderMode=cv2.BORDER_TRANSPARENT)
    else:
        garment_rotated = garment_resized
    
    # 2. Body taper adjustment (make garment follow body shape)
    if 0.8 < taper_ratio < 1.5:  # Apply taper for realistic body shapes
        # Create perspective transformation for natural taper
        src_points = np.float32([
            [0, 0],                    # Top-left
            [w, 0],                    # Top-right
            [0, h],                    # Bottom-left
            [w, h]                     # Bottom-right
        ])
        
        # Adjust bottom width based on taper ratio
        bottom_adjustment = int(w * 0.1 * (1.0 - taper_ratio))
        dst_points = np.float32([
            [0, 0],                                    # Top-left (unchanged)
            [w, 0],                                    # Top-right (unchanged)
            [bottom_adjustment, h],                    # Bottom-left (tapered)
            [w - bottom_adjustment, h]                 # Bottom-right (tapered)
        ])
        
        # Apply perspective transformation
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        garment_tapered = cv2.warpPerspective(garment_rotated, perspective_matrix, (w, h),
                                            flags=cv2.INTER_LANCZOS4,
                                            borderMode=cv2.BORDER_TRANSPARENT)
    else:
        garment_tapered = garment_rotated
    
    return garment_tapered

def safe_alpha_blending(background, garment_rgba, x, y):
    """
    Professional alpha blending with proper bounds checking
    """
    if garment_rgba is None:
        return background
    
    garment_h, garment_w = garment_rgba.shape[:2]
    bg_h, bg_w = background.shape[:2]
    
    # Convert to integers to avoid slice errors
    x, y = int(x), int(y)
    garment_w, garment_h = int(garment_w), int(garment_h)
    
    # Calculate actual overlay region with bounds checking
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + garment_w)
    y2 = min(bg_h, y + garment_h)
    
    # If no valid overlay region, return original
    if x1 >= x2 or y1 >= y2:
        return background
    
    # Calculate corresponding garment region
    garment_x1 = x1 - x
    garment_y1 = y1 - y
    garment_x2 = garment_x1 + (x2 - x1)
    garment_y2 = garment_y1 + (y2 - y1)
    
    # Extract regions
    bg_region = background[y1:y2, x1:x2]
    garment_region = garment_rgba[garment_y1:garment_y2, garment_x1:garment_x2]
    
    # Check if regions are valid
    if bg_region.size == 0 or garment_region.size == 0:
        return background
    
    # Extract alpha channel
    if garment_region.shape[2] == 4:
        alpha = garment_region[:, :, 3] / 255.0
        garment_rgb = garment_region[:, :, :3]
    else:
        # Create alpha from non-black pixels
        gray = cv2.cvtColor(garment_region, cv2.COLOR_BGR2GRAY)
        alpha = (gray > 10).astype(float)
        garment_rgb = garment_region
    
    # Create 3-channel alpha for broadcasting
    alpha_3d = np.stack([alpha, alpha, alpha], axis=2)
    
    # Advanced blending with edge softening
    alpha_soft = cv2.GaussianBlur(alpha, (3, 3), 0.5)
    alpha_3d_soft = np.stack([alpha_soft, alpha_soft, alpha_soft], axis=2)
    
    # Blend using soft alpha
    blended = (1 - alpha_3d_soft) * bg_region + alpha_3d_soft * garment_rgb
    
    # Update background
    result = background.copy()
    result[y1:y2, x1:x2] = blended.astype(np.uint8)
    
    return result

def main():
    print("=== PERFECT BODY-ALIGNED VIRTUAL TRY-ON ===")
    print("Loading advanced body alignment and background removal system...")
    print("Features:")
    print("  ✓ Precise body measurements using 9+ landmarks")
    print("  ✓ Automatic garment rotation to match body angle")
    print("  ✓ Body taper adjustment for natural fit")
    print("  ✓ Advanced background removal")
    print("  ✓ Professional alpha blending")
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Load garments
    garments = []
    
    # Check for garments in multiple locations
    garment_paths = [
        "Garments/tops/",
        "Shirts/"
    ]
    
    for path in garment_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    garments.append(os.path.join(path, file))
    
    if not garments:
        print("ERROR: No garments found!")
        print("Please add image files to Garments/tops/ or Shirts/ folder")
        return
    
    print(f"Found {len(garments)} garments:")
    for i, garment in enumerate(garments):
        print(f"  {i+1}. {os.path.basename(garment)}")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot access camera!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    current_garment = 0
    print("\n=== VIRTUAL TRY-ON STARTED ===")
    print("Controls:")
    print("  N = Next garment")
    print("  P = Previous garment")
    print("  S = Save screenshot")
    print("  ESC = Exit")
    print("\nStand in front of the camera and pose!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose
        results = pose.process(rgb_frame)
        
        # Draw title
        cv2.putText(frame, "PERFECT BODY-ALIGNED VIRTUAL TRY-ON", 
                   (w//2 - 280, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        if results.pose_landmarks:
            # Get landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Calculate precise body measurements
            body_measurements = calculate_precise_body_measurements(landmarks, w, h)
            
            # Extract measurements for display
            shoulder_width = body_measurements['shoulder_width']
            torso_height = body_measurements['torso_height']
            shoulder_center = body_measurements['shoulder_center']
            body_center = body_measurements['body_center']
            shoulder_angle = body_measurements['shoulder_angle']
            taper_ratio = body_measurements['taper_ratio']
            
            # Calculate garment position based on body center and measurements
            garment_width = max(int(shoulder_width * 1.3), 180)
            garment_height = max(int(torso_height * 1.4), 220)
            
            garment_x = body_center[0] - garment_width // 2
            garment_y = shoulder_center[1] - int(garment_height * 0.15)  # Slightly above shoulders
            
            # Ensure garment stays in frame
            garment_x = max(0, min(garment_x, w - garment_width))
            garment_y = max(0, min(garment_y, h - garment_height))
            
            # Load and process current garment
            garment_path = garments[current_garment]
            garment_img = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
            
            if garment_img is not None:
                # Enhanced background removal
                garment_processed = remove_garment_background(garment_img)
                
                # Create body-aligned garment with perfect fitting
                garment_aligned = create_body_aligned_garment(garment_processed, body_measurements)
                
                # Safe professional alpha blending overlay
                frame = safe_alpha_blending(frame, garment_aligned, garment_x, garment_y)
            
            # Draw pose landmarks (optional)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Show measurements and alignment info
            cv2.putText(frame, f"Shoulder Width: {shoulder_width}px", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Torso Height: {torso_height}px", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Body Angle: {np.degrees(shoulder_angle):.1f}°", 
                       (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Taper Ratio: {taper_ratio:.2f}", 
                       (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "POSE DETECTED - PERFECT ALIGNMENT!", 
                       (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "BODY-ALIGNED FITTING: ACTIVE", 
                       (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        else:
            cv2.putText(frame, "STAND IN FRONT OF CAMERA", 
                       (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "POSE NOT DETECTED", 
                       (w//2 - 150, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Show current garment info
        garment_name = os.path.basename(garments[current_garment])
        cv2.putText(frame, f"Current: {garment_name}", 
                   (20, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Garment {current_garment + 1} of {len(garments)}", 
                   (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Controls
        cv2.putText(frame, "N=Next P=Prev S=Save ESC=Exit", 
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show the frame
        cv2.imshow('Perfect Body-Aligned Virtual Try-On', frame)
        
        # Handle input
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('n') or key == ord('N'):
            current_garment = (current_garment + 1) % len(garments)
            print(f"Switched to: {os.path.basename(garments[current_garment])}")
        elif key == ord('p') or key == ord('P'):
            current_garment = (current_garment - 1) % len(garments)
            print(f"Switched to: {os.path.basename(garments[current_garment])}")
        elif key == ord('s') or key == ord('S'):
            # Ensure Screenshots directory exists
            os.makedirs("Screenshots", exist_ok=True)
            filename = f"Screenshots/body_aligned_tryon_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("Perfect Body-Aligned Virtual Try-On system closed.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSystem interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
