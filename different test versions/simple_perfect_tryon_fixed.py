"""
ENHANCED VIRTUAL TRY-ON WITH PERFECT BACKGROUND REMOVAL
This version removes t-shirt backgrounds for natural blending!
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

def create_smart_alpha_mask(garment_img):
    """
    Create an intelligent alpha mask by removing background colors
    and enhancing garment edges for natural blending
    """
    # Convert to different color spaces for better background detection
    hsv = cv2.cvtColor(garment_img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(garment_img, cv2.COLOR_BGR2LAB)
    
    # Create multiple masks for different background types
    masks = []
    
    # 1. White background removal (most common)
    white_mask = cv2.inRange(garment_img, (200, 200, 200), (255, 255, 255))
    masks.append(white_mask)
    
    # 2. Light gray background
    light_gray_mask = cv2.inRange(garment_img, (180, 180, 180), (220, 220, 220))
    masks.append(light_gray_mask)
    
    # 3. Very light colors (near white in HSV)
    light_hsv_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
    masks.append(light_hsv_mask)
    
    # 4. Edge-based background detection
    gray = cv2.cvtColor(garment_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Find the largest contour (likely the garment)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(contour_mask, [largest_contour], 255)
        
        # Dilate the contour mask to include garment edges
        kernel = np.ones((5, 5), np.uint8)
        contour_mask = cv2.dilate(contour_mask, kernel, iterations=2)
    else:
        contour_mask = np.ones(gray.shape, dtype=np.uint8) * 255
    
    # Combine all background masks
    combined_bg_mask = np.zeros(gray.shape, dtype=np.uint8)
    for mask in masks:
        combined_bg_mask = cv2.bitwise_or(combined_bg_mask, mask)
    
    # Create alpha mask (1 = garment, 0 = background)
    alpha = np.ones(gray.shape, dtype=np.float32)
    
    # Remove detected backgrounds
    alpha[combined_bg_mask > 0] = 0
    
    # Keep only the garment contour area
    alpha = alpha * (contour_mask.astype(np.float32) / 255.0)
    
    # Smooth the alpha mask for natural edges
    alpha = cv2.GaussianBlur(alpha, (5, 5), 1.0)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    alpha_uint8 = (alpha * 255).astype(np.uint8)
    alpha_uint8 = cv2.morphologyEx(alpha_uint8, cv2.MORPH_CLOSE, kernel)
    alpha_uint8 = cv2.morphologyEx(alpha_uint8, cv2.MORPH_OPEN, kernel)
    
    # Convert back to float
    alpha = alpha_uint8.astype(np.float32) / 255.0
    
    # Ensure alpha values are between 0 and 1
    alpha = np.clip(alpha, 0, 1)
    
    return alpha

def enhance_garment_colors(garment_img, alpha_mask):
    """
    Enhance garment colors for better visual appearance
    """
    # Convert to float for processing
    garment_float = garment_img.astype(np.float32) / 255.0
    
    # Apply subtle color enhancement
    enhanced = np.power(garment_float, 0.9)  # Gamma correction
    
    # Increase saturation slightly
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.1, 0, 1)  # Increase saturation
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Convert back to uint8
    enhanced = (enhanced * 255).astype(np.uint8)
    
    return enhanced

def main():
    print("=== ENHANCED VIRTUAL TRY-ON WITH PERFECT BACKGROUND REMOVAL ===")
    print("Loading system...")
    
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
    print("\n=== ENHANCED VIRTUAL TRY-ON STARTED ===")
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
        cv2.putText(frame, "ENHANCED VIRTUAL TRY-ON", 
                   (w//2 - 220, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        if results.pose_landmarks:
            # Get landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Get shoulder points
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            
            # Convert to pixel coordinates
            left_shoulder_px = (int(left_shoulder.x * w), int(left_shoulder.y * h))
            right_shoulder_px = (int(right_shoulder.x * w), int(right_shoulder.y * h))
            left_hip_px = (int(left_hip.x * w), int(left_hip.y * h))
            right_hip_px = (int(right_hip.x * w), int(right_hip.y * h))
            
            # Calculate measurements
            shoulder_width = abs(right_shoulder_px[0] - left_shoulder_px[0])
            torso_height = abs((left_hip_px[1] + right_hip_px[1]) // 2 - (left_shoulder_px[1] + right_shoulder_px[1]) // 2)
            
            # Calculate garment size and position
            garment_width = max(int(shoulder_width * 1.4), 200)  # 40% wider than shoulders
            garment_height = max(int(torso_height * 1.2), 250)  # 20% longer than torso
            
            # Position garment
            center_x = (left_shoulder_px[0] + right_shoulder_px[0]) // 2
            center_y = (left_shoulder_px[1] + right_shoulder_px[1]) // 2
            
            garment_x = center_x - garment_width // 2
            garment_y = center_y - int(garment_height * 0.1)  # Slightly above shoulders
            
            # Ensure garment stays in frame
            garment_x = max(0, min(garment_x, w - garment_width))
            garment_y = max(0, min(garment_y, h - garment_height))
            
            # Load and resize current garment
            garment_path = garments[current_garment]
            garment_img = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
            
            if garment_img is not None:
                # Resize garment
                garment_resized = cv2.resize(garment_img, (garment_width, garment_height), 
                                           interpolation=cv2.INTER_LANCZOS4)
                
                # Overlay garment
                y1, y2 = garment_y, garment_y + garment_height
                x1, x2 = garment_x, garment_x + garment_width
                
                # Ensure we don't go out of bounds
                if y2 <= h and x2 <= w and y1 >= 0 and x1 >= 0:
                    # Smart background removal and alpha blending
                    roi = frame[y1:y2, x1:x2]
                    
                    if len(garment_resized.shape) == 4:  # Has alpha channel
                        # Use existing alpha channel
                        garment_rgb = garment_resized[:, :, :3]
                        alpha = garment_resized[:, :, 3] / 255.0
                    else:
                        # Create smart alpha mask by removing background
                        garment_rgb = garment_resized.copy()
                        if len(garment_rgb.shape) == 3 and garment_rgb.shape[2] == 4:
                            garment_rgb = garment_rgb[:, :, :3]
                        
                        # Advanced background removal
                        alpha = create_smart_alpha_mask(garment_rgb)
                    
                    # Enhance garment colors
                    garment_enhanced = enhance_garment_colors(garment_rgb, alpha)
                    
                    # Apply advanced alpha blending with edge smoothing
                    alpha_smooth = cv2.GaussianBlur(alpha, (3, 3), 0.5)
                    alpha_smooth_3d = np.stack([alpha_smooth, alpha_smooth, alpha_smooth], axis=2)
                    
                    # Create feathered edges for more natural blending
                    alpha_feathered = cv2.GaussianBlur(alpha_smooth, (5, 5), 1.0)
                    alpha_feathered_3d = np.stack([alpha_feathered, alpha_feathered, alpha_feathered], axis=2)
                    
                    # Multi-layer blending for realism
                    # Layer 1: Base blend with feathered alpha
                    base_blend = (1 - alpha_feathered_3d) * roi.astype(np.float32) + alpha_feathered_3d * garment_enhanced.astype(np.float32)
                    
                    # Layer 2: Sharp details with smooth alpha
                    detail_blend = (1 - alpha_smooth_3d) * base_blend + alpha_smooth_3d * garment_enhanced.astype(np.float32)
                    
                    # Apply the final blend
                    frame[y1:y2, x1:x2] = np.clip(detail_blend, 0, 255).astype(np.uint8)
            
            # Draw pose landmarks (optional - comment out for cleaner look)
            # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Show measurements
            cv2.putText(frame, f"Shoulder Width: {shoulder_width}px", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Torso Height: {torso_height}px", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "POSE DETECTED - GARMENT FITTED!", 
                       (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "BACKGROUND REMOVED FOR NATURAL LOOK", 
                       (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
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
        cv2.imshow('ENHANCED Virtual Try-On', frame)
        
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
            filename = f"enhanced_tryon_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("Enhanced Virtual Try-On system closed.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSystem interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
