"""
SIMPLE BUT PERFECT VIRTUAL TRY-ON + BACKGROUND REMOVAL
This version removes garment backgrounds for natural fitting!
Features:
- Advanced background removal algorithms
- Professional alpha blending
- Edge smoothing and anti-aliasing
- Automatic transparency detection
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
from scipy import ndimage

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

def enhance_garment_alpha_blending(background, garment_rgba, x, y):
    """
    Professional alpha blending with edge smoothing
    """
    if garment_rgba is None:
        return background
    
    h, w = garment_rgba.shape[:2]
    bg_h, bg_w = background.shape[:2]
    
    # Ensure coordinates are valid
    if x < 0 or y < 0 or x + w > bg_w or y + h > bg_h:
        return background
    
    # Extract alpha channel
    if garment_rgba.shape[2] == 4:
        alpha = garment_rgba[:, :, 3] / 255.0
        garment_rgb = garment_rgba[:, :, :3]
    else:
        # Create alpha from non-black pixels
        gray = cv2.cvtColor(garment_rgba, cv2.COLOR_BGR2GRAY)
        alpha = (gray > 10).astype(float)
        garment_rgb = garment_rgba
    
    # Create 3-channel alpha for broadcasting
    alpha_3d = np.stack([alpha, alpha, alpha], axis=2)
    
    # Get background region
    bg_region = background[y:y+h, x:x+w]
    
    # Advanced blending with edge softening
    # Apply Gaussian blur to alpha for softer edges
    alpha_soft = cv2.GaussianBlur(alpha, (5, 5), 1.0)
    alpha_3d_soft = np.stack([alpha_soft, alpha_soft, alpha_soft], axis=2)
    
    # Blend using soft alpha
    blended = (1 - alpha_3d_soft) * bg_region + alpha_3d_soft * garment_rgb
    
    # Anti-aliasing on edges
    edge_kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
    edges = cv2.filter2D(alpha, -1, edge_kernel)
    edge_mask = np.abs(edges) > 0.1
    
    # Apply additional smoothing to edges
    if np.any(edge_mask):
        edge_alpha = alpha_3d_soft.copy()
        edge_alpha[edge_mask] *= 0.8  # Reduce alpha on edges
        blended = (1 - edge_alpha) * bg_region + edge_alpha * garment_rgb
    
    # Update background
    result = background.copy()
    result[y:y+h, x:x+w] = blended.astype(np.uint8)
    
    return result

def main():
    print("=== PERFECT VIRTUAL TRY-ON + BACKGROUND REMOVAL ===")
    print("Loading advanced background removal system...")
    
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
        cv2.putText(frame, "PERFECT VIRTUAL TRY-ON + BG REMOVAL", 
                   (w//2 - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
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
                # Enhanced background removal
                garment_processed = remove_garment_background(garment_img)
                
                # Resize garment with high-quality interpolation
                garment_resized = cv2.resize(garment_processed, (garment_width, garment_height), 
                                           interpolation=cv2.INTER_LANCZOS4)
                
                # Professional alpha blending overlay
                frame = enhance_garment_alpha_blending(frame, garment_resized, garment_x, garment_y)
            
            # Draw pose landmarks (optional)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Show measurements
            cv2.putText(frame, f"Shoulder Width: {shoulder_width}px", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Torso Height: {torso_height}px", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "POSE DETECTED - GARMENT FITTED!", 
                       (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "BACKGROUND REMOVAL: ACTIVE", 
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
        cv2.imshow('PERFECT Virtual Try-On', frame)
        
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
            filename = f"Screenshots/perfect_tryon_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("Virtual Try-On system closed.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSystem interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()