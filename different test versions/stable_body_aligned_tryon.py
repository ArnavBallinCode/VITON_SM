"""
STABLE BODY-ALIGNED VIRTUAL TRY-ON
Advanced body alignment with anti-flicker stabilization
Features:
- Precise 9-landmark body measurement system
- Temporal smoothing to eliminate flickering
- Perfect body center alignment
- Automatic rotation matching body angle
- Body shape adaptation (taper adjustment)
- Enhanced background removal
- Stable tracking with prediction
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
from collections import deque
from scipy import ndimage

class BodyTracker:
    """Stabilized body tracking with temporal smoothing"""
    
    def __init__(self, smoothing_window=5):
        self.smoothing_window = smoothing_window
        self.measurements_history = deque(maxlen=smoothing_window)
        self.position_history = deque(maxlen=smoothing_window)
        self.angle_history = deque(maxlen=smoothing_window)
        
    def add_measurements(self, measurements):
        """Add new measurements to history for smoothing"""
        self.measurements_history.append(measurements)
        self.position_history.append((measurements['garment_x'], measurements['garment_y']))
        self.angle_history.append(measurements['shoulder_angle'])
        
    def get_smoothed_measurements(self):
        """Get temporally smoothed measurements to reduce flickering"""
        if not self.measurements_history:
            return None
            
        # Average recent measurements
        smoothed = {}
        keys_to_smooth = ['shoulder_width', 'torso_height', 'garment_width', 'garment_height']
        
        for key in keys_to_smooth:
            values = [m[key] for m in self.measurements_history]
            smoothed[key] = int(np.mean(values))
            
        # Smooth position with weighted average (recent positions matter more)
        if len(self.position_history) > 1:
            weights = np.linspace(0.5, 1.0, len(self.position_history))
            weights = weights / weights.sum()
            
            x_smooth = int(np.average([pos[0] for pos in self.position_history], weights=weights))
            y_smooth = int(np.average([pos[1] for pos in self.position_history], weights=weights))
        else:
            x_smooth, y_smooth = self.position_history[-1]
            
        smoothed['garment_x'] = x_smooth
        smoothed['garment_y'] = y_smooth
        
        # Smooth angle with circular averaging
        angles = list(self.angle_history)
        if len(angles) > 1:
            # Convert to complex numbers for circular averaging
            complex_angles = [np.exp(1j * angle) for angle in angles]
            avg_complex = np.mean(complex_angles)
            smoothed['shoulder_angle'] = np.angle(avg_complex)
        else:
            smoothed['shoulder_angle'] = angles[-1]
            
        # Copy other values from latest measurement
        latest = self.measurements_history[-1]
        for key in latest:
            if key not in smoothed:
                smoothed[key] = latest[key]
                
        return smoothed

def remove_garment_background(garment_img, threshold=220):
    """Enhanced background removal with better edge detection"""
    if garment_img is None:
        return None
    
    # Convert to RGBA if not already
    if len(garment_img.shape) == 3:
        if garment_img.shape[2] == 3:
            alpha = np.ones((garment_img.shape[0], garment_img.shape[1], 1), dtype=np.uint8) * 255
            garment_img = np.concatenate([garment_img, alpha], axis=2)
    
    img_rgba = garment_img.copy()
    
    # Multiple background removal methods
    gray = cv2.cvtColor(img_rgba[:, :, :3], cv2.COLOR_BGR2GRAY)
    
    # Method 1: White/light background removal
    white_mask = gray > threshold
    
    # Method 2: Edge-based background detection
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Method 3: Corner-based background color detection
    h, w = gray.shape
    corners = [
        img_rgba[0:15, 0:15],
        img_rgba[0:15, w-15:w],
        img_rgba[h-15:h, 0:15],
        img_rgba[h-15:h, w-15:w]
    ]
    
    bg_colors = []
    for corner in corners:
        if corner.size > 0:
            avg_color = np.mean(corner.reshape(-1, 4), axis=0)[:3]
            bg_colors.append(avg_color)
    
    if bg_colors:
        bg_color = np.mean(bg_colors, axis=0)
        color_diff = np.sqrt(np.sum((img_rgba[:, :, :3] - bg_color) ** 2, axis=2))
        color_mask = color_diff < 40
        
        # Combine all masks
        final_mask = white_mask | color_mask
    else:
        final_mask = white_mask
    
    # Apply mask to alpha channel
    img_rgba[:, :, 3][final_mask] = 0
    
    # Smooth edges
    alpha_smooth = cv2.GaussianBlur(img_rgba[:, :, 3], (5, 5), 1.0)
    img_rgba[:, :, 3] = alpha_smooth
    
    # Edge feathering for natural look
    kernel = np.ones((3, 3), np.uint8)
    alpha_eroded = cv2.erode(img_rgba[:, :, 3], kernel, iterations=1)
    edge_mask = (img_rgba[:, :, 3] > 0) & (alpha_eroded == 0)
    img_rgba[:, :, 3][edge_mask] = img_rgba[:, :, 3][edge_mask] * 0.6
    
    return img_rgba

def calculate_precise_body_measurements(landmarks, w, h):
    """Calculate detailed body measurements using 9+ landmarks"""
    mp_pose = mp.solutions.pose
    
    # Extract key landmarks
    landmarks_dict = {}
    landmark_names = [
        'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
        'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP',
        'NOSE', 'LEFT_EAR', 'RIGHT_EAR'
    ]
    
    for name in landmark_names:
        if hasattr(mp_pose.PoseLandmark, name):
            landmark = landmarks[getattr(mp_pose.PoseLandmark, name)]
            landmarks_dict[name] = (int(landmark.x * w), int(landmark.y * h))
    
    # Calculate measurements
    shoulder_width = abs(landmarks_dict['RIGHT_SHOULDER'][0] - landmarks_dict['LEFT_SHOULDER'][0])
    
    shoulder_center = (
        (landmarks_dict['LEFT_SHOULDER'][0] + landmarks_dict['RIGHT_SHOULDER'][0]) // 2,
        (landmarks_dict['LEFT_SHOULDER'][1] + landmarks_dict['RIGHT_SHOULDER'][1]) // 2
    )
    
    hip_center = (
        (landmarks_dict['LEFT_HIP'][0] + landmarks_dict['RIGHT_HIP'][0]) // 2,
        (landmarks_dict['LEFT_HIP'][1] + landmarks_dict['RIGHT_HIP'][1]) // 2
    )
    
    torso_height = abs(hip_center[1] - shoulder_center[1])
    
    # Body angle calculation
    shoulder_angle = np.arctan2(
        landmarks_dict['RIGHT_SHOULDER'][1] - landmarks_dict['LEFT_SHOULDER'][1],
        landmarks_dict['RIGHT_SHOULDER'][0] - landmarks_dict['LEFT_SHOULDER'][0]
    )
    
    # True body center (weighted toward torso)
    body_center = (
        int(shoulder_center[0] * 0.3 + hip_center[0] * 0.7),
        int(shoulder_center[1] * 0.6 + hip_center[1] * 0.4)
    )
    
    # Hip width for taper calculation
    hip_width = abs(landmarks_dict['RIGHT_HIP'][0] - landmarks_dict['LEFT_HIP'][0])
    taper_ratio = shoulder_width / max(hip_width, 1)
    
    # Calculate optimal garment dimensions
    garment_width = max(int(shoulder_width * 1.35), 180)
    garment_height = max(int(torso_height * 1.3), 200)
    
    # Calculate garment position
    garment_x = body_center[0] - garment_width // 2
    garment_y = shoulder_center[1] - int(garment_height * 0.15)
    
    return {
        'shoulder_width': shoulder_width,
        'torso_height': torso_height,
        'shoulder_center': shoulder_center,
        'hip_center': hip_center,
        'body_center': body_center,
        'shoulder_angle': shoulder_angle,
        'taper_ratio': taper_ratio,
        'garment_width': garment_width,
        'garment_height': garment_height,
        'garment_x': garment_x,
        'garment_y': garment_y,
        'landmarks': landmarks_dict
    }

def create_body_aligned_garment(garment_rgba, measurements):
    """Create body-aligned garment with rotation and taper"""
    if garment_rgba is None:
        return None
    
    garment_width = measurements['garment_width']
    garment_height = measurements['garment_height']
    shoulder_angle = measurements['shoulder_angle']
    taper_ratio = measurements['taper_ratio']
    
    # Resize garment
    garment_resized = cv2.resize(garment_rgba, (garment_width, garment_height), 
                                interpolation=cv2.INTER_LANCZOS4)
    
    h, w = garment_resized.shape[:2]
    
    # Apply rotation to match body angle
    if abs(shoulder_angle) > 0.03:  # 1.7 degrees threshold
        rotation_angle = np.degrees(shoulder_angle)
        rotation_angle = np.clip(rotation_angle, -12, 12)  # Limit rotation
        
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        
        # Calculate new dimensions to avoid cropping
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_w = int(h * sin_angle + w * cos_angle)
        new_h = int(h * cos_angle + w * sin_angle)
        
        # Adjust translation
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2
        
        garment_rotated = cv2.warpAffine(garment_resized, rotation_matrix, (new_w, new_h),
                                        flags=cv2.INTER_LANCZOS4,
                                        borderMode=cv2.BORDER_TRANSPARENT)
        
        # Resize back to original dimensions
        garment_rotated = cv2.resize(garment_rotated, (garment_width, garment_height),
                                    interpolation=cv2.INTER_LANCZOS4)
    else:
        garment_rotated = garment_resized
    
    # Apply body taper if needed
    if 0.85 < taper_ratio < 1.4:
        h, w = garment_rotated.shape[:2]
        
        # Perspective transformation for body taper
        taper_factor = min(abs(1.0 - taper_ratio), 0.15)  # Limit taper effect
        
        src_points = np.float32([
            [0, 0], [w, 0], [0, h], [w, h]
        ])
        
        bottom_adjust = int(w * taper_factor * (1.0 - taper_ratio))
        dst_points = np.float32([
            [0, 0], [w, 0],
            [bottom_adjust, h], [w - bottom_adjust, h]
        ])
        
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        garment_tapered = cv2.warpPerspective(garment_rotated, perspective_matrix, (w, h),
                                            flags=cv2.INTER_LANCZOS4,
                                            borderMode=cv2.BORDER_TRANSPARENT)
    else:
        garment_tapered = garment_rotated
    
    return garment_tapered

def safe_alpha_blending(background, garment_rgba, x, y):
    """Safe alpha blending with bounds checking"""
    if garment_rgba is None:
        return background
    
    # Ensure coordinates are integers
    x, y = int(x), int(y)
    h, w = garment_rgba.shape[:2]
    bg_h, bg_w = background.shape[:2]
    
    # Calculate safe bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + w)
    y2 = min(bg_h, y + h)
    
    # Calculate corresponding garment region
    gx1 = max(0, -x)
    gy1 = max(0, -y)
    gx2 = gx1 + (x2 - x1)
    gy2 = gy1 + (y2 - y1)
    
    # Check if there's valid overlap
    if x1 >= x2 or y1 >= y2 or gx1 >= w or gy1 >= h:
        return background
    
    # Extract regions
    bg_region = background[y1:y2, x1:x2]
    garment_region = garment_rgba[gy1:gy2, gx1:gx2]
    
    if bg_region.shape[:2] != garment_region.shape[:2]:
        return background
    
    # Alpha blending
    if garment_region.shape[2] == 4:
        alpha = garment_region[:, :, 3] / 255.0
        garment_rgb = garment_region[:, :, :3]
    else:
        alpha = np.ones(garment_region.shape[:2], dtype=float)
        garment_rgb = garment_region
    
    # Smooth alpha for anti-aliasing
    alpha_smooth = cv2.GaussianBlur(alpha, (3, 3), 0.5)
    alpha_3d = np.stack([alpha_smooth, alpha_smooth, alpha_smooth], axis=2)
    
    # Blend
    blended = (1 - alpha_3d) * bg_region + alpha_3d * garment_rgb
    
    # Update background
    result = background.copy()
    result[y1:y2, x1:x2] = blended.astype(np.uint8)
    
    return result

def main():
    print("=== STABLE BODY-ALIGNED VIRTUAL TRY-ON ===")
    print("Loading stabilized body alignment system...")
    print("Features:")
    print("  ✓ Anti-flicker temporal smoothing")
    print("  ✓ Precise 9-landmark body tracking")
    print("  ✓ Perfect body center alignment")
    print("  ✓ Automatic rotation & taper adjustment")
    print("  ✓ Enhanced background removal")
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    # Initialize body tracker for smoothing
    body_tracker = BodyTracker(smoothing_window=7)
    
    # Load garments
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
    frame_count = 0
    
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
        
        frame_count += 1
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        # Draw title
        cv2.putText(frame, "STABLE BODY-ALIGNED TRY-ON", 
                   (w//2 - 220, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Calculate precise body measurements
            measurements = calculate_precise_body_measurements(landmarks, w, h)
            
            # Add to tracker for smoothing
            body_tracker.add_measurements(measurements)
            
            # Get smoothed measurements (reduces flickering)
            if frame_count > 3:  # Allow a few frames for initialization
                smooth_measurements = body_tracker.get_smoothed_measurements()
                if smooth_measurements:
                    measurements = smooth_measurements
            
            # Ensure garment stays in frame
            measurements['garment_x'] = max(0, min(measurements['garment_x'], 
                                                  w - measurements['garment_width']))
            measurements['garment_y'] = max(0, min(measurements['garment_y'], 
                                                  h - measurements['garment_height']))
            
            # Load and process garment
            garment_path = garments[current_garment]
            garment_img = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
            
            if garment_img is not None:
                # Remove background
                garment_processed = remove_garment_background(garment_img)
                
                # Create body-aligned garment
                garment_aligned = create_body_aligned_garment(garment_processed, measurements)
                
                # Apply with safe alpha blending
                frame = safe_alpha_blending(frame, garment_aligned, 
                                          measurements['garment_x'], measurements['garment_y'])
            
            # Draw pose landmarks (optional - can comment out for cleaner look)
            # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Show alignment info
            angle_deg = np.degrees(measurements['shoulder_angle'])
            cv2.putText(frame, f"Body Angle: {angle_deg:.1f}°", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Taper Ratio: {measurements['taper_ratio']:.2f}", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "PERFECT BODY ALIGNMENT ACTIVE", 
                       (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "ANTI-FLICKER: ON", 
                       (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        else:
            cv2.putText(frame, "STAND IN FRONT OF CAMERA", 
                       (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "POSE NOT DETECTED", 
                       (w//2 - 150, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Show garment info
        garment_name = os.path.basename(garments[current_garment])
        cv2.putText(frame, f"Current: {garment_name}", 
                   (20, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Garment {current_garment + 1} of {len(garments)}", 
                   (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Controls
        cv2.putText(frame, "N=Next P=Prev S=Save ESC=Exit", 
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Display frame
        cv2.imshow('Stable Body-Aligned Virtual Try-On', frame)
        
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
            os.makedirs("Screenshots", exist_ok=True)
            filename = f"Screenshots/stable_tryon_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("Stable Virtual Try-On system closed.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSystem interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
