import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe modules
mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
segmentor = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Load all available garment images
garment_files = [
    'Garments/tops/1.png', 'Garments/tops/3.png', 'Garments/tops/4.png',
    'Shirts/1.png', 'Shirts/3.png', 'Shirts/4.png'
]

garments = []
for file in garment_files:
    try:
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img is not None:
            garments.append((img, file))
            print(f"Loaded garment: {file}")
    except:
        continue

if not garments:
    print("ERROR: No garments found!")
    exit()

current_garment_idx = 0
garment_img = garments[current_garment_idx][0]
print(f"Starting with: {garments[current_garment_idx][1]}")
print("Controls: N=Next garment, P=Previous garment, S=Save screenshot, ESC=Exit")

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Remove background using segmentation
    seg_result = segmentor.process(rgb)
    condition = np.stack((seg_result.segmentation_mask,) * 3, axis=-1) > 0.3
    frame = np.where(condition, frame, (10, 10, 10)).astype(np.uint8)

    # Pose estimation
    results = pose.process(rgb)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        # Get shoulder landmarks
        l_sh = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h]).astype(int)
        r_sh = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h]).astype(int)

        # Compute width and angle
        shoulder_width = int(np.linalg.norm(l_sh - r_sh))
        angle = np.degrees(np.arctan2(r_sh[1] - l_sh[1], r_sh[0] - l_sh[0]))

        # Add debug info
        cv2.putText(frame, f"Shoulder Width: {shoulder_width}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Angle: {angle:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "POSE DETECTED", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show current garment
        garment_name = garments[current_garment_idx][1].split('/')[-1]
        cv2.putText(frame, f"Garment: {garment_name}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Resize and rotate garment
        new_garment = cv2.resize(garment_img, (shoulder_width * 2, shoulder_width * 2))
        center = (new_garment.shape[1] // 2, new_garment.shape[0] // 3)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(new_garment, M, (new_garment.shape[1], new_garment.shape[0]), borderValue=(0, 0, 0, 0))

        # Compute overlay position
        top_left = l_sh - np.array([int(shoulder_width * 0.3), int(shoulder_width * 0.3)])

        # Overlay garment (with alpha handling)
        y1, y2 = max(0, top_left[1]), min(frame.shape[0], top_left[1] + rotated.shape[0])
        x1, x2 = max(0, top_left[0]), min(frame.shape[1], top_left[0] + rotated.shape[1])

        # Handle different image formats
        if rotated.shape[2] == 4:  # Has alpha channel
            alpha = rotated[:, :, 3] / 255.0
            for c in range(3):
                try:
                    garment_region = rotated[:y2 - y1, :x2 - x1, c]
                    alpha_region = alpha[:y2 - y1, :x2 - x1]
                    frame_region = frame[y1:y2, x1:x2, c]
                    frame[y1:y2, x1:x2, c] = (alpha_region * garment_region + 
                                             (1 - alpha_region) * frame_region)
                except:
                    pass
        else:  # No alpha channel - create PERFECT advanced mask to remove background
            rotated_rgb = rotated[:, :, :3]
            
            # === ADVANCED MULTI-METHOD BACKGROUND REMOVAL ===
            
            # Method 1: Convert to multiple color spaces for comprehensive analysis
            hsv = cv2.cvtColor(rotated_rgb, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(rotated_rgb, cv2.COLOR_BGR2LAB)
            
            # Method 2: Enhanced white/light background detection in BGR
            # Detect pure white and near-white backgrounds
            white_mask1 = cv2.inRange(rotated_rgb, (240, 240, 240), (255, 255, 255))
            white_mask2 = cv2.inRange(rotated_rgb, (220, 220, 220), (255, 255, 255))
            
            # Method 3: Light gray variations
            light_gray_mask = cv2.inRange(rotated_rgb, (200, 200, 200), (240, 240, 240))
            very_light_gray = cv2.inRange(rotated_rgb, (245, 245, 245), (255, 255, 255))
            
            # Method 4: HSV-based detection for better color separation
            # Low saturation (grayish/whitish colors)
            low_sat_mask = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
            very_low_sat = cv2.inRange(hsv, (0, 0, 230), (180, 25, 255))
            
            # Method 5: LAB color space for perceptual uniformity
            # High L* value (lightness) detection
            lab_light_mask = cv2.inRange(lab, (200, 0, 0), (255, 255, 255))
            
            # Method 6: Edge-based background detection
            gray = cv2.cvtColor(rotated_rgb, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find the largest contour (likely the garment)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                garment_mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.fillPoly(garment_mask, [largest_contour], 255)
                
                # Dilate to include garment edges
                kernel_dilate = np.ones((5, 5), np.uint8)
                garment_mask = cv2.dilate(garment_mask, kernel_dilate, iterations=2)
            else:
                garment_mask = np.ones(gray.shape, dtype=np.uint8) * 255
            
            # === COMBINE ALL METHODS FOR ULTIMATE PRECISION ===
            
            # Combine all background detection methods
            background_mask = cv2.bitwise_or(white_mask1, white_mask2)
            background_mask = cv2.bitwise_or(background_mask, light_gray_mask)
            background_mask = cv2.bitwise_or(background_mask, very_light_gray)
            background_mask = cv2.bitwise_or(background_mask, low_sat_mask)
            background_mask = cv2.bitwise_or(background_mask, very_low_sat)
            background_mask = cv2.bitwise_or(background_mask, lab_light_mask)
            
            # Refine using garment contour
            background_mask = cv2.bitwise_and(background_mask, cv2.bitwise_not(garment_mask))
            
            # === ADVANCED MORPHOLOGICAL PROCESSING ===
            
            # Remove noise with opening
            kernel_open = np.ones((3, 3), np.uint8)
            background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, kernel_open)
            
            # Fill holes with closing
            kernel_close = np.ones((5, 5), np.uint8)
            background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel_close)
            
            # === CREATE PERFECT ALPHA MASK ===
            
            # Invert to get garment mask
            garment_alpha = cv2.bitwise_not(background_mask).astype(np.float32) / 255.0
            
            # Multiple stages of smoothing for natural edges
            garment_alpha = cv2.GaussianBlur(garment_alpha, (5, 5), 1.0)
            garment_alpha = cv2.bilateralFilter(garment_alpha, 9, 75, 75)
            
            # Edge preservation with guided filter effect
            garment_alpha = cv2.medianBlur((garment_alpha * 255).astype(np.uint8), 3).astype(np.float32) / 255.0
            
            # Final smoothing
            garment_alpha = cv2.GaussianBlur(garment_alpha, (3, 3), 0.5)
            
            # === ADVANCED ALPHA BLENDING WITH LIGHTING ADAPTATION ===
            
            try:
                garment_region = rotated_rgb[:y2 - y1, :x2 - x1]
                alpha_region = garment_alpha[:y2 - y1, :x2 - x1]
                frame_region = frame[y1:y2, x1:x2].astype(np.float32)
                
                # Adaptive lighting compensation
                frame_brightness = np.mean(frame_region)
                garment_brightness = np.mean(garment_region)
                brightness_ratio = frame_brightness / (garment_brightness + 1e-6)
                
                # Adjust garment lighting to match scene
                adjusted_garment = garment_region.astype(np.float32) * min(brightness_ratio, 1.5)
                adjusted_garment = np.clip(adjusted_garment, 0, 255)
                
                # Professional alpha blending with gamma correction
                alpha_3d = np.stack([alpha_region, alpha_region, alpha_region], axis=2)
                
                # Gamma correction for natural blending
                gamma = 1.2
                frame_gamma = np.power(frame_region / 255.0, gamma)
                garment_gamma = np.power(adjusted_garment / 255.0, gamma)
                
                # Blend in gamma space
                blended_gamma = (1 - alpha_3d) * frame_gamma + alpha_3d * garment_gamma
                
                # Convert back to linear space
                blended = np.power(blended_gamma, 1/gamma) * 255.0
                
                frame[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
            except Exception as e:
                print(f"Blending error: {e}")
                pass
    else:
        cv2.putText(frame, "NO POSE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Stand in front of camera", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Virtual Try-On', frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    elif key == ord('n') or key == ord('N'):  # Next garment
        current_garment_idx = (current_garment_idx + 1) % len(garments)
        garment_img = garments[current_garment_idx][0]
        print(f"Switched to: {garments[current_garment_idx][1]}")
    elif key == ord('p') or key == ord('P'):  # Previous garment
        current_garment_idx = (current_garment_idx - 1) % len(garments)
        garment_img = garments[current_garment_idx][0]
        print(f"Switched to: {garments[current_garment_idx][1]}")
    elif key == ord('s') or key == ord('S'):  # Save screenshot
        cv2.imwrite(f'virtual_tryon_screenshot_{current_garment_idx}.png', frame)
        print(f"Screenshot saved as virtual_tryon_screenshot_{current_garment_idx}.png")

cap.release()
cv2.destroyAllWindows()
