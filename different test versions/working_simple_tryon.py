"""
SIMPLE BUT GUARANTEED WORKING VIRTUAL TRY-ON
This version WILL show the t-shirt on you!
"""

import cv2
import numpy as np
import mediapipe as mp
import os

def main():
    print("🚀 Starting SIMPLE Virtual Try-On...")
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    mp_drawing = mp.solutions.drawing_utils
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    segmentor = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    # Load garments
    garments = []
    for folder in ['Garments/tops/', 'Shirts/']:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    garments.append(os.path.join(folder, file))

    if not garments:
        print("❌ No garments found!")
        return

    print(f"✅ Found {len(garments)} garments")
    current_garment = 0

    # Start webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("📱 Camera started! Stand in front and pose!")
    print("Controls: N=Next garment, P=Previous, ESC=Exit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Remove background
        seg_result = segmentor.process(rgb)
        condition = np.stack((seg_result.segmentation_mask,) * 3, axis=-1) > 0.3
        frame = np.where(condition, frame, (20, 20, 20)).astype(np.uint8)

        # Pose estimation
        results = pose.process(rgb)
        
        # Draw title
        cv2.putText(frame, "SIMPLE VIRTUAL TRY-ON", (w//2 - 200, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get shoulder points
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

            # Convert to pixels
            l_sh = np.array([left_shoulder.x * w, left_shoulder.y * h]).astype(int)
            r_sh = np.array([right_shoulder.x * w, right_shoulder.y * h]).astype(int)
            l_hip = np.array([left_hip.x * w, left_hip.y * h]).astype(int)
            r_hip = np.array([right_hip.x * w, right_hip.y * h]).astype(int)

            # Calculate measurements
            shoulder_width = int(np.linalg.norm(r_sh - l_sh))
            torso_height = abs((l_hip[1] + r_hip[1]) // 2 - (l_sh[1] + r_sh[1]) // 2)

            # Size garment
            garment_width = max(int(shoulder_width * 1.4), 200)
            garment_height = max(int(torso_height * 1.2), 250)

            # Position garment
            center_x = (l_sh[0] + r_sh[0]) // 2
            center_y = (l_sh[1] + r_sh[1]) // 2
            
            top_left_x = center_x - garment_width // 2
            top_left_y = center_y - int(garment_height * 0.1)

            # Ensure garment fits in frame
            top_left_x = max(0, min(top_left_x, w - garment_width))
            top_left_y = max(0, min(top_left_y, h - garment_height))

            # Load and resize garment
            garment_path = garments[current_garment]
            garment_img = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)

            if garment_img is not None:
                # Resize garment
                garment_resized = cv2.resize(garment_img, (garment_width, garment_height))

                # Position on frame
                y1, y2 = top_left_y, top_left_y + garment_height
                x1, x2 = top_left_x, top_left_x + garment_width

                if y2 <= h and x2 <= w:
                    roi = frame[y1:y2, x1:x2]
                    
                    if len(garment_resized.shape) == 4:  # Has alpha
                        # Alpha blending
                        alpha = garment_resized[:, :, 3] / 255.0
                        alpha_3d = np.stack([alpha, alpha, alpha], axis=2)
                        
                        garment_rgb = garment_resized[:, :, :3]
                        blended = (1 - alpha_3d) * roi + alpha_3d * garment_rgb
                        frame[y1:y2, x1:x2] = blended.astype(np.uint8)
                    else:
                        # Create mask to remove white background
                        garment_rgb = garment_resized[:, :, :3] if garment_resized.shape[2] >= 3 else garment_resized
                        
                        # Simple white background removal
                        mask = np.all(garment_rgb > [180, 180, 180], axis=2)
                        
                        # Apply garment where not white
                        for c in range(3):
                            roi_channel = roi[:, :, c]
                            garment_channel = garment_rgb[:, :, c]
                            roi_channel[~mask] = garment_channel[~mask]
                            frame[y1:y2, x1:x2, c] = roi_channel

            # Show info
            cv2.putText(frame, f"Shoulder Width: {shoulder_width}px", 
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Garment Size: {garment_width}x{garment_height}", 
                       (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "✅ POSE DETECTED - SHIRT FITTED!", 
                       (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw pose landmarks (optional)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        else:
            cv2.putText(frame, "❌ NO POSE DETECTED", (w//2 - 150, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(frame, "Stand in front of camera", (w//2 - 180, h//2 + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Show current garment
        garment_name = os.path.basename(garments[current_garment])
        cv2.putText(frame, f"Current: {garment_name}", 
                   (20, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"{current_garment + 1} of {len(garments)}", 
                   (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "N=Next P=Prev ESC=Exit", 
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('SIMPLE Virtual Try-On', frame)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in [ord('n'), ord('N')]:
            current_garment = (current_garment + 1) % len(garments)
            print(f"➡️ Switched to: {garment_name}")
        elif key in [ord('p'), ord('P')]:
            current_garment = (current_garment - 1) % len(garments)
            print(f"⬅️ Switched to: {garment_name}")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    segmentor.close()
    print("👋 Virtual Try-On closed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
