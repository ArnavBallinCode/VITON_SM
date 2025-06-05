import cv2
import mediapipe as mp
import numpy as np
import os
import time

def main():
    print("=== SIMPLE PERFECT VIRTUAL TRY-ON WITH BACKGROUND REMOVAL ===")
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

    # Initialize MediaPipe Selfie Segmentation for background removal
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    # Load garments (same as before)
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
    print("\n=== VIRTUAL TRY-ON STARTED WITH BACKGROUND REMOVAL ===")
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

        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get pose results
        results_pose = pose.process(rgb_frame)

        # Get segmentation mask (person vs background)
        results_seg = selfie_segmentation.process(rgb_frame)
        mask = results_seg.segmentation_mask

        # Create a binary mask where person is 1, background 0
        condition = np.stack((mask,) * 3, axis=-1) > 0.5  # Threshold mask

        # Create background image (black or custom)
        background = np.zeros(frame.shape, dtype=np.uint8)  # black background
        # Or you can load custom background here:
        # background = cv2.imread('background.jpg')
        # background = cv2.resize(background, (w, h))

        # Combine frame and background using mask
        frame_no_bg = np.where(condition, frame, background)

        # Now, overlay the garment on frame_no_bg (instead of original frame)
        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark

            # Extract key points in pixel coordinates
            left_shoulder_px = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                                int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
            right_shoulder_px = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                                 int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
            left_hip_px = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w),
                           int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h))
            right_hip_px = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w),
                            int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h))

            # Calculate dimensions
            shoulder_width = abs(right_shoulder_px[0] - left_shoulder_px[0])
            torso_height = abs(((left_hip_px[1] + right_hip_px[1]) // 2) - ((left_shoulder_px[1] + right_shoulder_px[1]) // 2))

            garment_width = max(int(shoulder_width * 1.4), 200)
            garment_height = max(int(torso_height * 1.2), 250)

            center_x = (left_shoulder_px[0] + right_shoulder_px[0]) // 2
            center_y = (left_shoulder_px[1] + right_shoulder_px[1]) // 2

            garment_x = max(0, min(center_x - garment_width // 2, w - garment_width))
            garment_y = max(0, min(center_y - int(garment_height * 0.1), h - garment_height))

            garment_path = garments[current_garment]
            garment_img = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)

            if garment_img is not None:
                garment_resized = cv2.resize(garment_img, (garment_width, garment_height), interpolation=cv2.INTER_LANCZOS4)

                y1, y2 = garment_y, garment_y + garment_height
                x1, x2 = garment_x, garment_x + garment_width

                if y2 <= h and x2 <= w and y1 >= 0 and x1 >= 0:
                    if garment_resized.shape[2] == 4:  # has alpha
                        alpha = garment_resized[:, :, 3] / 255.0
                        alpha = np.stack([alpha]*3, axis=2)

                        roi = frame_no_bg[y1:y2, x1:x2]
                        garment_rgb = garment_resized[:, :, :3]

                        blended = (1 - alpha) * roi + alpha * garment_rgb
                        frame_no_bg[y1:y2, x1:x2] = blended.astype(np.uint8)
                    else:
                        frame_no_bg[y1:y2, x1:x2] = garment_resized

            mp_drawing.draw_landmarks(frame_no_bg, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.putText(frame_no_bg, f"Shoulder Width: {shoulder_width}px", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame_no_bg, f"Torso Height: {torso_height}px", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame_no_bg, "POSE DETECTED - GARMENT FITTED!", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            cv2.putText(frame_no_bg, "STAND IN FRONT OF CAMERA", (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame_no_bg, "POSE NOT DETECTED", (w//2 - 150, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        garment_name = os.path.basename(garments[current_garment])
        cv2.putText(frame_no_bg, f"Current: {garment_name}", (20, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_no_bg, f"Garment {current_garment + 1} of {len(garments)}", (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame_no_bg, "N=Next P=Prev S=Save ESC=Exit", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('PERFECT Virtual Try-On', frame_no_bg)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in [ord('n'), ord('N')]:
            current_garment = (current_garment + 1) % len(garments)
            print(f"Switched to: {os.path.basename(garments[current_garment])}")
        elif key in [ord('p'), ord('P')]:
            current_garment = (current_garment - 1) % len(garments)
            print(f"Switched to: {os.path.basename(garments[current_garment])}")
        elif key in [ord('s'), ord('S')]:
            filename = f"perfect_tryon_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame_no_bg)
            print(f"Screenshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    selfie_segmentation.close()
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
