import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Pose and Hands with improved confidence values
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=True,
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    model_complexity=1,
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)

# Initialize video capture with better settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Path to shirt images and load list of shirts
shirtFolderPath = "Shirts"
listShirts = [f for f in os.listdir(shirtFolderPath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
listShirts.sort()  # Sort for consistent ordering
print("List of shirts:", listShirts)
print(f"Number of shirts found: {len(listShirts)}")

if len(listShirts) == 0:
    print("No shirt images found in the Shirts folder!")
    exit()

# Improved shirt sizing parameters for better accuracy
fixedRatio = 2.5  # Optimized for realistic coverage
shirtRatioHeightWidth = 1.2  # Better proportions for shirt fit

imageNumber = 0

# Load button images
imgButtonRight = cv2.imread("button.png", cv2.IMREAD_UNCHANGED)
if imgButtonRight is None:
    print("Error loading right button image. Please check the path.")
    exit()
imgButtonLeft = cv2.flip(imgButtonRight, 1)

counterRight = 0
counterLeft = 0
selectionSpeed = 10

def overlay_image_alpha(background, overlay, x, y):
    """Enhanced overlay function with better alpha blending and error handling"""
    background_width = background.shape[1]
    background_height = background.shape[0]

    # Ensure x and y are within bounds
    x = max(0, min(x, background_width - 1))
    y = max(0, min(y, background_height - 1))

    h, w = overlay.shape[0], overlay.shape[1]

    # Ensure the overlay fits within the background
    if x + w > background_width:
        w = background_width - x
    if y + h > background_height:
        h = background_height - y

    if w <= 0 or h <= 0:
        return background  # Nothing to overlay

    # Resize overlay if necessary with high-quality interpolation
    if overlay.shape[:2] != (h, w):
        overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Handle different image formats
    if len(overlay.shape) == 3:
        if overlay.shape[2] == 3:  # BGR image, add alpha channel
            overlay = np.concatenate([
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
            ], axis=2)
        elif overlay.shape[2] == 4:  # BGRA image
            pass
    else:
        return background  # Cannot process

    overlay_image = overlay[..., :3].astype(float)
    mask = overlay[..., 3:].astype(float) / 255.0

    # Enhanced blending with gamma correction
    background_section = background[y:y+h, x:x+w].astype(float)
    
    # Apply gamma correction for better blending
    gamma = 2.2
    background_section = np.power(background_section / 255.0, gamma)
    overlay_image = np.power(overlay_image / 255.0, gamma)
    
    # Blend
    blended = (1.0 - mask) * background_section + mask * overlay_image
    
    # Convert back from gamma space
    blended = np.power(blended, 1.0 / gamma) * 255.0
    
    background[y:y+h, x:x+w] = blended.astype(np.uint8)

    return background

print("Virtual Try-On Application Started!")
print("Controls:")
print("- Point at left/right buttons to change shirts")
print("- Press 'Esc' to exit")
print("- Press 'r' to reset to first shirt")
print("- Press 'q' to quit")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find poses and hands
    pose_results = pose.process(image_rgb)
    hands_results = hands.process(image_rgb)
    
    if pose_results.pose_landmarks:
        # Get landmarks for shoulders and additional body points for better accuracy
        lm11 = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        lm12 = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lm23 = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        lm24 = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Convert normalized coordinates to pixel coordinates
        ih, iw, _ = image.shape
        lm11_px = (int(lm11.x * iw), int(lm11.y * ih))
        lm12_px = (int(lm12.x * iw), int(lm12.y * ih))
        lm23_px = (int(lm23.x * iw), int(lm23.y * ih))
        lm24_px = (int(lm24.x * iw), int(lm24.y * ih))
        
        # Calculate shirt dimensions with improved accuracy
        shoulder_width = abs(lm11_px[0] - lm12_px[0])
        body_length = abs((lm11_px[1] + lm12_px[1]) // 2 - (lm23_px[1] + lm24_px[1]) // 2)
        
        shirt_width = int(shoulder_width * fixedRatio)
        shirt_height = int(body_length * 1.3)  # Dynamic height based on body proportions
        
        # Better positioning with center alignment
        center_x = (lm11_px[0] + lm12_px[0]) // 2
        shirt_top_left = (
            max(0, min(iw - shirt_width, center_x - shirt_width // 2)),
            max(0, min(ih - shirt_height, min(lm11_px[1], lm12_px[1]) - int(shirt_height * 0.15)))
        )
        
        # Load and resize shirt image with error handling
        imgShirtPath = os.path.join(shirtFolderPath, listShirts[imageNumber])
        try:
            imgShirt = cv2.imread(imgShirtPath, cv2.IMREAD_UNCHANGED)
            if imgShirt is not None:
                # Ensure minimum shirt dimensions for better quality
                if shirt_width < 50 or shirt_height < 50:
                    shirt_width = max(shirt_width, 100)
                    shirt_height = max(shirt_height, 120)
                
                imgShirt = cv2.resize(imgShirt, (shirt_width, shirt_height), 
                                    interpolation=cv2.INTER_LANCZOS4)
                
                # Overlay shirt on image
                image = overlay_image_alpha(image, imgShirt, shirt_top_left[0], shirt_top_left[1])
            else:
                print(f"Failed to load shirt: {imgShirtPath}")
        except Exception as e:
            print(f"Error processing shirt {imgShirtPath}: {e}")
        
        # Optional: Draw pose landmarks for debugging (comment out for better performance)
        # mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Enhanced UI with current shirt information
    cv2.putText(image, f"Shirt: {imageNumber + 1}/{len(listShirts)}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(image, f"Current: {listShirts[imageNumber]}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, "Point at buttons to change shirts", 
                (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Overlay buttons with better positioning
    button_margin = 20
    button_y = image.shape[0] // 2 - imgButtonRight.shape[0] // 2
    image = overlay_image_alpha(image, imgButtonRight, 
                                image.shape[1] - imgButtonRight.shape[1] - button_margin, button_y)
    image = overlay_image_alpha(image, imgButtonLeft, button_margin, button_y)
    
    # Enhanced hand gesture detection with multiple finger tips
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            # Get multiple finger tip landmarks for better accuracy
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            
            # Use average position for more stable detection
            x = int((index_finger_tip.x + middle_finger_tip.x) / 2 * image.shape[1])
            y = int((index_finger_tip.y + middle_finger_tip.y) / 2 * image.shape[0])
            
            # Draw finger position for visual feedback
            cv2.circle(image, (x, y), 10, (0, 255, 255), -1)

            # Right button detection with improved bounds
            button_right_x_start = image.shape[1] - imgButtonRight.shape[1] - button_margin
            button_right_x_end = image.shape[1] - button_margin
            button_y_start = button_y
            button_y_end = button_y + imgButtonRight.shape[0]
            
            if (button_right_x_start <= x <= button_right_x_end and 
                button_y_start <= y <= button_y_end):
                counterRight += 1
                progress_angle = min(counterRight * selectionSpeed, 360)
                cv2.ellipse(image, (button_right_x_start + imgButtonRight.shape[1] // 2, 
                                  button_y + imgButtonRight.shape[0] // 2), 
                           (40, 40), 0, 0, progress_angle, (0, 255, 0), 8)
                if progress_angle >= 360:
                    counterRight = 0
                    imageNumber = (imageNumber + 1) % len(listShirts)
                    print(f"Switched to next shirt: {listShirts[imageNumber]}")
            
            # Left button detection with improved bounds
            button_left_x_start = button_margin
            button_left_x_end = button_margin + imgButtonLeft.shape[1]
            
            elif (button_left_x_start <= x <= button_left_x_end and 
                  button_y_start <= y <= button_y_end):
                counterLeft += 1
                progress_angle = min(counterLeft * selectionSpeed, 360)
                cv2.ellipse(image, (button_left_x_start + imgButtonLeft.shape[1] // 2, 
                                  button_y + imgButtonLeft.shape[0] // 2), 
                           (40, 40), 0, 0, progress_angle, (0, 255, 0), 8)
                if progress_angle >= 360:
                    counterLeft = 0
                    imageNumber = (imageNumber - 1) if imageNumber > 0 else len(listShirts) - 1
                    print(f"Switched to previous shirt: {listShirts[imageNumber]}")
            
            else:
                counterRight = 0
                counterLeft = 0
                
            # Draw hand landmarks for debugging
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Check for keyboard input
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # Press 'Esc' to exit
        break
    elif key == ord('r'):  # Press 'r' to reset to first shirt
        imageNumber = 0
        print(f"Reset to first shirt: {listShirts[imageNumber]}")
    elif key == ord('q'):  # Press 'q' to quit
        break

    cv2.imshow('Virtual Try-On', image)

print("Application closed.")
cap.release()
cv2.destroyAllWindows()
