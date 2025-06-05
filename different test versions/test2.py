import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Pose and Selfie Segmentation
mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation
pose = mp_pose.Pose()
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Load input image
img = cv2.imread("/Users/arnavangarkar/Desktop/Arnav/VITON/Virtual-Shirt-Try-On/Shirts/1.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Segment the person from background
segmentation_result = selfie_segmentation.process(img_rgb)
mask = segmentation_result.segmentation_mask
condition = mask > 0.6

# Replace background with white
bg = np.ones(img.shape, dtype=np.uint8) * 255
output_img = np.where(condition[:, :, np.newaxis], img, bg)

# Detect pose
pose_result = pose.process(img_rgb)
if not pose_result.pose_landmarks:
    raise ValueError("Pose landmarks not detected.")

# Get left and right shoulder coordinates
h, w, _ = img.shape
landmarks = pose_result.pose_landmarks.landmark
left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

left_shoulder_xy = (int(left_shoulder.x * w), int(left_shoulder.y * h))
right_shoulder_xy = (int(right_shoulder.x * w), int(right_shoulder.y * h))

# Compute shoulder center and width
shoulder_center = (
    (left_shoulder_xy[0] + right_shoulder_xy[0]) // 2,
    (left_shoulder_xy[1] + right_shoulder_xy[1]) // 2,
)
shoulder_width = int(np.linalg.norm(np.array(left_shoulder_xy) - np.array(right_shoulder_xy)))

# Load garment with alpha channel
garment_path = os.path.join("Shirts", "3.png")
garment_img = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
if garment_img is None:
    raise FileNotFoundError(f"Garment image not found at path: {garment_path}")

# Resize garment to fit shoulder width
scaling_factor = 2.0  # You can tweak this value to better fit the torso
resized_width = int(shoulder_width * scaling_factor)
aspect_ratio = garment_img.shape[0] / garment_img.shape[1]
resized_height = int(resized_width * aspect_ratio)
garment_img = cv2.resize(garment_img, (resized_width, resized_height))

# Extract RGBA channels
if garment_img.shape[2] == 4:
    alpha = garment_img[:, :, 3] / 255.0
    garment_rgb = garment_img[:, :, :3]
else:
    alpha = np.ones(garment_img.shape[:2])
    garment_rgb = garment_img

# Compute garment placement
x_offset = shoulder_center[0] - resized_width // 2
y_offset = shoulder_center[1]

# Overlay garment on image
overlay_img = output_img.copy()
for c in range(3):
    for y in range(resized_height):
        for x in range(resized_width):
            if 0 <= y + y_offset < h and 0 <= x + x_offset < w:
                overlay_img[y + y_offset, x + x_offset, c] = (
                    alpha[y, x] * garment_rgb[y, x, c] +
                    (1 - alpha[y, x]) * overlay_img[y + y_offset, x + x_offset, c]
                )

# Save and show result
cv2.imwrite("virtual_try_on_output.jpg", overlay_img)
cv2.imshow("Virtual Try-On", overlay_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
