import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time
from typing import Dict, List, Tuple, Optional

# Smart Mirror Virtual Try-On System
# Supports both tops and bottoms with improved body landmark detection

# Configuration for Smart Mirror Virtual Try-On
class VirtualTryOnConfig:
    def __init__(self):
        self.camera_width = 1280
        self.camera_height = 720
        self.fps = 30
        
        # Garment categories
        self.garment_types = ["tops", "bottoms"]
        self.current_garment_type = "tops"
        
        # Fitting parameters
        self.top_ratio = 2.5  # Shoulder width multiplier for tops
        self.bottom_ratio = 1.8  # Hip width multiplier for bottoms
        self.top_height_ratio = 1.3  # Body length ratio for tops
        self.bottom_height_ratio = 1.5  # Leg length ratio for bottoms
        
        # Detection confidence
        self.pose_confidence = 0.7
        self.hand_confidence = 0.7
        
        # UI settings
        self.selection_speed = 8
        self.button_margin = 20
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0

config = VirtualTryOnConfig()

# Initialize MediaPipe with optimized settings for Smart Mirror
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=True,
    min_detection_confidence=config.pose_confidence, 
    min_tracking_confidence=config.pose_confidence
)

hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    model_complexity=1,
    min_detection_confidence=config.hand_confidence, 
    min_tracking_confidence=config.hand_confidence
)

# Smart Mirror Garment Management System
class GarmentManager:
    def __init__(self):
        self.garments = {"tops": [], "bottoms": []}
        self.current_indices = {"tops": 0, "bottoms": 0}
        self.load_garments()
    
    def load_garments(self):
        """Load garments from organized folders"""
        base_path = "Garments"
        
        # Create directories if they don't exist
        for garment_type in ["tops", "bottoms"]:
            path = os.path.join(base_path, garment_type)
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"Created directory: {path}")
        
        # Load existing garments
        for garment_type in self.garments.keys():
            path = os.path.join(base_path, garment_type)
            if os.path.exists(path):
                files = [f for f in os.listdir(path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                files.sort()
                self.garments[garment_type] = files
                print(f"Loaded {len(files)} {garment_type}: {files}")
        
        # Fallback to legacy Shirts folder for compatibility
        if not self.garments["tops"] and os.path.exists("Shirts"):
            files = [f for f in os.listdir("Shirts") 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            files.sort()
            self.garments["tops"] = files
            print(f"Loaded {len(files)} tops from legacy Shirts folder: {files}")
    
    def get_current_garment_path(self, garment_type: str) -> Optional[str]:
        """Get path to current garment"""
        if not self.garments[garment_type]:
            return None
        
        filename = self.garments[garment_type][self.current_indices[garment_type]]
        
        # Check organized folders first
        path = os.path.join("Garments", garment_type, filename)
        if os.path.exists(path):
            return path
        
        # Fallback to legacy folder for tops
        if garment_type == "tops":
            legacy_path = os.path.join("Shirts", filename)
            if os.path.exists(legacy_path):
                return legacy_path
        
        return None
    
    def next_garment(self, garment_type: str):
        """Switch to next garment"""
        if self.garments[garment_type]:
            self.current_indices[garment_type] = (
                self.current_indices[garment_type] + 1
            ) % len(self.garments[garment_type])
    
    def previous_garment(self, garment_type: str):
        """Switch to previous garment"""
        if self.garments[garment_type]:
            self.current_indices[garment_type] = (
                self.current_indices[garment_type] - 1
            ) % len(self.garments[garment_type])
    
    def get_garment_info(self, garment_type: str) -> Dict:
        """Get current garment information"""
        if not self.garments[garment_type]:
            return {"name": "None", "index": 0, "total": 0}
        
        return {
            "name": self.garments[garment_type][self.current_indices[garment_type]],
            "index": self.current_indices[garment_type] + 1,
            "total": len(self.garments[garment_type])
        }

# Advanced Body Tracking and Garment Fitting System
class BodyTracker:
    def __init__(self):
        self.landmark_history = []
        self.max_history = 5  # Frames to smooth over
        self.last_valid_landmarks = None
    
    def extract_key_landmarks(self, pose_landmarks) -> Optional[Dict]:
        """Extract and validate key body landmarks"""
        if not pose_landmarks:
            return None
        
        try:
            landmarks = {
                'left_shoulder': pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],
                'right_shoulder': pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                'left_hip': pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP],
                'right_hip': pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP],
                'left_knee': pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE],
                'right_knee': pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE],
                'left_ankle': pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE],
                'right_ankle': pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE],
                'nose': pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            }
            
            # Validate landmarks (check if they're visible and within frame)
            for name, landmark in landmarks.items():
                if landmark.visibility < 0.5:
                    return None
            
            return landmarks
        except:
            return None
    
    def smooth_landmarks(self, landmarks: Dict) -> Dict:
        """Apply temporal smoothing to landmarks"""
        if not landmarks:
            return self.last_valid_landmarks
        
        self.landmark_history.append(landmarks)
        if len(self.landmark_history) > self.max_history:
            self.landmark_history.pop(0)
        
        # Simple moving average smoothing
        smoothed = {}
        for name in landmarks.keys():
            x_sum = sum(frame[name].x for frame in self.landmark_history)
            y_sum = sum(frame[name].y for frame in self.landmark_history)
            z_sum = sum(frame[name].z for frame in self.landmark_history)
            
            # Create a simple object to store smoothed coordinates
            class SmoothedLandmark:
                def __init__(self, x, y, z, visibility=1.0):
                    self.x = x
                    self.y = y
                    self.z = z
                    self.visibility = visibility
            
            smoothed[name] = SmoothedLandmark(
                x_sum / len(self.landmark_history),
                y_sum / len(self.landmark_history),
                z_sum / len(self.landmark_history)
            )
        
        self.last_valid_landmarks = smoothed
        return smoothed
    
    def calculate_garment_fitting(self, landmarks: Dict, frame_shape: Tuple, garment_type: str) -> Dict:
        """Calculate optimal garment positioning and sizing"""
        ih, iw = frame_shape[:2]
        
        # Convert to pixel coordinates
        coords = {}
        for name, landmark in landmarks.items():
            coords[name] = (int(landmark.x * iw), int(landmark.y * ih))
        
        if garment_type == "tops":
            return self._calculate_top_fitting(coords, iw, ih, landmarks)
        elif garment_type == "bottoms":
            return self._calculate_bottom_fitting(coords, iw, ih, landmarks)
        else:
            return None
    
    def _calculate_top_fitting(self, coords: Dict, iw: int, ih: int, landmarks: Dict) -> Dict:
        """Calculate fitting for tops (shirts, jackets, etc.)"""
        shoulder_width = abs(coords['left_shoulder'][0] - coords['right_shoulder'][0])
        shoulder_center_x = (coords['left_shoulder'][0] + coords['right_shoulder'][0]) // 2
        shoulder_center_y = (coords['left_shoulder'][1] + coords['right_shoulder'][1]) // 2
        
        # Calculate torso length (shoulder to hip)
        torso_length = abs(shoulder_center_y - (coords['left_hip'][1] + coords['right_hip'][1]) // 2)
        
        # Garment dimensions
        garment_width = int(shoulder_width * config.top_ratio)
        garment_height = int(torso_length * config.top_height_ratio)
        
        # Ensure minimum dimensions
        garment_width = max(garment_width, 80)
        garment_height = max(garment_height, 100)
        
        # Position (centered on shoulders, slightly above)
        x = max(0, min(iw - garment_width, shoulder_center_x - garment_width // 2))
        y = max(0, min(ih - garment_height, shoulder_center_y - int(garment_height * 0.1)))
        
        return {
            'x': x, 'y': y,
            'width': garment_width, 'height': garment_height,
            'confidence': min(landmarks['left_shoulder'].visibility, landmarks['right_shoulder'].visibility)
        }
    
    def _calculate_bottom_fitting(self, coords: Dict, iw: int, ih: int, landmarks: Dict) -> Dict:
        """Calculate fitting for bottoms (pants, skirts, etc.)"""
        hip_width = abs(coords['left_hip'][0] - coords['right_hip'][0])
        hip_center_x = (coords['left_hip'][0] + coords['right_hip'][0]) // 2
        hip_center_y = (coords['left_hip'][1] + coords['right_hip'][1]) // 2
        
        # Calculate leg length (hip to ankle)
        leg_length = abs(hip_center_y - (coords['left_ankle'][1] + coords['right_ankle'][1]) // 2)
        
        # Garment dimensions
        garment_width = int(hip_width * config.bottom_ratio)
        garment_height = int(leg_length * config.bottom_height_ratio)
        
        # Ensure minimum dimensions
        garment_width = max(garment_width, 60)
        garment_height = max(garment_height, 120)
        
        # Position (centered on hips)
        x = max(0, min(iw - garment_width, hip_center_x - garment_width // 2))
        y = max(0, min(ih - garment_height, hip_center_y - int(garment_height * 0.1)))
        
        return {
            'x': x, 'y': y,
            'width': garment_width, 'height': garment_height,
            'confidence': min(landmarks['left_hip'].visibility, landmarks['right_hip'].visibility)
        }

# Enhanced UI and Interaction System
class SmartMirrorUI:
    def __init__(self):
        self.counter_right = 0
        self.counter_left = 0
        self.counter_mode = 0
        self.load_ui_elements()
    
    def load_ui_elements(self):
        """Load UI button images"""
        self.button_right = cv2.imread("button.png", cv2.IMREAD_UNCHANGED)
        if self.button_right is None:
            # Create a simple button if image not found
            self.button_right = self.create_default_button("→")
        
        self.button_left = cv2.flip(self.button_right, 1)
        self.button_mode = self.create_default_button("⚙")
    
    def create_default_button(self, text: str) -> np.ndarray:
        """Create a default button with text"""
        button = np.zeros((80, 80, 4), dtype=np.uint8)
        # Create a circle
        cv2.circle(button, (40, 40), 35, (100, 100, 100, 255), -1)
        cv2.circle(button, (40, 40), 35, (255, 255, 255, 255), 3)
        
        # Add text
        cv2.putText(button, text, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255), 2)
        return button
    
    def draw_ui(self, image: np.ndarray, garment_manager: GarmentManager) -> np.ndarray:
        """Draw the Smart Mirror UI"""
        ih, iw = image.shape[:2]
        
        # Performance info
        config.fps_counter += 1
        current_time = time.time()
        if current_time - config.fps_time > 1.0:
            config.current_fps = config.fps_counter
            config.fps_counter = 0
            config.fps_time = current_time
        
        # Header info
        cv2.putText(image, "Smart Mirror - Virtual Try-On", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f"FPS: {config.current_fps}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Current mode and garment info
        mode_text = f"Mode: {config.current_garment_type.upper()}"
        cv2.putText(image, mode_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Current garment info for active type
        info = garment_manager.get_garment_info(config.current_garment_type)
        if info["total"] > 0:
            garment_text = f"{config.current_garment_type}: {info['index']}/{info['total']} - {info['name']}"
            cv2.putText(image, garment_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(image, f"No {config.current_garment_type} available", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
        
        # Instructions
        instructions = [
            "Controls:",
            "• Point at ← → to change garments",
            "• Point at ⚙ to switch mode (tops/bottoms)",
            "• Press 'T' for tops, 'B' for bottoms",
            "• Press 'Esc' to exit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(image, instruction, (10, ih - 120 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw buttons
        button_y = ih // 2 - self.button_right.shape[0] // 2
        
        # Left button (previous)
        image = self.overlay_image_alpha(image, self.button_left, config.button_margin, button_y)
        
        # Right button (next)
        right_x = iw - self.button_right.shape[1] - config.button_margin
        image = self.overlay_image_alpha(image, self.button_right, right_x, button_y)
        
        # Mode button (top center)
        mode_x = iw // 2 - self.button_mode.shape[1] // 2
        image = self.overlay_image_alpha(image, self.button_mode, mode_x, 20)
        
        return image
    
    def overlay_image_alpha(self, background: np.ndarray, overlay: np.ndarray, x: int, y: int) -> np.ndarray:
        """Enhanced overlay with alpha blending"""
        h, w = overlay.shape[:2]
        
        # Bounds checking
        x = max(0, min(x, background.shape[1] - w))
        y = max(0, min(y, background.shape[0] - h))
        
        if len(overlay.shape) == 3 and overlay.shape[2] == 4:
            # Handle RGBA overlay
            overlay_img = overlay[:, :, :3]
            mask = overlay[:, :, 3:] / 255.0
            
            background_section = background[y:y+h, x:x+w]
            blended = (1.0 - mask) * background_section + mask * overlay_img
            background[y:y+h, x:x+w] = blended.astype(np.uint8)
        else:
            # Handle RGB overlay
            background[y:y+h, x:x+w] = overlay
        
        return background
    
    def handle_hand_gestures(self, hands_results, image: np.ndarray, garment_manager: GarmentManager):
        """Handle hand gesture interactions"""
        if not hands_results.multi_hand_landmarks:
            self.counter_right = 0
            self.counter_left = 0
            self.counter_mode = 0
            return
        
        ih, iw = image.shape[:2]
        
        for hand_landmarks in hands_results.multi_hand_landmarks:
            # Get finger tip position
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_tip.x * iw), int(index_tip.y * ih)
            
            # Draw finger position
            cv2.circle(image, (x, y), 12, (0, 255, 255), -1)
            cv2.circle(image, (x, y), 15, (255, 255, 255), 2)
            
            # Button boundaries
            button_y = ih // 2 - self.button_right.shape[0] // 2
            button_height = self.button_right.shape[0]
            
            # Left button (previous)
            left_bounds = (config.button_margin, 
                          config.button_margin + self.button_left.shape[1],
                          button_y, 
                          button_y + button_height)
            
            # Right button (next)
            right_x = iw - self.button_right.shape[1] - config.button_margin
            right_bounds = (right_x, 
                           right_x + self.button_right.shape[1],
                           button_y, 
                           button_y + button_height)
            
            # Mode button
            mode_x = iw // 2 - self.button_mode.shape[1] // 2
            mode_bounds = (mode_x, 
                          mode_x + self.button_mode.shape[1],
                          20, 
                          20 + self.button_mode.shape[0])
            
            # Check button interactions
            if self.point_in_bounds(x, y, left_bounds):
                self.counter_left += 1
                progress = min(self.counter_left * config.selection_speed, 360)
                center = (left_bounds[0] + self.button_left.shape[1] // 2, 
                         left_bounds[2] + button_height // 2)
                cv2.ellipse(image, center, (45, 45), 0, 0, progress, (0, 255, 0), 8)
                
                if progress >= 360:
                    garment_manager.previous_garment(config.current_garment_type)
                    self.counter_left = 0
                    print(f"Previous {config.current_garment_type}")
            
            elif self.point_in_bounds(x, y, right_bounds):
                self.counter_right += 1
                progress = min(self.counter_right * config.selection_speed, 360)
                center = (right_bounds[0] + self.button_right.shape[1] // 2, 
                         right_bounds[2] + button_height // 2)
                cv2.ellipse(image, center, (45, 45), 0, 0, progress, (0, 255, 0), 8)
                
                if progress >= 360:
                    garment_manager.next_garment(config.current_garment_type)
                    self.counter_right = 0
                    print(f"Next {config.current_garment_type}")
            
            elif self.point_in_bounds(x, y, mode_bounds):
                self.counter_mode += 1
                progress = min(self.counter_mode * config.selection_speed, 360)
                center = (mode_bounds[0] + self.button_mode.shape[1] // 2, 
                         mode_bounds[2] + self.button_mode.shape[0] // 2)
                cv2.ellipse(image, center, (35, 35), 0, 0, progress, (255, 255, 0), 6)
                
                if progress >= 360:
                    # Switch between tops and bottoms
                    config.current_garment_type = "bottoms" if config.current_garment_type == "tops" else "tops"
                    self.counter_mode = 0
                    print(f"Switched to {config.current_garment_type} mode")
            
            else:
                self.counter_left = 0
                self.counter_right = 0
                self.counter_mode = 0
    
    def point_in_bounds(self, x: int, y: int, bounds: Tuple) -> bool:
        """Check if point is within rectangular bounds"""
        x1, x2, y1, y2 = bounds
        return x1 <= x <= x2 and y1 <= y <= y2

# Garment Rendering System
class GarmentRenderer:
    def __init__(self):
        self.garment_cache = {}
    
    def load_garment(self, path: str) -> Optional[np.ndarray]:
        """Load and cache garment image"""
        if path in self.garment_cache:
            return self.garment_cache[path]
        
        try:
            garment = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if garment is not None:
                self.garment_cache[path] = garment
                return garment
        except Exception as e:
            print(f"Error loading garment {path}: {e}")
        
        return None
    
    def render_garment(self, image: np.ndarray, garment_path: str, fitting: Dict) -> np.ndarray:
        """Render garment on the person"""
        garment = self.load_garment(garment_path)
        if garment is None:
            return image
        
        # Resize garment to fit
        resized_garment = cv2.resize(garment, (fitting['width'], fitting['height']), 
                                   interpolation=cv2.INTER_LANCZOS4)
        
        # Apply overlay
        return self.overlay_garment(image, resized_garment, fitting['x'], fitting['y'])
    
    def overlay_garment(self, background: np.ndarray, garment: np.ndarray, x: int, y: int) -> np.ndarray:
        """Overlay garment with proper alpha blending"""
        h, w = garment.shape[:2]
        
        # Ensure coordinates are within bounds
        x = max(0, min(x, background.shape[1] - w))
        y = max(0, min(y, background.shape[0] - h))
        
        if len(garment.shape) == 3:
            if garment.shape[2] == 4:  # RGBA
                garment_img = garment[:, :, :3].astype(float)
                alpha = garment[:, :, 3:].astype(float) / 255.0
            else:  # RGB
                garment_img = garment.astype(float)
                alpha = np.ones((h, w, 1), dtype=float)
        else:
            return background
        
        # Alpha blending
        background_section = background[y:y+h, x:x+w].astype(float)
        blended = (1.0 - alpha) * background_section + alpha * garment_img
        background[y:y+h, x:x+w] = blended.astype(np.uint8)
        
        return background

# Initialize system components
garment_manager = GarmentManager()
body_tracker = BodyTracker()
ui = SmartMirrorUI()
renderer = GarmentRenderer()

# Initialize camera
cap = cv2.VideoCapture(0)

# Check if camera is available
if not cap.isOpened():
    print("Error: Cannot access camera!")
    print("Please check:")
    print("1. Camera is connected and not used by another application")
    print("2. Camera permissions are granted in System Preferences > Security & Privacy > Camera")
    
    # Try alternative camera indices
    for i in range(1, 5):
        print(f"Trying camera index {i}...")
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at index {i}")
            break
    
    if not cap.isOpened():
        print("No camera found. Please check your camera connection and permissions.")
        exit()

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
cap.set(cv2.CAP_PROP_FPS, config.fps)

# Test camera
ret, test_frame = cap.read()
if not ret:
    print("Error: Cannot read from camera!")
    print("Please grant camera access in System Preferences > Security & Privacy > Camera")
    cap.release()
    exit()
else:
    print(f"Smart Mirror initialized successfully! Resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
    print("Available garment types:", config.garment_types)

print("\n=== Smart Mirror Virtual Try-On System ===")
print("Features:")
print("- Support for tops and bottoms")
print("- Advanced body landmark detection")
print("- Real-time garment fitting")
print("- Hand gesture controls")
print("- Error handling and validation")
print("\nStarting application...")

# Main application loop
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip image for mirror effect
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process pose and hands
    pose_results = pose.process(image_rgb)
    hands_results = hands.process(image_rgb)
    
    # Extract and track body landmarks
    if pose_results.pose_landmarks:
        landmarks = body_tracker.extract_key_landmarks(pose_results.pose_landmarks)
        
        if landmarks:
            # Apply smoothing
            smoothed_landmarks = body_tracker.smooth_landmarks(landmarks)
            
            if smoothed_landmarks:
                # Calculate garment fitting for current type
                fitting = body_tracker.calculate_garment_fitting(
                    smoothed_landmarks, image.shape, config.current_garment_type
                )
                
                if fitting and fitting['confidence'] > 0.6:
                    # Get current garment path
                    garment_path = garment_manager.get_current_garment_path(config.current_garment_type)
                    
                    if garment_path:
                        # Render garment
                        image = renderer.render_garment(image, garment_path, fitting)
                        
                        # Draw confidence indicator
                        conf_text = f"Fit Confidence: {fitting['confidence']:.2f}"
                        cv2.putText(image, conf_text, (10, 150), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # Show detection status
                    cv2.putText(image, "Adjusting fit...", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        # Show detection prompt
        cv2.putText(image, "Please stand in front of camera", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
        cv2.putText(image, "Ensure your full body is visible", (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
    
    # Draw UI
    image = ui.draw_ui(image, garment_manager)
    
    # Handle hand gestures
    ui.handle_hand_gestures(hands_results, image, garment_manager)
    
    # Handle keyboard input
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # Esc
        break
    elif key == ord('t') or key == ord('T'):
        config.current_garment_type = "tops"
        print("Switched to tops mode")
    elif key == ord('b') or key == ord('B'):
        config.current_garment_type = "bottoms"
        print("Switched to bottoms mode")
    elif key == ord('s') or key == ord('S'):
        # Save screenshot
        timestamp = int(time.time())
        screenshot_path = f"screenshot_{timestamp}.png"
        cv2.imwrite(screenshot_path, image)
        print(f"Screenshot saved: {screenshot_path}")
    elif key == ord('r') or key == ord('R'):
        # Reset to first garment
        garment_manager.current_indices[config.current_garment_type] = 0
        print(f"Reset to first {config.current_garment_type}")

    # Display the result
    cv2.imshow('Smart Mirror - Virtual Try-On', image)

print("Smart Mirror application closed.")
cap.release()
cv2.destroyAllWindows()
