import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time
import math
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass

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

# Initialize video capture with better settings and error handling
cap = cv2.VideoCapture(0)

# Check if camera is available
if not cap.isOpened():
    print("Error: Cannot access camera!")
    print("Please check:")
    print("1. Camera is connected and not used by another application")
    print("2. Camera permissions are granted in System Preferences > Security & Privacy > Camera")
    print("3. Try running the application again after granting permissions")
    
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

cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
cap.set(cv2.CAP_PROP_FPS, config.fps)

# Test camera capture
ret, test_frame = cap.read()
if not ret:
    print("Error: Cannot read from camera!")
    print("Please grant camera access in System Preferences > Security & Privacy > Camera")
    cap.release()
    exit()
else:
    print(f"Camera initialized successfully! Resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")

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

garment_manager = GarmentManager()

# Smart Mirror UI and Interaction System
class SmartMirrorUI:
    def __init__(self):
        self.current_mode = "tops"  # "tops" or "bottoms"
        self.gesture_counters = {"left": 0, "right": 0, "mode_switch": 0}
        self.selection_threshold = 45  # Frames to hold for selection
        self.mode_switch_threshold = 60  # Frames to hold for mode switch
        
        # Load UI buttons
        self.load_buttons()
        
        # Performance tracking
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def load_buttons(self):
        """Load UI button images"""
        try:
            self.button_right = cv2.imread("button.png", cv2.IMREAD_UNCHANGED)
            self.button_left = cv2.flip(self.button_right.copy(), 1) if self.button_right is not None else None
            
            if self.button_right is None:
                # Create simple circular buttons as fallback
                self.button_right = self.create_default_button("→")
                self.button_left = self.create_default_button("←")
                print("Using default buttons (button.png not found)")
            else:
                print("Loaded button images successfully")
        except Exception as e:
            print(f"Error loading buttons: {e}")
            self.button_right = self.create_default_button("→")
            self.button_left = self.create_default_button("←")
    
    def create_default_button(self, text: str) -> np.ndarray:
        """Create a default button with text"""
        button = np.zeros((80, 80, 4), dtype=np.uint8)
        cv2.circle(button, (40, 40), 35, (100, 100, 100, 200), -1)
        cv2.circle(button, (40, 40), 35, (255, 255, 255, 255), 3)
        
        # Add text
        font_scale = 1.5
        thickness = 2
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (80 - text_size[0]) // 2
        text_y = (80 + text_size[1]) // 2
        
        cv2.putText(button, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (255, 255, 255, 255), thickness)
        
        return button

    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            current_time = time.time()
            self.current_fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time

    def draw_ui(self, image: np.ndarray) -> np.ndarray:
        """Draw Smart Mirror UI elements"""
        height, width = image.shape[:2]
        
        # Main title
        cv2.putText(image, "SMART MIRROR - VIRTUAL TRY-ON", 
                   (width//2 - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Current mode indicator
        mode_color = (0, 255, 255) if self.current_mode == "tops" else (255, 0, 255)
        cv2.putText(image, f"MODE: {self.current_mode.upper()}", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
        
        # Garment info
        garment_info = garment_manager.get_garment_info(self.current_mode)
        cv2.putText(image, f"Garment: {garment_info['index']}/{garment_info['total']}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"File: {garment_info['name']}", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Performance info
        cv2.putText(image, f"FPS: {self.current_fps:.1f}", 
                   (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw control buttons
        button_y = height // 2 - 40
        button_margin = 30
        
        # Right button (next)
        if self.button_right is not None:
            overlay_image_alpha(image, self.button_right, 
                              width - self.button_right.shape[1] - button_margin, button_y)
        
        # Left button (previous)
        if self.button_left is not None:
            overlay_image_alpha(image, self.button_left, button_margin, button_y)
        
        # Mode switch area (center bottom)
        mode_switch_y = height - 100
        cv2.rectangle(image, (width//2 - 100, mode_switch_y), 
                     (width//2 + 100, mode_switch_y + 60), (50, 50, 50), -1)
        cv2.rectangle(image, (width//2 - 100, mode_switch_y), 
                     (width//2 + 100, mode_switch_y + 60), (255, 255, 255), 2)
        cv2.putText(image, "SWITCH MODE", (width//2 - 70, mode_switch_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"{self.current_mode} → {'bottoms' if self.current_mode == 'tops' else 'tops'}", 
                   (width//2 - 60, mode_switch_y + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Instructions
        instructions = [
            "CONTROLS:",
            "• Point at buttons to change garments",
            "• Point at center to switch mode (tops/bottoms)",
            "• Hold gesture for 2 seconds to activate",
            "• Press T/B to switch modes, S for screenshot, R to reset, ESC to exit"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = height - 200 + (i * 25)
            color = (255, 255, 0) if i == 0 else (255, 255, 255)
            thickness = 2 if i == 0 else 1
            cv2.putText(image, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        
        return image

    def process_gestures(self, image: np.ndarray, hands_results) -> np.ndarray:
        """Process hand gestures for Smart Mirror control"""
        if not hands_results.multi_hand_landmarks:
            # Reset all counters when no hands detected
            self.gesture_counters = {"left": 0, "right": 0, "mode_switch": 0}
            return image
        
        height, width = image.shape[:2]
        
        for hand_landmarks in hands_results.multi_hand_landmarks:
            # Get finger tip positions
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            
            # Check if finger is extended (pointing gesture)
            finger_extended = index_tip.y < index_mcp.y
            
            if finger_extended:
                # Get average position for stability
                x = int((index_tip.x + middle_tip.x) / 2 * width)
                y = int((index_tip.y + middle_tip.y) / 2 * height)
                
                # Draw finger indicator
                cv2.circle(image, (x, y), 15, (0, 255, 255), -1)
                cv2.circle(image, (x, y), 10, (255, 255, 0), -1)
                
                # Check which area is being pointed at
                button_y = height // 2 - 40
                button_margin = 30
                
                # Right button area
                if (width - 110 <= x <= width - button_margin and 
                    button_y <= y <= button_y + 80):
                    self.gesture_counters["right"] += 1
                    self.gesture_counters["left"] = 0
                    self.gesture_counters["mode_switch"] = 0
                    
                    # Draw progress indicator
                    progress = min(self.gesture_counters["right"] / self.selection_threshold, 1.0)
                    progress_angle = int(progress * 360)
                    center = (width - 70, button_y + 40)
                    cv2.ellipse(image, center, (50, 50), 0, 0, progress_angle, (0, 255, 0), 6)
                    
                    # Show progress percentage
                    cv2.putText(image, f"{int(progress * 100)}%", 
                               (center[0] - 20, center[1] + 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    if self.gesture_counters["right"] >= self.selection_threshold:
                        garment_manager.next_garment(self.current_mode)
                        self.gesture_counters["right"] = 0
                        print(f"Next {self.current_mode} selected")
                
                # Left button area
                elif (button_margin <= x <= button_margin + 80 and 
                      button_y <= y <= button_y + 80):
                    self.gesture_counters["left"] += 1
                    self.gesture_counters["right"] = 0
                    self.gesture_counters["mode_switch"] = 0
                    
                    # Draw progress indicator
                    progress = min(self.gesture_counters["left"] / self.selection_threshold, 1.0)
                    progress_angle = int(progress * 360)
                    center = (70, button_y + 40)
                    cv2.ellipse(image, center, (50, 50), 0, 0, progress_angle, (0, 255, 0), 6)
                    
                    # Show progress percentage
                    cv2.putText(image, f"{int(progress * 100)}%", 
                               (center[0] - 20, center[1] + 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    if self.gesture_counters["left"] >= self.selection_threshold:
                        garment_manager.previous_garment(self.current_mode)
                        self.gesture_counters["left"] = 0
                        print(f"Previous {self.current_mode} selected")
                
                # Mode switch area
                elif (width//2 - 100 <= x <= width//2 + 100 and 
                      height - 100 <= y <= height - 40):
                    self.gesture_counters["mode_switch"] += 1
                    self.gesture_counters["left"] = 0
                    self.gesture_counters["right"] = 0
                    
                    # Draw progress indicator
                    progress = min(self.gesture_counters["mode_switch"] / self.mode_switch_threshold, 1.0)
                    progress_angle = int(progress * 360)
                    center = (width//2, height - 70)
                    cv2.ellipse(image, center, (60, 30), 0, 0, progress_angle, (255, 0, 255), 4)
                    
                    # Show progress percentage
                    cv2.putText(image, f"{int(progress * 100)}%", 
                               (center[0] - 20, center[1] + 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    
                    if self.gesture_counters["mode_switch"] >= self.mode_switch_threshold:
                        self.current_mode = "bottoms" if self.current_mode == "tops" else "tops"
                        self.gesture_counters["mode_switch"] = 0
                        print(f"Switched to {self.current_mode} mode")
                
                else:
                    # Reset counters if pointing elsewhere
                    self.gesture_counters = {"left": 0, "right": 0, "mode_switch": 0}
            else:
                # Reset counters if finger not extended
                self.gesture_counters = {"left": 0, "right": 0, "mode_switch": 0}
        
        return image

# Garment Rendering System with Caching
class GarmentRenderer:
    def __init__(self):
        self.garment_cache = {}
        self.max_cache_size = 10
    
    def load_garment(self, path: str, size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Load and cache garment with specified size"""
        cache_key = f"{path}_{size[0]}x{size[1]}"
        
        if cache_key in self.garment_cache:
            return self.garment_cache[cache_key]
        
        try:
            garment = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if garment is None:
                return None
            
            # Resize with high-quality interpolation
            garment = cv2.resize(garment, size, interpolation=cv2.INTER_LANCZOS4)
            
            # Manage cache size
            if len(self.garment_cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.garment_cache))
                del self.garment_cache[oldest_key]
            
            self.garment_cache[cache_key] = garment
            return garment
        
        except Exception as e:
            print(f"Error loading garment {path}: {e}")
            return None
    
    def render_garment(self, image: np.ndarray, garment_path: str, fitting: Dict) -> np.ndarray:
        """Render garment on image with fitting parameters"""
        if not garment_path or not fitting:
            return image
        
        size = (fitting['width'], fitting['height'])
        garment = self.load_garment(garment_path, size)
        
        if garment is not None:
            # Apply confidence-based alpha adjustment
            confidence_factor = max(0.3, min(1.0, fitting['confidence']))
            
            # Adjust garment alpha based on confidence
            if len(garment.shape) == 4:  # Has alpha channel
                garment[:, :, 3] = (garment[:, :, 3] * confidence_factor).astype(np.uint8)
            
            image = overlay_image_alpha(image, garment, fitting['x'], fitting['y'])
        
        return image

# Initialize system components
smart_mirror_ui = SmartMirrorUI()
garment_renderer = GarmentRenderer()

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
            return self._calculate_top_fitting(coords, iw, ih)
        elif garment_type == "bottoms":
            return self._calculate_bottom_fitting(coords, iw, ih)
        else:
            return None
    
    def _calculate_top_fitting(self, coords: Dict, iw: int, ih: int) -> Dict:
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
    
    def _calculate_bottom_fitting(self, coords: Dict, iw: int, ih: int) -> Dict:
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

body_tracker = BodyTracker()

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

    frame_count += 1
    
    # Calculate FPS every 30 frames
    if frame_count % 30 == 0:
        current_time = time.time()
        fps = 30 / (current_time - start_time)
        start_time = current_time
    
    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find poses and hands
    pose_results = pose.process(image_rgb)
    hands_results = hands.process(image_rgb)
    
    # Track pose detection confidence
    if pose_results.pose_landmarks:
        # Calculate detection confidence (if available)
        avg_confidence = 1.0  # Default high confidence
        if hasattr(pose_results, 'pose_world_landmarks') and pose_results.pose_world_landmarks:
            confidences = [lm.visibility for lm in pose_results.pose_world_landmarks.landmark if hasattr(lm, 'visibility')]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
        
        pose_detection_confidence.append(avg_confidence)
        if len(pose_detection_confidence) > 30:  # Keep last 30 frames
            pose_detection_confidence.pop(0)
    
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
        
        # Calculate shirt dimensions with improved accuracy and smoothing
        shoulder_width = abs(lm11_px[0] - lm12_px[0])
        body_length = abs((lm11_px[1] + lm12_px[1]) // 2 - (lm23_px[1] + lm24_px[1]) // 2)
        
        # Ensure reasonable body proportions
        if body_length < shoulder_width * 0.5:
            body_length = int(shoulder_width * 1.2)
        
        shirt_width = int(shoulder_width * fixedRatio)
        shirt_height = int(body_length * 1.3)  # Dynamic height based on body proportions
        
        # Better positioning with center alignment
        center_x = (lm11_px[0] + lm12_px[0]) // 2
        new_shirt_top_left = (
            max(0, min(iw - shirt_width, center_x - shirt_width // 2)),
            max(0, min(ih - shirt_height, min(lm11_px[1], lm12_px[1]) - int(shirt_height * 0.15)))
        )
        
        # Apply smoothing for stable shirt positioning
        if smooth_shirt_pos is None:
            smooth_shirt_pos = new_shirt_top_left
            smooth_shirt_size = (shirt_width, shirt_height)
        else:
            smooth_shirt_pos = (
                int(smooth_shirt_pos[0] * (1 - smoothing_factor) + new_shirt_top_left[0] * smoothing_factor),
                int(smooth_shirt_pos[1] * (1 - smoothing_factor) + new_shirt_top_left[1] * smoothing_factor)
            )
            smooth_shirt_size = (
                int(smooth_shirt_size[0] * (1 - smoothing_factor) + shirt_width * smoothing_factor),
                int(smooth_shirt_size[1] * (1 - smoothing_factor) + shirt_height * smoothing_factor)
            )
        
        shirt_top_left = smooth_shirt_pos
        shirt_width, shirt_height = smooth_shirt_size
        
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
    else:
        # Reset smoothing if no pose detected
        smooth_shirt_pos = None
        smooth_shirt_size = None
    
    # Enhanced UI with current shirt information and performance metrics
    ui_color = (255, 255, 255)
    cv2.putText(image, f"Shirt: {imageNumber + 1}/{len(listShirts)}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, ui_color, 2)
    cv2.putText(image, f"Current: {listShirts[imageNumber]}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ui_color, 2)
    
    # Performance indicators
    pose_status = "✓ Pose Detected" if pose_results.pose_landmarks else "✗ No Pose"
    cv2.putText(image, pose_status, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                (0, 255, 0) if pose_results.pose_landmarks else (0, 0, 255), 2)
    
    # Show average confidence if available
    if pose_detection_confidence:
        avg_conf = sum(pose_detection_confidence) / len(pose_detection_confidence)
        cv2.putText(image, f"Confidence: {avg_conf:.2f}", (10, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_color, 1)
    
    # Show FPS
    cv2.putText(image, f"FPS: {fps:.1f}", (10, 170), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_color, 1)
    
    cv2.putText(image, "Point at buttons to change shirts | 'r' to reset | 'q'/'Esc' to quit", 
                (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_color, 1)
    
    # Overlay buttons with better positioning
    button_margin = 20
    button_y = image.shape[0] // 2 - imgButtonRight.shape[0] // 2
    image = overlay_image_alpha(image, imgButtonRight, 
                                image.shape[1] - imgButtonRight.shape[1] - button_margin, button_y)
    image = overlay_image_alpha(image, imgButtonLeft, button_margin, button_y)
    
    # Enhanced hand gesture detection with multiple finger validation
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            # Get multiple finger tip landmarks for better accuracy
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            
            # Check if finger is extended (pointing gesture validation)
            finger_extended = index_finger_tip.y < index_finger_mcp.y
            
            if finger_extended:
                # Use average position for more stable detection
                x = int((index_finger_tip.x + middle_finger_tip.x) / 2 * image.shape[1])
                y = int((index_finger_tip.y + middle_finger_tip.y) / 2 * image.shape[0])
                
                # Draw finger position for visual feedback
                cv2.circle(image, (x, y), 12, (0, 255, 255), -1)
                cv2.circle(image, (x, y), 8, (255, 255, 0), -1)

                # Right button detection with improved bounds
                button_right_x_start = image.shape[1] - imgButtonRight.shape[1] - button_margin
                button_right_x_end = image.shape[1] - button_margin
                button_y_start = button_y
                button_y_end = button_y + imgButtonRight.shape[0]
                
                # Left button detection with improved bounds
                button_left_x_start = button_margin
                button_left_x_end = button_margin + imgButtonLeft.shape[1]
                
                if (button_right_x_start <= x <= button_right_x_end and 
                    button_y_start <= y <= button_y_end):
                    counterRight += 1
                    progress_angle = min(counterRight * selectionSpeed, 360)
                    cv2.ellipse(image, (button_right_x_start + imgButtonRight.shape[1] // 2, 
                                      button_y + imgButtonRight.shape[0] // 2), 
                               (45, 45), 0, 0, progress_angle, (0, 255, 0), 6)
                    
                    # Add progress text
                    progress_pct = int((progress_angle / 360) * 100)
                    cv2.putText(image, f"{progress_pct}%", 
                               (button_right_x_start + 10, button_y + imgButtonRight.shape[0] + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    if progress_angle >= 360:
                        counterRight = 0
                        imageNumber = (imageNumber + 1) % len(listShirts)
                        print(f"Switched to next shirt: {listShirts[imageNumber]}")
                
                elif (button_left_x_start <= x <= button_left_x_end and 
                      button_y_start <= y <= button_y_end):
                    counterLeft += 1
                    progress_angle = min(counterLeft * selectionSpeed, 360)
                    cv2.ellipse(image, (button_left_x_start + imgButtonLeft.shape[1] // 2, 
                                      button_y + imgButtonLeft.shape[0] // 2), 
                               (45, 45), 0, 0, progress_angle, (0, 255, 0), 6)
                    
                    # Add progress text
                    progress_pct = int((progress_angle / 360) * 100)
                    cv2.putText(image, f"{progress_pct}%", 
                               (button_left_x_start + 10, button_y + imgButtonLeft.shape[0] + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    if progress_angle >= 360:
                        counterLeft = 0
                        imageNumber = (imageNumber - 1) if imageNumber > 0 else len(listShirts) - 1
                        print(f"Switched to previous shirt: {listShirts[imageNumber]}")
                
                else:
                    counterRight = 0
                    counterLeft = 0
            else:
                counterRight = 0
                counterLeft = 0
                
            # Draw hand landmarks for debugging (optional)
            # mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        # No hands detected, reset counters
        counterRight = 0
        counterLeft = 0

    # Enhanced keyboard controls
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # Press 'Esc' to exit
        break
    elif key == ord('r'):  # Press 'r' to reset to first shirt
        imageNumber = 0
        print(f"Reset to first shirt: {listShirts[imageNumber]}")
    elif key == ord('q'):  # Press 'q' to quit
        break
    elif key == ord('n'):  # Press 'n' for next shirt
        imageNumber = (imageNumber + 1) % len(listShirts)
        print(f"Next shirt: {listShirts[imageNumber]}")
    elif key == ord('p'):  # Press 'p' for previous shirt
        imageNumber = (imageNumber - 1) if imageNumber > 0 else len(listShirts) - 1
        print(f"Previous shirt: {listShirts[imageNumber]}")
    elif key == ord('s'):  # Press 's' to save screenshot
        timestamp = int(time.time())
        screenshot_path = f"try_on_screenshot_{timestamp}.jpg"
        cv2.imwrite(screenshot_path, image)
        print(f"Screenshot saved: {screenshot_path}")
    elif key == ord('d'):  # Press 'd' to toggle debug mode
        # Toggle pose landmarks display
        pass

    cv2.imshow('Virtual Try-On', image)

print("Application closed.")
cap.release()
cv2.destroyAllWindows()
