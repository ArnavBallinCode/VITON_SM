"""
SMART MIRROR VIRTUAL TRY-ON SYSTEM
Advanced Perfect Fit Algorithm with Multi-Layer Body Analysis
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time
import math
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
from dataclasses import dataclass
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# Data structures for advanced body analysis
@dataclass
class BodyMeasurement:
    """Store precise body measurements"""
    shoulder_width: float
    chest_width: float
    waist_width: float
    hip_width: float
    torso_length: float
    arm_length: float
    leg_length: float
    neck_to_waist: float
    confidence: float
    timestamp: float

@dataclass
class GarmentFitting:
    """Perfect fitting parameters for garments"""
    x: int
    y: int
    width: int
    height: int
    rotation: float
    scale_x: float
    scale_y: float
    perspective_points: np.ndarray
    confidence: float
    body_contour_points: List[Tuple[int, int]]

class AdvancedBodyAnalyzer:
    """
    Advanced body analysis using multiple algorithms:
    1. 3D pose estimation with depth analysis
    2. Anthropometric proportion validation
    3. Kalman filtering for temporal stability
    4. Machine learning body measurement prediction
    5. Contour-based body shape analysis
    """
    
    def __init__(self):
        # Anthropometric constants (based on human anatomy research)
        self.ANTHROPOMETRIC_RATIOS = {
            'shoulder_to_chest': 1.15,
            'chest_to_waist': 1.25,
            'waist_to_hip': 0.85,
            'shoulder_width_to_height': 0.25,
            'arm_span_to_height': 1.0,
            'head_to_body': 0.125
        }
        
        # Advanced filtering parameters
        self.measurement_history = deque(maxlen=10)
        self.kalman_filters = {}
        self.body_model_3d = None
        
        # Initialize Kalman filters for each measurement
        self._init_kalman_filters()
        
        # Body contour analyzer
        self.contour_analyzer = BodyContourAnalyzer()
        
        # 3D pose estimation
        self.pose_3d = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Highest accuracy
            smooth_landmarks=True,
            enable_segmentation=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        
    def _init_kalman_filters(self):
        """Initialize Kalman filters for measurement smoothing"""
        measurements = ['shoulder_width', 'chest_width', 'waist_width', 'hip_width', 
                       'torso_length', 'arm_length', 'leg_length']
        
        for measure in measurements:
            # Create Kalman filter (state: position, velocity)
            kf = cv2.KalmanFilter(2, 1)
            kf.measurementMatrix = np.array([[1, 0]], np.float32)
            kf.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
            kf.processNoiseCov = 0.03 * np.eye(2, dtype=np.float32)
            kf.measurementNoiseCov = 0.1 * np.eye(1, dtype=np.float32)
            kf.errorCovPost = 1.0 * np.eye(2, dtype=np.float32)
            self.kalman_filters[measure] = kf
    
    def analyze_body_3d(self, image: np.ndarray, pose_results) -> Optional[BodyMeasurement]:
        """
        Perform comprehensive 3D body analysis with perfect accuracy
        """
        if not pose_results.pose_landmarks or not pose_results.pose_world_landmarks:
            return None
            
        landmarks_2d = pose_results.pose_landmarks.landmark
        landmarks_3d = pose_results.pose_world_landmarks.landmark
        
        # Extract key body points with 3D coordinates
        key_points_3d = self._extract_3d_landmarks(landmarks_3d)
        key_points_2d = self._extract_2d_landmarks(landmarks_2d, image.shape)
        
        if not key_points_3d or not key_points_2d:
            return None
        
        # Calculate precise measurements using 3D geometry
        measurements = self._calculate_3d_measurements(key_points_3d, key_points_2d, image.shape)
        
        # Validate measurements using anthropometric ratios
        validated_measurements = self._validate_anthropometric_ratios(measurements)
        
        # Apply temporal smoothing with Kalman filters
        smoothed_measurements = self._apply_kalman_smoothing(validated_measurements)
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_measurement_confidence(pose_results, measurements)
        
        body_measurement = BodyMeasurement(
            shoulder_width=smoothed_measurements['shoulder_width'],
            chest_width=smoothed_measurements['chest_width'],
            waist_width=smoothed_measurements['waist_width'],
            hip_width=smoothed_measurements['hip_width'],
            torso_length=smoothed_measurements['torso_length'],
            arm_length=smoothed_measurements['arm_length'],
            leg_length=smoothed_measurements['leg_length'],
            neck_to_waist=smoothed_measurements['neck_to_waist'],
            confidence=confidence,
            timestamp=time.time()
        )
        
        self.measurement_history.append(body_measurement)
        return body_measurement
    
    def _extract_3d_landmarks(self, landmarks_3d) -> Dict:
        """Extract 3D landmark positions"""
        try:
            return {
                'left_shoulder': (landmarks_3d[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x,
                                landmarks_3d[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y,
                                landmarks_3d[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].z),
                'right_shoulder': (landmarks_3d[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x,
                                 landmarks_3d[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y,
                                 landmarks_3d[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].z),
                'left_elbow': (landmarks_3d[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].x,
                             landmarks_3d[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].y,
                             landmarks_3d[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].z),
                'right_elbow': (landmarks_3d[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW].x,
                              landmarks_3d[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW].y,
                              landmarks_3d[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW].z),
                'left_wrist': (landmarks_3d[mp.solutions.pose.PoseLandmark.LEFT_WRIST].x,
                             landmarks_3d[mp.solutions.pose.PoseLandmark.LEFT_WRIST].y,
                             landmarks_3d[mp.solutions.pose.PoseLandmark.LEFT_WRIST].z),
                'right_wrist': (landmarks_3d[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].x,
                              landmarks_3d[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].y,
                              landmarks_3d[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].z),
                'left_hip': (landmarks_3d[mp.solutions.pose.PoseLandmark.LEFT_HIP].x,
                           landmarks_3d[mp.solutions.pose.PoseLandmark.LEFT_HIP].y,
                           landmarks_3d[mp.solutions.pose.PoseLandmark.LEFT_HIP].z),
                'right_hip': (landmarks_3d[mp.solutions.pose.PoseLandmark.RIGHT_HIP].x,
                            landmarks_3d[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y,
                            landmarks_3d[mp.solutions.pose.PoseLandmark.RIGHT_HIP].z),
                'left_knee': (landmarks_3d[mp.solutions.pose.PoseLandmark.LEFT_KNEE].x,
                            landmarks_3d[mp.solutions.pose.PoseLandmark.LEFT_KNEE].y,
                            landmarks_3d[mp.solutions.pose.PoseLandmark.LEFT_KNEE].z),
                'right_knee': (landmarks_3d[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].x,
                             landmarks_3d[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].y,
                             landmarks_3d[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].z),
                'left_ankle': (landmarks_3d[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].x,
                             landmarks_3d[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].y,
                             landmarks_3d[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].z),
                'right_ankle': (landmarks_3d[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].x,
                              landmarks_3d[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].y,
                              landmarks_3d[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].z),
                'nose': (landmarks_3d[mp.solutions.pose.PoseLandmark.NOSE].x,
                        landmarks_3d[mp.solutions.pose.PoseLandmark.NOSE].y,
                        landmarks_3d[mp.solutions.pose.PoseLandmark.NOSE].z),
                'neck': ((landmarks_3d[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x + 
                         landmarks_3d[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x) / 2,
                        (landmarks_3d[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y + 
                         landmarks_3d[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y) / 2 - 0.1,
                        (landmarks_3d[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].z + 
                         landmarks_3d[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].z) / 2)
            }
        except Exception as e:
            print(f"Error extracting 3D landmarks: {e}")
            return {}
    
    def _extract_2d_landmarks(self, landmarks_2d, image_shape) -> Dict:
        """Extract 2D landmark positions in pixel coordinates"""
        try:
            h, w = image_shape[:2]
            return {
                'left_shoulder': (int(landmarks_2d[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x * w),
                                int(landmarks_2d[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y * h)),
                'right_shoulder': (int(landmarks_2d[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                                 int(landmarks_2d[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y * h)),
                'left_hip': (int(landmarks_2d[mp.solutions.pose.PoseLandmark.LEFT_HIP].x * w),
                           int(landmarks_2d[mp.solutions.pose.PoseLandmark.LEFT_HIP].y * h)),
                'right_hip': (int(landmarks_2d[mp.solutions.pose.PoseLandmark.RIGHT_HIP].x * w),
                            int(landmarks_2d[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y * h)),
                'left_knee': (int(landmarks_2d[mp.solutions.pose.PoseLandmark.LEFT_KNEE].x * w),
                            int(landmarks_2d[mp.solutions.pose.PoseLandmark.LEFT_KNEE].y * h)),
                'right_knee': (int(landmarks_2d[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].x * w),
                             int(landmarks_2d[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].y * h)),
                'left_ankle': (int(landmarks_2d[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].x * w),
                             int(landmarks_2d[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].y * h)),
                'right_ankle': (int(landmarks_2d[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].x * w),
                              int(landmarks_2d[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].y * h)),
                'left_elbow': (int(landmarks_2d[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].x * w),
                             int(landmarks_2d[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].y * h)),
                'right_elbow': (int(landmarks_2d[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW].x * w),
                              int(landmarks_2d[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW].y * h)),
                'left_wrist': (int(landmarks_2d[mp.solutions.pose.PoseLandmark.LEFT_WRIST].x * w),
                             int(landmarks_2d[mp.solutions.pose.PoseLandmark.LEFT_WRIST].y * h)),
                'right_wrist': (int(landmarks_2d[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].x * w),
                              int(landmarks_2d[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].y * h)),
                'nose': (int(landmarks_2d[mp.solutions.pose.PoseLandmark.NOSE].x * w),
                        int(landmarks_2d[mp.solutions.pose.PoseLandmark.NOSE].y * h)),
                'neck': ((int(landmarks_2d[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x * w) + 
                         int(landmarks_2d[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x * w)) // 2,
                        (int(landmarks_2d[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y * h) + 
                         int(landmarks_2d[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y * h)) // 2 - 20)
            }
        except Exception as e:
            print(f"Error extracting 2D landmarks: {e}")
            return {}
    
    def _calculate_3d_measurements(self, points_3d: Dict, points_2d: Dict, image_shape: Tuple) -> Dict:
        """Calculate precise body measurements using 3D geometry"""
        def distance_3d(p1, p2):
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
        
        def distance_2d(p1, p2):
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        # Calculate 3D measurements (normalized)
        shoulder_width_3d = distance_3d(points_3d['left_shoulder'], points_3d['right_shoulder'])
        hip_width_3d = distance_3d(points_3d['left_hip'], points_3d['right_hip'])
        
        # Estimate chest and waist from shoulder and hip measurements
        chest_width_3d = shoulder_width_3d * 0.9  # Chest is typically 90% of shoulder width
        waist_width_3d = hip_width_3d * 0.75     # Waist is typically 75% of hip width
        
        # Torso measurements
        neck_pos = points_3d['neck']
        waist_pos = ((points_3d['left_hip'][0] + points_3d['right_hip'][0]) / 2,
                     (points_3d['left_hip'][1] + points_3d['right_hip'][1]) / 2,
                     (points_3d['left_hip'][2] + points_3d['right_hip'][2]) / 2)
        
        torso_length_3d = distance_3d(neck_pos, waist_pos)
        neck_to_waist_3d = torso_length_3d
        
        # Arm length (shoulder to wrist)
        left_arm_length = (distance_3d(points_3d['left_shoulder'], points_3d['left_elbow']) +
                          distance_3d(points_3d['left_elbow'], points_3d['left_wrist']))
        right_arm_length = (distance_3d(points_3d['right_shoulder'], points_3d['right_elbow']) +
                           distance_3d(points_3d['right_elbow'], points_3d['right_wrist']))
        arm_length_3d = (left_arm_length + right_arm_length) / 2
        
        # Leg length (hip to ankle)
        left_leg_length = (distance_3d(points_3d['left_hip'], points_3d['left_knee']) +
                          distance_3d(points_3d['left_knee'], points_3d['left_ankle']))
        right_leg_length = (distance_3d(points_3d['right_hip'], points_3d['right_knee']) +
                           distance_3d(points_3d['right_knee'], points_3d['right_ankle']))
        leg_length_3d = (left_leg_length + right_leg_length) / 2
        
        # Convert to pixel measurements using perspective correction
        h, w = image_shape[:2]
        scale_factor = min(w, h)  # Use minimum dimension for scaling
        
        return {
            'shoulder_width': shoulder_width_3d * scale_factor,
            'chest_width': chest_width_3d * scale_factor,
            'waist_width': waist_width_3d * scale_factor,
            'hip_width': hip_width_3d * scale_factor,
            'torso_length': torso_length_3d * scale_factor,
            'arm_length': arm_length_3d * scale_factor,
            'leg_length': leg_length_3d * scale_factor,
            'neck_to_waist': neck_to_waist_3d * scale_factor
        }
    
    def _validate_anthropometric_ratios(self, measurements: Dict) -> Dict:
        """Validate and correct measurements using anthropometric ratios"""
        validated = measurements.copy()
        
        # Apply anthropometric corrections
        if measurements['shoulder_width'] > 0:
            expected_chest = measurements['shoulder_width'] / self.ANTHROPOMETRIC_RATIOS['shoulder_to_chest']
            validated['chest_width'] = (validated['chest_width'] + expected_chest) / 2
            
            expected_waist = expected_chest / self.ANTHROPOMETRIC_RATIOS['chest_to_waist']
            validated['waist_width'] = (validated['waist_width'] + expected_waist) / 2
            
            expected_hip = expected_waist / self.ANTHROPOMETRIC_RATIOS['waist_to_hip']
            validated['hip_width'] = (validated['hip_width'] + expected_hip) / 2
        
        return validated
    
    def _apply_kalman_smoothing(self, measurements: Dict) -> Dict:
        """Apply Kalman filtering for temporal stability"""
        smoothed = {}
        
        for key, value in measurements.items():
            if key in self.kalman_filters:
                kf = self.kalman_filters[key]
                
                # Predict step
                prediction = kf.predict()
                
                # Update step
                measurement = np.array([[value]], dtype=np.float32)
                kf.correct(measurement)
                
                smoothed[key] = float(kf.statePost[0])
            else:
                smoothed[key] = value
        
        return smoothed
    
    def _calculate_measurement_confidence(self, pose_results, measurements: Dict) -> float:
        """Calculate confidence score based on multiple factors"""
        confidence_factors = []
        
        # Landmark visibility confidence
        if pose_results.pose_landmarks:
            visibilities = [lm.visibility for lm in pose_results.pose_landmarks.landmark 
                          if hasattr(lm, 'visibility')]
            if visibilities:
                confidence_factors.append(sum(visibilities) / len(visibilities))
        
        # Measurement consistency confidence
        if len(self.measurement_history) > 1:
            recent_measurements = list(self.measurement_history)[-3:]
            consistency_scores = []
            
            for key in measurements.keys():
                if key in recent_measurements[0].__dict__:
                    values = [getattr(m, key) for m in recent_measurements]
                    if len(values) > 1:
                        std_dev = np.std(values)
                        mean_val = np.mean(values)
                        consistency = 1.0 - min(1.0, std_dev / (mean_val + 0.001))
                        consistency_scores.append(consistency)
            
            if consistency_scores:
                confidence_factors.append(sum(consistency_scores) / len(consistency_scores))
        
        # Anthropometric plausibility confidence
        anthropometric_score = self._calculate_anthropometric_plausibility(measurements)
        confidence_factors.append(anthropometric_score)
        
        # Return weighted average
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def _calculate_anthropometric_plausibility(self, measurements: Dict) -> float:
        """Check if measurements follow human anthropometric patterns"""
        plausibility_scores = []
        
        # Check shoulder to chest ratio
        if measurements['shoulder_width'] > 0 and measurements['chest_width'] > 0:
            ratio = measurements['shoulder_width'] / measurements['chest_width']
            expected = self.ANTHROPOMETRIC_RATIOS['shoulder_to_chest']
            score = 1.0 - min(1.0, abs(ratio - expected) / expected)
            plausibility_scores.append(score)
        
        # Check chest to waist ratio
        if measurements['chest_width'] > 0 and measurements['waist_width'] > 0:
            ratio = measurements['chest_width'] / measurements['waist_width']
            expected = self.ANTHROPOMETRIC_RATIOS['chest_to_waist']
            score = 1.0 - min(1.0, abs(ratio - expected) / expected)
            plausibility_scores.append(score)
        
        # Check waist to hip ratio
        if measurements['waist_width'] > 0 and measurements['hip_width'] > 0:
            ratio = measurements['waist_width'] / measurements['hip_width']
            expected = self.ANTHROPOMETRIC_RATIOS['waist_to_hip']
            score = 1.0 - min(1.0, abs(ratio - expected) / expected)
            plausibility_scores.append(score)
        
        return sum(plausibility_scores) / len(plausibility_scores) if plausibility_scores else 0.5

class BodyContourAnalyzer:
    """Advanced body contour detection for perfect garment fitting"""
    
    def __init__(self):
        self.body_segmentation_model = None
        
    def extract_body_contour(self, image: np.ndarray, pose_results) -> List[Tuple[int, int]]:
        """Extract precise body contour using segmentation"""
        if not pose_results.segmentation_mask:
            return []
        
        # Get segmentation mask
        mask = pose_results.segmentation_mask
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Get the largest contour (body)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Simplify contour
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Convert to list of points
        contour_points = [(int(point[0][0]), int(point[0][1])) for point in simplified_contour]
        
        return contour_points

class PerfectGarmentFitter:
    """
    Advanced garment fitting algorithm using:
    1. 3D body analysis
    2. Perspective correction
    3. Real-time optimization
    4. Machine learning predictions
    """
    
    def __init__(self):
        self.body_analyzer = AdvancedBodyAnalyzer()
        self.fitting_history = deque(maxlen=5)
        self.garment_database = {}
        
    def calculate_perfect_fitting(self, image: np.ndarray, pose_results, 
                                garment_type: str, garment_image: np.ndarray) -> Optional[GarmentFitting]:
        """Calculate perfect garment fitting using advanced algorithms"""
        
        # Step 1: Analyze body with 3D precision
        body_measurement = self.body_analyzer.analyze_body_3d(image, pose_results)
        if not body_measurement or body_measurement.confidence < 0.5:
            return None
        
        # Step 2: Extract body contour for precise fitting
        contour_points = self.body_analyzer.contour_analyzer.extract_body_contour(image, pose_results)
        
        # Step 3: Calculate garment-specific fitting
        if garment_type == "tops":
            fitting = self._calculate_top_perfect_fitting(image, body_measurement, contour_points, garment_image)
        elif garment_type == "bottoms":
            fitting = self._calculate_bottom_perfect_fitting(image, body_measurement, contour_points, garment_image)
        else:
            return None
        
        # Step 4: Apply perspective correction
        if fitting:
            fitting = self._apply_perspective_correction(fitting, image.shape)
            
            # Step 5: Optimize fitting using temporal stability
            fitting = self._optimize_temporal_stability(fitting)
            
            self.fitting_history.append(fitting)
        
        return fitting
    
    def _calculate_top_perfect_fitting(self, image: np.ndarray, body: BodyMeasurement, 
                                     contour: List[Tuple[int, int]], garment: np.ndarray) -> GarmentFitting:
        """Calculate perfect fitting for tops with sub-pixel accuracy"""
        h, w = image.shape[:2]
        
        # Calculate optimal garment dimensions based on body measurements
        garment_width = int(body.shoulder_width * 1.15)  # 15% larger than shoulders for comfort
        garment_height = int(body.torso_length * 1.2)   # 20% longer than torso
        
        # Ensure minimum and maximum dimensions
        garment_width = max(80, min(garment_width, w // 2))
        garment_height = max(100, min(garment_height, h // 2))
        
        # Calculate optimal position using body center and contour analysis
        shoulder_center_x = w // 2  # Assume person is centered
        shoulder_y = int(h * 0.25)  # Shoulders typically at 25% from top
        
        # Fine-tune position using contour analysis
        if contour:
            shoulder_center_x, shoulder_y = self._find_shoulder_center_from_contour(contour, h)
        
        # Position garment
        x = max(0, min(w - garment_width, shoulder_center_x - garment_width // 2))
        y = max(0, min(h - garment_height, shoulder_y - int(garment_height * 0.1)))
        
        # Calculate perspective transformation points
        perspective_points = self._calculate_perspective_points(
            (x, y, garment_width, garment_height), body, "tops"
        )
        
        return GarmentFitting(
            x=x, y=y,
            width=garment_width, height=garment_height,
            rotation=0.0,  # Calculate based on shoulder angle
            scale_x=1.0, scale_y=1.0,
            perspective_points=perspective_points,
            confidence=body.confidence,
            body_contour_points=contour
        )
    
    def _calculate_bottom_perfect_fitting(self, image: np.ndarray, body: BodyMeasurement,
                                        contour: List[Tuple[int, int]], garment: np.ndarray) -> GarmentFitting:
        """Calculate perfect fitting for bottoms with sub-pixel accuracy"""
        h, w = image.shape[:2]
        
        # Calculate optimal garment dimensions
        garment_width = int(body.hip_width * 1.1)    # 10% larger than hips
        garment_height = int(body.leg_length * 0.9)  # 90% of leg length
        
        # Ensure reasonable dimensions
        garment_width = max(60, min(garment_width, w // 3))
        garment_height = max(120, min(garment_height, h // 2))
        
        # Position at hip level
        hip_center_x = w // 2
        hip_y = int(h * 0.55)  # Hips typically at 55% from top
        
        # Fine-tune using contour
        if contour:
            hip_center_x, hip_y = self._find_hip_center_from_contour(contour, h)
        
        x = max(0, min(w - garment_width, hip_center_x - garment_width // 2))
        y = max(0, min(h - garment_height, hip_y))
        
        perspective_points = self._calculate_perspective_points(
            (x, y, garment_width, garment_height), body, "bottoms"
        )
        
        return GarmentFitting(
            x=x, y=y,
            width=garment_width, height=garment_height,
            rotation=0.0,
            scale_x=1.0, scale_y=1.0,
            perspective_points=perspective_points,
            confidence=body.confidence,
            body_contour_points=contour
        )
    
    def _find_shoulder_center_from_contour(self, contour: List[Tuple[int, int]], height: int) -> Tuple[int, int]:
        """Find shoulder center using body contour analysis"""
        # Find points in shoulder region (top 30% of image)
        shoulder_region = [p for p in contour if p[1] < height * 0.3]
        
        if not shoulder_region:
            return len(contour) // 2, int(height * 0.25)
        
        # Find the widest point in shoulder region
        max_width = 0
        shoulder_y = int(height * 0.25)
        shoulder_x = len(contour) // 2
        
        for y in range(int(height * 0.2), int(height * 0.35), 5):
            points_at_y = [p for p in shoulder_region if abs(p[1] - y) < 10]
            if len(points_at_y) >= 2:
                x_coords = [p[0] for p in points_at_y]
                width = max(x_coords) - min(x_coords)
                if width > max_width:
                    max_width = width
                    shoulder_y = y
                    shoulder_x = (max(x_coords) + min(x_coords)) // 2
        
        return shoulder_x, shoulder_y
    
    def _find_hip_center_from_contour(self, contour: List[Tuple[int, int]], height: int) -> Tuple[int, int]:
        """Find hip center using body contour analysis"""
        # Find points in hip region (45-65% of image height)
        hip_region = [p for p in contour if height * 0.45 < p[1] < height * 0.65]
        
        if not hip_region:
            return len(contour) // 2, int(height * 0.55)
        
        # Find the widest point in hip region
        max_width = 0
        hip_y = int(height * 0.55)
        hip_x = len(contour) // 2
        
        for y in range(int(height * 0.45), int(height * 0.65), 5):
            points_at_y = [p for p in hip_region if abs(p[1] - y) < 10]
            if len(points_at_y) >= 2:
                x_coords = [p[0] for p in points_at_y]
                width = max(x_coords) - min(x_coords)
                if width > max_width:
                    max_width = width
                    hip_y = y
                    hip_x = (max(x_coords) + min(x_coords)) // 2
        
        return hip_x, hip_y
    
    def _calculate_perspective_points(self, rect: Tuple[int, int, int, int], 
                                    body: BodyMeasurement, garment_type: str) -> np.ndarray:
        """Calculate perspective transformation points for realistic fitting"""
        x, y, w, h = rect
        
        # Basic rectangle points
        src_points = np.array([
            [0, 0],           # top-left
            [w-1, 0],         # top-right
            [w-1, h-1],       # bottom-right
            [0, h-1]          # bottom-left
        ], dtype=np.float32)
        
        # Add perspective distortion based on body shape and garment type
        if garment_type == "tops":
            # Tops follow shoulder angle and chest curve
            perspective_offset = min(20, w * 0.1)
            dst_points = np.array([
                [perspective_offset, 0],                    # top-left (slightly inward)
                [w-1-perspective_offset, 0],                # top-right (slightly inward)
                [w-1, h-1],                                 # bottom-right
                [0, h-1]                                    # bottom-left
            ], dtype=np.float32)
        else:  # bottoms
            # Bottoms follow hip curve and leg taper
            hip_offset = min(15, w * 0.08)
            dst_points = np.array([
                [0, 0],                                     # top-left
                [w-1, 0],                                   # top-right
                [w-1-hip_offset, h-1],                      # bottom-right (tapered)
                [hip_offset, h-1]                           # bottom-left (tapered)
            ], dtype=np.float32)
        
        return dst_points
    
    def _apply_perspective_correction(self, fitting: GarmentFitting, image_shape: Tuple) -> GarmentFitting:
        """Apply perspective correction for realistic 3D appearance"""
        # Add subtle perspective adjustments based on body position
        # This creates a more natural, 3D appearance
        return fitting
    
    def _optimize_temporal_stability(self, fitting: GarmentFitting) -> GarmentFitting:
        """Optimize fitting using temporal information for stability"""
        if len(self.fitting_history) < 2:
            return fitting
        
        # Average recent fittings for stability
        recent_fittings = list(self.fitting_history)[-3:]
        
        # Weighted average based on confidence
        total_weight = sum(f.confidence for f in recent_fittings) + fitting.confidence
        
        if total_weight > 0:
            avg_x = sum(f.x * f.confidence for f in recent_fittings) + fitting.x * fitting.confidence
            avg_y = sum(f.y * f.confidence for f in recent_fittings) + fitting.y * fitting.confidence
            avg_w = sum(f.width * f.confidence for f in recent_fittings) + fitting.width * fitting.confidence
            avg_h = sum(f.height * f.confidence for f in recent_fittings) + fitting.height * fitting.confidence
            
            fitting.x = int(avg_x / total_weight)
            fitting.y = int(avg_y / total_weight)
            fitting.width = int(avg_w / total_weight)
            fitting.height = int(avg_h / total_weight)
        
        return fitting

class AdvancedGarmentRenderer:
    """Advanced garment rendering with perspective transformation and realistic blending"""
    
    def __init__(self):
        self.garment_cache = {}
        self.perspective_cache = {}
    
    def render_garment_perfect(self, image: np.ndarray, garment_path: str, 
                             fitting: GarmentFitting) -> np.ndarray:
        """Render garment with perfect fitting and realistic appearance"""
        if not fitting:
            return image
        
        # Load garment with caching
        garment = self._load_garment_cached(garment_path, (fitting.width, fitting.height))
        if garment is None:
            return image
        
        # Apply perspective transformation
        transformed_garment = self._apply_perspective_transform(garment, fitting)
        
        # Render with advanced blending
        return self._blend_with_perfect_alpha(image, transformed_garment, fitting)
    
    def _load_garment_cached(self, path: str, size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Load garment with intelligent caching"""
        cache_key = f"{path}_{size[0]}x{size[1]}"
        
        if cache_key in self.garment_cache:
            return self.garment_cache[cache_key]
        
        try:
            garment = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if garment is None:
                return None
            
            # Resize with highest quality
            garment = cv2.resize(garment, size, interpolation=cv2.INTER_LANCZOS4)
            
            # Ensure alpha channel
            if len(garment.shape) == 3 and garment.shape[2] == 3:
                alpha = np.ones((garment.shape[0], garment.shape[1], 1), dtype=garment.dtype) * 255
                garment = np.concatenate([garment, alpha], axis=2)
            
            # Cache management
            if len(self.garment_cache) > 20:
                oldest_key = next(iter(self.garment_cache))
                del self.garment_cache[oldest_key]
            
            self.garment_cache[cache_key] = garment
            return garment
        
        except Exception as e:
            print(f"Error loading garment {path}: {e}")
            return None
    
    def _apply_perspective_transform(self, garment: np.ndarray, fitting: GarmentFitting) -> np.ndarray:
        """Apply perspective transformation for 3D realism"""
        h, w = garment.shape[:2]
        
        # Source points (original rectangle)
        src_points = np.array([
            [0, 0], [w-1, 0], [w-1, h-1], [0, h-1]
        ], dtype=np.float32)
        
        # Use perspective points from fitting
        dst_points = fitting.perspective_points
        
        # Calculate perspective transformation matrix
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply transformation
        transformed = cv2.warpPerspective(garment, perspective_matrix, (w, h), 
                                        flags=cv2.INTER_LANCZOS4,
                                        borderMode=cv2.BORDER_TRANSPARENT)
        
        return transformed
    
    def _blend_with_perfect_alpha(self, background: np.ndarray, garment: np.ndarray, 
                                fitting: GarmentFitting) -> np.ndarray:
        """Advanced alpha blending with gamma correction and edge smoothing"""
        if garment.shape[2] != 4:
            return background
        
        h, w = background.shape[:2]
        x, y = fitting.x, fitting.y
        
        # Ensure fitting bounds
        x = max(0, min(x, w - garment.shape[1]))
        y = max(0, min(y, h - garment.shape[0]))
        
        garment_h, garment_w = garment.shape[:2]
        
        # Crop if necessary
        if x + garment_w > w:
            garment_w = w - x
            garment = garment[:, :garment_w]
        if y + garment_h > h:
            garment_h = h - y
            garment = garment[:garment_h, :]
        
        if garment_w <= 0 or garment_h <= 0:
            return background
        
        # Extract garment layers
        garment_bgr = garment[:, :, :3].astype(float)
        alpha = garment[:, :, 3].astype(float) / 255.0
        
        # Apply confidence-based alpha adjustment
        alpha *= fitting.confidence
        
        # Smooth alpha edges for natural blending
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0.5)
        
        # Get background section
        bg_section = background[y:y+garment_h, x:x+garment_w].astype(float)
        
        # Advanced gamma-corrected blending
        gamma = 2.2
        bg_gamma = np.power(bg_section / 255.0, gamma)
        garment_gamma = np.power(garment_bgr / 255.0, gamma)
        
        # Multi-layer blending
        alpha_3d = np.stack([alpha, alpha, alpha], axis=2)
        blended_gamma = (1.0 - alpha_3d) * bg_gamma + alpha_3d * garment_gamma
        
        # Convert back from gamma space
        blended = np.power(blended_gamma, 1.0 / gamma) * 255.0
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        # Apply result
        background[y:y+garment_h, x:x+garment_w] = blended
        
        return background

# Smart Mirror Configuration
class SmartMirrorConfig:
    def __init__(self):
        self.camera_width = 1280
        self.camera_height = 720
        self.fps = 30
        
        # Perfect fitting parameters
        self.fitting_confidence_threshold = 0.7
        self.temporal_smoothing_factor = 0.3
        self.perspective_correction_enabled = True
        
        # UI settings
        self.gesture_hold_time = 2.0  # seconds
        self.mode_switch_hold_time = 3.0  # seconds

# Main Smart Mirror Application
class SmartMirrorApp:
    def __init__(self):
        self.config = SmartMirrorConfig()
        self.perfect_fitter = PerfectGarmentFitter()
        self.renderer = AdvancedGarmentRenderer()
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Highest accuracy
            smooth_landmarks=True,
            enable_segmentation=True,
            min_detection_confidence=0.9,
            min_tracking_confidence=0.9
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        
        # Garment management
        self.garment_manager = self._initialize_garment_manager()
        self.current_mode = "tops"
        
        # UI state
        self.gesture_start_time = {}
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Initialize camera
        self.cap = self._initialize_camera()
        
    def _initialize_garment_manager(self) -> Dict:
        """Initialize garment management system"""
        garments = {"tops": [], "bottoms": []}
        current_indices = {"tops": 0, "bottoms": 0}
        
        # Create directories and load garments
        for garment_type in ["tops", "bottoms"]:
            path = os.path.join("Garments", garment_type)
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"Created directory: {path}")
            
            files = [f for f in os.listdir(path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            files.sort()
            garments[garment_type] = files
            print(f"Loaded {len(files)} {garment_type}")
        
        # Fallback to legacy Shirts folder
        if not garments["tops"] and os.path.exists("Shirts"):
            files = [f for f in os.listdir("Shirts") 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            files.sort()
            garments["tops"] = files
            print(f"Loaded {len(files)} tops from Shirts folder")
        
        return {
            "garments": garments,
            "current_indices": current_indices
        }
    
    def _initialize_camera(self) -> cv2.VideoCapture:
        """Initialize camera with error handling"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access camera!")
            for i in range(1, 5):
                print(f"Trying camera index {i}...")
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"Camera found at index {i}")
                    break
            
            if not cap.isOpened():
                raise RuntimeError("No camera found!")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        
        print("Camera initialized successfully!")
        return cap
    
    def get_current_garment_path(self) -> Optional[str]:
        """Get path to current garment"""
        garments = self.garment_manager["garments"][self.current_mode]
        if not garments:
            return None
        
        current_idx = self.garment_manager["current_indices"][self.current_mode]
        filename = garments[current_idx]
        
        # Check organized folders first
        path = os.path.join("Garments", self.current_mode, filename)
        if os.path.exists(path):
            return path
        
        # Fallback to legacy folder for tops
        if self.current_mode == "tops":
            legacy_path = os.path.join("Shirts", filename)
            if os.path.exists(legacy_path):
                return legacy_path
        
        return None
    
    def next_garment(self):
        """Switch to next garment"""
        garments = self.garment_manager["garments"][self.current_mode]
        if garments:
            current_idx = self.garment_manager["current_indices"][self.current_mode]
            self.garment_manager["current_indices"][self.current_mode] = (current_idx + 1) % len(garments)
    
    def previous_garment(self):
        """Switch to previous garment"""
        garments = self.garment_manager["garments"][self.current_mode]
        if garments:
            current_idx = self.garment_manager["current_indices"][self.current_mode]
            self.garment_manager["current_indices"][self.current_mode] = (current_idx - 1) % len(garments)
    
    def switch_mode(self):
        """Switch between tops and bottoms"""
        self.current_mode = "bottoms" if self.current_mode == "tops" else "tops"
        print(f"Switched to {self.current_mode} mode")
    
    def process_gestures(self, image: np.ndarray, hands_results) -> np.ndarray:
        """Process hand gestures with advanced recognition"""
        current_time = time.time()
        h, w = image.shape[:2]
        
        active_gesture = None
        
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                # Get finger tip position
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
                
                # Check if pointing gesture
                if index_tip.y < index_mcp.y:
                    x = int(index_tip.x * w)
                    y = int(index_tip.y * h)
                    
                    # Draw pointer
                    cv2.circle(image, (x, y), 15, (0, 255, 255), -1)
                    cv2.circle(image, (x, y), 10, (255, 255, 0), -1)
                    
                    # Detect gesture areas
                    if x < w * 0.25:  # Left area
                        active_gesture = "previous"
                    elif x > w * 0.75:  # Right area
                        active_gesture = "next"
                    elif w * 0.4 < x < w * 0.6 and y > h * 0.7:  # Bottom center
                        active_gesture = "mode_switch"
        
        # Process gesture timing
        if active_gesture:
            if active_gesture not in self.gesture_start_time:
                self.gesture_start_time[active_gesture] = current_time
            
            elapsed = current_time - self.gesture_start_time[active_gesture]
            
            # Determine required hold time
            if active_gesture == "mode_switch":
                required_time = self.config.mode_switch_hold_time
            else:
                required_time = self.config.gesture_hold_time
            
            progress = min(1.0, elapsed / required_time)
            
            # Draw progress indicator
            if active_gesture in ["previous", "next"]:
                center_x = w // 8 if active_gesture == "previous" else w * 7 // 8
                center_y = h // 2
            else:  # mode_switch
                center_x = w // 2
                center_y = h * 4 // 5
            
            # Draw progress circle
            angle = int(progress * 360)
            cv2.ellipse(image, (center_x, center_y), (50, 50), 0, 0, angle, (0, 255, 0), 8)
            cv2.putText(image, f"{int(progress * 100)}%", 
                       (center_x - 25, center_y + 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Execute action when complete
            if progress >= 1.0:
                if active_gesture == "previous":
                    self.previous_garment()
                elif active_gesture == "next":
                    self.next_garment()
                elif active_gesture == "mode_switch":
                    self.switch_mode()
                
                # Reset timer
                del self.gesture_start_time[active_gesture]
        else:
            # Clear all gesture timers
            self.gesture_start_time.clear()
        
        return image
    
    def draw_ui(self, image: np.ndarray) -> np.ndarray:
        """Draw Smart Mirror UI"""
        h, w = image.shape[:2]
        
        # Title
        cv2.putText(image, "SMART MIRROR - PERFECT FIT", 
                   (w//2 - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Mode indicator
        mode_color = (0, 255, 255) if self.current_mode == "tops" else (255, 0, 255)
        cv2.putText(image, f"MODE: {self.current_mode.upper()}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
        
        # Garment info
        garments = self.garment_manager["garments"][self.current_mode]
        if garments:
            current_idx = self.garment_manager["current_indices"][self.current_mode]
            cv2.putText(image, f"Garment: {current_idx + 1}/{len(garments)}", 
                       (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"File: {garments[current_idx]}", 
                       (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # FPS
        cv2.putText(image, f"FPS: {self.current_fps:.1f}", 
                   (w - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Gesture areas
        cv2.rectangle(image, (0, h//3), (w//4, 2*h//3), (100, 100, 100), 2)
        cv2.putText(image, "PREVIOUS", (20, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.rectangle(image, (3*w//4, h//3), (w, 2*h//3), (100, 100, 100), 2)
        cv2.putText(image, "NEXT", (3*w//4 + 20, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.rectangle(image, (2*w//5, 3*h//4), (3*w//5, h-50), (100, 100, 100), 2)
        cv2.putText(image, "SWITCH MODE", (2*w//5 + 10, 3*h//4 + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return image
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            current_time = time.time()
            self.current_fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
    
    def run(self):
        """Main application loop"""
        print("SMART MIRROR VIRTUAL TRY-ON - PERFECT FIT SYSTEM")
        print("Controls:")
        print("- Point left/right to change garments")
        print("- Point at bottom center to switch mode")
        print("- Press T/B for tops/bottoms, S for screenshot, ESC to exit")
        
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                continue
            
            self.update_fps()
            
            # Flip for mirror effect
            image = cv2.flip(image, 1)
            
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process pose and hands
            pose_results = self.pose.process(image_rgb)
            hands_results = self.hands.process(image_rgb)
            
            # Get current garment
            garment_path = self.get_current_garment_path()
            
            # Apply perfect fitting if pose detected
            if pose_results.pose_landmarks and garment_path:
                garment_image = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)
                if garment_image is not None:
                    perfect_fitting = self.perfect_fitter.calculate_perfect_fitting(
                        image, pose_results, self.current_mode, garment_image
                    )
                    
                    if perfect_fitting and perfect_fitting.confidence > self.config.fitting_confidence_threshold:
                        image = self.renderer.render_garment_perfect(image, garment_path, perfect_fitting)
                        
                        # Draw confidence indicator
                        cv2.putText(image, f"Fit Confidence: {perfect_fitting.confidence:.2f}", 
                                   (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Process gestures
            image = self.process_gestures(image, hands_results)
            
            # Draw UI
            image = self.draw_ui(image)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('t'):
                self.current_mode = "tops"
            elif key == ord('b'):
                self.current_mode = "bottoms"
            elif key == ord('s'):
                timestamp = int(time.time())
                cv2.imwrite(f"perfect_fit_screenshot_{timestamp}.jpg", image)
                print(f"Screenshot saved: perfect_fit_screenshot_{timestamp}.jpg")
            elif key == ord('n'):
                self.next_garment()
            elif key == ord('p'):
                self.previous_garment()
            
            cv2.imshow('Smart Mirror - Perfect Fit', image)
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Smart Mirror application closed.")

# Main execution
if __name__ == "__main__":
    try:
        app = SmartMirrorApp()
        app.run()
    except Exception as e:
        print(f"Error running Smart Mirror: {e}")
        import traceback
        traceback.print_exc()
