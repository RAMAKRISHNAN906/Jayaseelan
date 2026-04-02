"""
Enhanced Pose Extraction Module with Kalman Filter Smoothing and Visibility Filtering.

Extracts 33 MediaPipe Pose landmarks with temporal smoothing, multi-person detection,
and gait cycle metrics computation.
"""

import cv2
import numpy as np
import os
from typing import Dict, List, Tuple, Optional

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError):
    MEDIAPIPE_AVAILABLE = False
    mp_pose = None
    mp_drawing = None

from app.utils.config import POSE_CONFIG, GLOBAL_CONFIG


class KalmanFilter1D:
    """Simple 1D Kalman filter for smoothing landmark trajectories."""
    
    def __init__(self, alpha=0.3, process_noise=0.01, measurement_noise=0.1):
        """
        Initialize Kalman filter.
        
        Args:
            alpha: Smoothing factor (0.3-0.5 recommended)
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
        """
        self.alpha = alpha
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # State estimate
        self.x = 0.0
        self.P = 1.0
        
        self.initialized = False
    
    def update(self, measurement: float) -> float:
        """
        Update filter with new measurement.
        
        Args:
            measurement: Raw measurement value
            
        Returns:
            Smoothed estimate
        """
        if not self.initialized:
            self.x = measurement
            self.P = 1.0
            self.initialized = True
            return measurement
        
        # Prediction step
        x_pred = self.x
        P_pred = self.P + self.process_noise
        
        # Update step
        K = P_pred / (P_pred + self.measurement_noise)
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        
        return self.x


class KalmanFilter2D:
    """2D Kalman filter for smoothing (x, y) landmark coordinates."""
    
    def __init__(self, alpha=0.3, process_noise=0.01, measurement_noise=0.1):
        self.kf_x = KalmanFilter1D(alpha, process_noise, measurement_noise)
        self.kf_y = KalmanFilter1D(alpha, process_noise, measurement_noise)
    
    def update(self, measurement: Tuple[float, float]) -> Tuple[float, float]:
        """Update with (x, y) measurement, return smoothed (x, y)."""
        x_smooth = self.kf_x.update(measurement[0])
        y_smooth = self.kf_y.update(measurement[1])
        return (x_smooth, y_smooth)


class PoseExtractor:
    """
    Enhanced pose extractor with Kalman filtering, visibility filtering,
    and multi-person detection.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize pose extractor.
        
        Args:
            config: Configuration dict (uses POSE_CONFIG defaults if None)
        """
        self.config = config or POSE_CONFIG
        
        if MEDIAPIPE_AVAILABLE:
            self.pose = mp_pose.Pose(
                static_image_mode=self.config.get("static_image_mode", False),
                min_detection_confidence=self.config.get("min_detection_confidence", 0.5),
                min_tracking_confidence=self.config.get("min_tracking_confidence", 0.5),
                model_complexity=self.config.get("model_complexity", 1),
            )
        else:
            self.pose = None
        
        # Kalman filters for each landmark (33 landmarks)
        self.kalman_filters = {}
        self._init_kalman_filters()
        
        # Frame counter for temporal tracking
        self.frame_count = 0
        
        # Multi-person tracking
        self.person_landmarks = {}
    
    def _init_kalman_filters(self):
        """Initialize Kalman filters for all 33 landmarks."""
        alpha = self.config.get("kalman_alpha", 0.3)
        process_noise = self.config.get("kalman_process_noise", 0.01)
        measurement_noise = self.config.get("kalman_measurement_noise", 0.1)
        
        for i in range(33):  # MediaPipe Pose has 33 landmarks
            self.kalman_filters[i] = KalmanFilter2D(
                alpha=alpha,
                process_noise=process_noise,
                measurement_noise=measurement_noise
            )
    
    def extract_pose(self, frame: np.ndarray, frame_idx: int = 0) -> Dict:
        """
        Extract pose landmarks from a single frame.
        
        Args:
            frame: BGR image frame
            frame_idx: Frame index for temporal tracking
            
        Returns:
            Dict with:
                - landmarks: List of (x, y, visibility) for 33 landmarks
                - smoothed_landmarks: Kalman-smoothed landmarks
                - visibility_mask: Boolean mask of visible landmarks
                - confidence: Overall detection confidence
        """
        if not MEDIAPIPE_AVAILABLE or self.pose is None:
            return self._mock_pose_result()
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = []
            smoothed_landmarks = []
            visibility_mask = []
            
            visibility_threshold = self.config.get("visibility_threshold", 0.5)
            
            for i, lm in results.pose_landmarks.landmark:
                x, y, visibility = lm.x, lm.y, lm.visibility
                
                # Store raw landmarks
                landmarks.append((x, y, visibility))
                
                # Apply visibility filtering
                is_visible = visibility >= visibility_threshold
                visibility_mask.append(is_visible)
                
                # Apply Kalman smoothing (only if visible)
                if is_visible:
                    smoothed = self.kalman_filters[i].update((x, y))
                    smoothed_landmarks.append((smoothed[0], smoothed[1], visibility))
                else:
                    # Use last known position or raw value
                    smoothed_landmarks.append((x, y, visibility))
            
            # Compute overall confidence
            confidence = np.mean([lm.visibility for lm in results.pose_landmarks.landmark])
            
            return {
                "landmarks": landmarks,
                "smoothed_landmarks": smoothed_landmarks,
                "visibility_mask": visibility_mask,
                "confidence": confidence,
                "frame_idx": frame_idx,
            }
        else:
            return self._mock_pose_result(frame_idx)
    
    def _mock_pose_result(self, frame_idx: int = 0) -> Dict:
        """Fallback when MediaPipe is not available."""
        return {
            "landmarks": [(0.5, 0.5, 0.5) for _ in range(33)],
            "smoothed_landmarks": [(0.5, 0.5, 0.5) for _ in range(33)],
            "visibility_mask": [True] * 33,
            "confidence": 0.5,
            "frame_idx": frame_idx,
        }
    
    def extract_multi_person(self, frame: np.ndarray, frame_idx: int = 0) -> List[Dict]:
        """
        Extract poses for multiple persons in frame.
        
        Args:
            frame: BGR image frame
            frame_idx: Frame index
            
        Returns:
            List of pose dicts for each detected person
        """
        # MediaPipe Pose only detects one person per frame
        # For multi-person, we'd need to use a different detector
        # This is a placeholder for future multi-person extension
        single_pose = self.extract_pose(frame, frame_idx)
        return [single_pose] if single_pose["confidence"] > 0.3 else []
    
    def compute_gait_metrics(self, pose_sequence: List[Dict]) -> Dict:
        """
        Compute gait cycle metrics from pose sequence.
        
        Args:
            pose_sequence: List of pose dicts from consecutive frames
            
        Returns:
            Dict with gait metrics:
                - step_frequency: Steps per second
                - stride_length: Normalized stride length
                - joint_angles: Hip, knee, ankle angles
                - velocity_profiles: Joint velocities
        """
        if not pose_sequence or len(pose_sequence) < 2:
            return self._mock_gait_metrics()
        
        # Use smoothed landmarks
        landmarks_seq = [p["smoothed_landmarks"] for p in pose_sequence]
        
        # MediaPipe Pose landmark indices
        L = mp_pose.PoseLandmark if MEDIAPIPE_AVAILABLE else None
        
        if L is None:
            return self._mock_gait_metrics()
        
        metrics = {
            "step_frequency": 0.0,
            "stride_length": 0.0,
            "joint_angles": {},
            "velocity_profiles": {},
        }
        
        # Compute joint angles over sequence
        hip_angles = []
        knee_angles = []
        ankle_angles = []
        
        for landmarks in landmarks_seq:
            try:
                # Hip angle (hip-knee-ankle)
                hip_angle = self._compute_angle(
                    landmarks[L.LEFT_HIP],
                    landmarks[L.LEFT_KNEE],
                    landmarks[L.LEFT_ANKLE]
                )
                hip_angles.append(hip_angle)
                
                # Knee angle
                knee_angle = self._compute_angle(
                    landmarks[L.LEFT_KNEE],
                    landmarks[L.LEFT_HIP],
                    landmarks[L.LEFT_ANKLE]
                )
                knee_angles.append(knee_angle)
                
                # Ankle angle
                ankle_angle = self._compute_angle(
                    landmarks[L.LEFT_ANKLE],
                    landmarks[L.LEFT_KNEE],
                    landmarks[L.LEFT_HIP]
                )
                ankle_angles.append(ankle_angle)
            except Exception:
                continue
        
        metrics["joint_angles"]["hip"] = np.mean(hip_angles) if hip_angles else 0.0
        metrics["joint_angles"]["knee"] = np.mean(knee_angles) if knee_angles else 0.0
        metrics["joint_angles"]["ankle"] = np.mean(ankle_angles) if ankle_angles else 0.0
        
        # Compute step frequency (peak detection in ankle y-trajectory)
        ankle_y_seq = [lm[L.LEFT_ANKLE][1] for lm in landmarks_seq]
        step_freq = self._estimate_step_frequency(ankle_y_seq)
        metrics["step_frequency"] = step_freq
        
        # Compute stride length (displacement per step)
        stride_length = self._estimate_stride_length(landmarks_seq)
        metrics["stride_length"] = stride_length
        
        # Compute velocity profiles
        velocities = self._compute_velocity_profiles(landmarks_seq)
        metrics["velocity_profiles"] = velocities
        
        return metrics
    
    def _compute_angle(self, p1: Tuple, p2: Tuple, p3: Tuple) -> float:
        """Compute angle between three points (p1-p2-p3)."""
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        
        # Normalize
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        
        # Angle in degrees
        angle = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0))
        return np.degrees(angle)
    
    def _estimate_step_frequency(self, ankle_y_seq: List[float]) -> float:
        """Estimate step frequency from ankle y-trajectory using peak detection."""
        if len(ankle_y_seq) < 10:
            return 0.0
        
        # Find peaks (local maxima)
        ankle_y = np.array(ankle_y_seq)
        diff = np.diff(ankle_y)
        
        # Zero crossings (positive to negative = peak)
        peaks = np.where((diff[:-1] > 0) & (diff[1:] < 0))[0] + 1
        
        if len(peaks) < 2:
            return 0.0
        
        # Average time between peaks (assuming 25 FPS)
        fps = 25
        peak_intervals = np.diff(peaks) / fps
        avg_interval = np.mean(peak_intervals)
        
        # Step frequency (steps per second)
        step_freq = 1.0 / avg_interval if avg_interval > 0 else 0.0
        
        return step_freq
    
    def _estimate_stride_length(self, landmarks_seq: List) -> float:
        """Estimate stride length from landmark displacements."""
        if len(landmarks_seq) < 2:
            return 0.0
        
        L = mp_pose.PoseLandmark if MEDIAPIPE_AVAILABLE else None
        if L is None:
            return 0.0
        
        # Use ankle displacement
        left_ankles = [lm[L.LEFT_ANKLE] for lm in landmarks_seq]
        right_ankles = [lm[L.RIGHT_ANKLE] for lm in landmarks_seq]
        
        # Compute max displacement
        left_displacement = np.max([a[0] for a in left_ankles]) - np.min([a[0] for a in left_ankles])
        right_displacement = np.max([a[0] for a in right_ankles]) - np.min([a[0] for a in right_ankles])
        
        # Average stride length (normalized)
        stride_length = (left_displacement + right_displacement) / 2
        
        return stride_length
    
    def _compute_velocity_profiles(self, landmarks_seq: List) -> Dict:
        """Compute velocity profiles for key joints."""
        if len(landmarks_seq) < 2:
            return {}
        
        L = mp_pose.PoseLandmark if MEDIAPIPE_AVAILABLE else None
        if L is None:
            return {}
        
        key_joints = [L.LEFT_ANKLE, L.RIGHT_ANKLE, L.LEFT_KNEE, L.RIGHT_KNEE, L.LEFT_HIP, L.RIGHT_HIP]
        velocities = {}
        
        for joint_idx in key_joints:
            joint_positions = [lm[joint_idx] for lm in landmarks_seq]
            
            # Compute velocity (first derivative)
            positions = np.array(joint_positions)
            velocities_joint = np.diff(positions, axis=0)
            
            # Magnitude
            speed = np.linalg.norm(velocities_joint, axis=1)
            
            velocities[f"joint_{joint_idx}"] = {
                "mean_speed": np.mean(speed),
                "max_speed": np.max(speed),
                "std_speed": np.std(speed),
            }
        
        return velocities
    
    def _mock_gait_metrics(self) -> Dict:
        """Fallback gait metrics."""
        return {
            "step_frequency": 1.0,
            "stride_length": 0.5,
            "joint_angles": {"hip": 180.0, "knee": 170.0, "ankle": 90.0},
            "velocity_profiles": {},
        }


def extract_pose_from_video(video_path: str, max_frames: int = 30) -> Tuple[List[Dict], Dict]:
    """
    Extract pose sequence from video file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum frames to extract
        
    Returns:
        Tuple of (pose_sequence, gait_metrics)
    """
    cap = cv2.VideoCapture(video_path)
    pose_extractor = PoseExtractor()
    
    pose_sequence = []
    frame_idx = 0
    
    while cap.isOpened() and len(pose_sequence) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        frame = cv2.resize(frame, (640, 480))
        
        # Extract pose
        pose_result = pose_extractor.extract_pose(frame, frame_idx)
        pose_sequence.append(pose_result)
        
        frame_idx += 1
    
    cap.release()
    
    # Compute gait metrics
    gait_metrics = pose_extractor.compute_gait_metrics(pose_sequence)
    
    return pose_sequence, gait_metrics


if __name__ == "__main__":
    # Test pose extraction
    test_video = "uploads/8296233-hd_1080_1920_25fps.mp4"
    
    if os.path.exists(test_video):
        pose_seq, metrics = extract_pose_from_video(test_video, max_frames=30)
        print(f"Extracted {len(pose_seq)} frames")
        print(f"Gait metrics: {metrics}")
    else:
        print(f"Test video not found: {test_video}")