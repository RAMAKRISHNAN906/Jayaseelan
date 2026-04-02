"""
Feature Engineering Module for Gait Analysis.

Computes joint angles, velocities, step frequency, stride length, and normalized feature vectors.
Includes data augmentation and noise reduction.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from app.utils.config import DATA_CONFIG, POSE_CONFIG

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError):
    MEDIAPIPE_AVAILABLE = False
    mp_pose = None


class KalmanFilter1D:
    """1D Kalman filter for noise reduction."""
    
    def __init__(self, alpha=0.3, process_noise=0.01, measurement_noise=0.1):
        self.alpha = alpha
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.x = 0.0
        self.P = 1.0
        self.initialized = False
    
    def update(self, measurement: float) -> float:
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return measurement
        
        P_pred = self.P + self.process_noise
        K = P_pred / (P_pred + self.measurement_noise)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * P_pred
        
        return self.x


class FeatureEngineer:
    """
    Feature engineering for gait analysis.
    
    Computes:
    - Joint angles (hip, knee, ankle, shoulder, elbow)
    - Velocities (first derivative of landmark positions)
    - Step frequency (temporal peak detection)
    - Stride length (spatial displacement)
    - Normalized feature vectors
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or DATA_CONFIG
        
        # Kalman filters for noise reduction
        self.kalman_filters = {}
        self._init_kalman_filters()
    
    def _init_kalman_filters(self):
        """Initialize Kalman filters for feature smoothing."""
        alpha = self.config.get("kalman_alpha", 0.3) if isinstance(self.config.get("kalman_alpha"), float) else 0.3
        for i in range(20):  # For various features
            self.kalman_filters[i] = KalmanFilter1D(alpha=alpha)
    
    def compute_joint_angles(self, landmarks: List[Tuple[float, float, float]]) -> Dict:
        """
        Compute joint angles from landmarks.
        
        Args:
            landmarks: List of (x, y, visibility) tuples
            
        Returns:
            Dict with joint angles in degrees
        """
        if not landmarks or len(landmarks) < 33:
            return self._mock_joint_angles()
        
        angles = {}
        
        # Hip angle (hip-knee-ankle)
        angles["left_hip"] = self._compute_angle(
            landmarks[12], landmarks[13], landmarks[14]  # LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        )
        angles["right_hip"] = self._compute_angle(
            landmarks[9], landmarks[10], landmarks[11]  # RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        )
        
        # Knee angle
        angles["left_knee"] = self._compute_angle(
            landmarks[13], landmarks[12], landmarks[14]
        )
        angles["right_knee"] = self._compute_angle(
            landmarks[10], landmarks[9], landmarks[11]
        )
        
        # Ankle angle
        angles["left_ankle"] = self._compute_angle(
            landmarks[14], landmarks[13], landmarks[12]
        )
        angles["right_ankle"] = self._compute_angle(
            landmarks[11], landmarks[10], landmarks[9]
        )
        
        # Shoulder angle (shoulder-elbow-wrist)
        angles["left_shoulder"] = self._compute_angle(
            landmarks[5], landmarks[6], landmarks[7]  # LEFT_SHOULDER, LEFT_ELBOW, LEFT_Wrist
        )
        angles["right_shoulder"] = self._compute_angle(
            landmarks[2], landmarks[3], landmarks[4]  # RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_Wrist
        )
        
        # Apply Kalman smoothing
        for key in angles:
            idx = hash(key) % 20
            angles[key] = self.kalman_filters[idx].update(angles[key])
        
        return angles
    
    def _compute_angle(self, p1: Tuple, p2: Tuple, p3: Tuple) -> float:
        """Compute angle between three points (p1-p2-p3)."""
        v1 = np.array([p1[0], p1[1]]) - np.array([p2[0], p2[1]])
        v2 = np.array([p3[0], p3[1]]) - np.array([p2[0], p2[1]])
        
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        
        angle = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0))
        return np.degrees(angle)
    
    def _mock_joint_angles(self) -> Dict:
        """Fallback joint angles."""
        return {
            "left_hip": 170.0, "right_hip": 170.0,
            "left_knee": 160.0, "right_knee": 160.0,
            "left_ankle": 90.0, "right_ankle": 90.0,
            "left_shoulder": 175.0, "right_shoulder": 175.0,
        }
    
    def compute_velocities(self, landmarks_seq: List[List[Tuple]]) -> Dict:
        """
        Compute velocity profiles from landmark sequence.
        
        Args:
            landmarks_seq: Sequence of landmark lists
            
        Returns:
            Dict with velocity metrics
        """
        if len(landmarks_seq) < 2:
            return self._mock_velocities()
        
        velocities = {}
        
        # Key landmark indices for velocity computation
        key_indices = [12, 13, 14, 9, 10, 11]  # Hip, knee, ankle
        
        for idx in key_indices:
            positions = np.array([lm[idx][:2] for lm in landmarks_seq])
            
            # Compute velocity (first derivative)
            if len(positions) > 1:
                diff = np.diff(positions, axis=0)
                speeds = np.linalg.norm(diff, axis=1)
                
                velocities[f"landmark_{idx}"] = {
                    "mean_speed": float(np.mean(speeds)),
                    "max_speed": float(np.max(speeds)),
                    "std_speed": float(np.std(speeds)),
                }
        
        return velocities
    
    def _mock_velocities(self) -> Dict:
        """Fallback velocities."""
        return {
            "landmark_12": {"mean_speed": 0.01, "max_speed": 0.05, "std_speed": 0.01},
            "landmark_13": {"mean_speed": 0.02, "max_speed": 0.08, "std_speed": 0.02},
        }
    
    def compute_step_frequency(self, ankle_y_seq: List[float], fps: float = 25.0) -> float:
        """
        Estimate step frequency from ankle y-trajectory.
        
        Args:
            ankle_y_seq: Sequence of ankle y-coordinates
            fps: Frame rate
            
        Returns:
            Step frequency (steps per second)
        """
        if len(ankle_y_seq) < 10:
            return 1.0
        
        ankle_y = np.array(ankle_y_seq)
        
        # Find peaks (local maxima)
        diff = np.diff(ankle_y)
        peaks = np.where((diff[:-1] > 0) & (diff[1:] < 0))[0] + 1
        
        if len(peaks) < 2:
            return 1.0
        
        # Average time between peaks
        peak_intervals = np.diff(peaks) / fps
        avg_interval = np.mean(peak_intervals)
        
        step_freq = 1.0 / avg_interval if avg_interval > 0 else 1.0
        
        # Apply Kalman smoothing
        step_freq = self.kalman_filters[0].update(step_freq)
        
        return step_freq
    
    def compute_stride_length(self, landmarks_seq: List[List[Tuple]]) -> float:
        """
        Estimate stride length from landmark displacements.
        
        Args:
            landmarks_seq: Sequence of landmark lists
            
        Returns:
            Normalized stride length
        """
        if len(landmarks_seq) < 2:
            return 0.5
        
        # Use ankle displacement
        left_ankles = np.array([lm[14][:2] for lm in landmarks_seq])  # LEFT_ANKLE
        right_ankles = np.array([lm[11][:2] for lm in landmarks_seq])  # RIGHT_ANKLE
        
        # Compute max displacement
        left_displacement = np.max(left_ankles[:, 0]) - np.min(left_ankles[:, 0])
        right_displacement = np.max(right_ankles[:, 0]) - np.min(right_ankles[:, 0])
        
        stride_length = (left_displacement + right_displacement) / 2
        
        # Apply Kalman smoothing
        stride_length = self.kalman_filters[1].update(stride_length)
        
        return stride_length
    
    def create_feature_vector(self, pose_sequence: List[Dict]) -> np.ndarray:
        """
        Create normalized feature vector from pose sequence.
        
        Args:
            pose_sequence: List of pose dicts
            
        Returns:
            Normalized feature vector (1D array)
        """
        if not pose_sequence:
            return np.zeros(50)
        
        features = []
        
        # Extract landmarks from sequence
        landmarks_seq = [p.get("smoothed_landmarks", p.get("landmarks", [])) for p in pose_sequence]
        
        # 1. Joint angles (8 features)
        angles = self.compute_joint_angles(landmarks_seq[0] if landmarks_seq else [])
        features.extend([angles.get(k, 180.0) / 180.0 for k in ["left_hip", "right_hip", "left_knee", "right_knee",
                                                                  "left_ankle", "right_ankle", "left_shoulder", "right_shoulder"]])
        
        # 2. Velocities (6 features)
        velocities = self.compute_velocities(landmarks_seq)
        for key in ["landmark_12", "landmark_13", "landmark_14"]:
            if key in velocities:
                features.append(velocities[key]["mean_speed"] * 10)
                features.append(velocities[key]["max_speed"] * 10)
            else:
                features.extend([0.0, 0.0])
        
        # 3. Step frequency (1 feature)
        if landmarks_seq:
            ankle_y_seq = [lm[14][1] for lm in landmarks_seq]  # LEFT_ANKLE y
            step_freq = self.compute_step_frequency(ankle_y_seq)
            features.append(step_freq / 2.0)  # Normalize (assume max 2 steps/sec)
        else:
            features.append(0.5)
        
        # 4. Stride length (1 feature)
        stride_length = self.compute_stride_length(landmarks_seq)
        features.append(stride_length)
        
        # 5. Spine alignment (1 feature)
        if landmarks_seq:
            spine_scores = []
            for lm in landmarks_seq:
                if len(lm) >= 33:
                    mid_shoulder = (np.array(lm[5][:2]) + np.array(lm[2][:2])) / 2
                    mid_hip = (np.array(lm[12][:2]) + np.array(lm[9][:2])) / 2
                    spine_vec = mid_hip - mid_shoulder
                    spine_lean = abs(spine_vec[0]) / (abs(spine_vec[1]) + 1e-6)
                    spine_score = max(0, 1 - spine_lean * 3)
                    spine_scores.append(spine_score)
            features.append(np.mean(spine_scores) if spine_scores else 0.5)
        else:
            features.append(0.5)
        
        # 6. Shoulder symmetry (1 feature)
        if landmarks_seq:
            shoulder_scores = []
            for lm in landmarks_seq:
                if len(lm) >= 33:
                    shoulder_diff = abs(lm[5][1] - lm[2][1])
                    shoulder_score = max(0, 1 - shoulder_diff * 10)
                    shoulder_scores.append(shoulder_score)
            features.append(np.mean( shoulder_scores) if shoulder_scores else 0.5)
        else:
            features.append(0.5)
        
        # 7. Hip symmetry (1 feature)
        if landmarks_seq:
            hip_scores = []
            for lm in landmarks_seq:
                if len(lm) >= 33:
                    hip_diff = abs(lm[12][1] - lm[9][1])
                    hip_score = max(0, 1 - hip_diff * 10)
                    hip_scores.append(hip_score)
            features.append(np.mean(hip_scores) if hip_scores else 0.5)
        else:
            features.append(0.5)
        
        # Normalize to [0, 1]
        feature_vector = np.array(features, dtype=np.float32)
        feature_vector = np.clip(feature_vector, 0, 1)
        
        return feature_vector
    
    def augment_sequence(self, pose_sequence: List[Dict], augmentations: List[str] = None) -> List[Dict]:
        """
        Apply data augmentation to pose sequence.
        
        Args:
            pose_sequence: Original pose sequence
            augmentations: List of augmentation types to apply
            
        Returns:
            Augmented pose sequence
        """
        if augmentations is None:
            augmentations = []
        
        augmented = pose_sequence.copy()
        
        for aug_type in augmentations:
            if aug_type == "temporal_jitter":
                augmented = self._apply_temporal_jitter(augmented)
            elif aug_type == "spatial_flip":
                augmented = self._apply_spatial_flip(augmented)
            elif aug_type == "noise":
                augmented = self._apply_noise(augmented)
        
        return augmented
    
    def _apply_temporal_jitter(self, pose_sequence: List[Dict]) -> List[Dict]:
        """Apply temporal jittering (frame timing variation)."""
        jitter_range = self.config.get("jitter_range", 0.1)
        
        if len(pose_sequence) < 3:
            return pose_sequence
        
        # Randomly skip or duplicate frames
        jittered = []
        for i, pose in enumerate(pose_sequence):
            if np.random.random() < jitter_range:
                # Skip this frame
                continue
            jittered.append(pose)
        
        # Ensure minimum length
        while len(jittered) < len(pose_sequence):
            jittered.append(pose_sequence[np.random.randint(len(pose_sequence))])
        
        return jittered[:len(pose_sequence)]
    
    def _apply_spatial_flip(self, pose_sequence: List[Dict]) -> List[Dict]:
        """Apply horizontal flipping."""
        flipped = []
        
        for pose in pose_sequence:
            landmarks = pose.get("landmarks", [])
            flipped_landmarks = []
            
            for i, lm in landmarks:
                # Flip x-coordinate
                flipped_lm = (1.0 - lm[0], lm[1], lm[2])
                flipped_landmarks.append(flipped_lm)
            
            flipped_pose = pose.copy()
            flipped_pose["landmarks"] = flipped_landmarks
            flipped.append(flipped_pose)
        
        return flipped
    
    def _apply_noise(self, pose_sequence: List[Dict]) -> List[Dict]:
        """Apply noise injection to landmarks."""
        noise_std = self.config.get("noise_std", 0.01)
        
        noisy = []
        for pose in pose_sequence:
            landmarks = pose.get("landmarks", [])
            noisy_landmarks = []
            
            for lm in landmarks:
                noise_x = np.random.normal(0, noise_std)
                noise_y = np.random.normal(0, noise_std)
                noisy_lm = (lm[0] + noise_x, lm[1] + noise_y, lm[2])
                noisy_landmarks.append(noisy_lm)
            
            noisy_pose = pose.copy()
            noisy_pose["landmarks"] = noisy_landmarks
            noisy.append(noisy_pose)
        
        return noisy


def normalize_frame(frame: np.ndarray, use_histogram_eq: bool = True, use_gamma: bool = True) -> np.ndarray:
    """
    Normalize frame with histogram equalization and gamma correction.
    
    Args:
        frame: Input BGR frame
        use_histogram_eq: Apply histogram equalization
        use_gamma: Apply gamma correction for low-light
        
    Returns:
        Normalized frame
    """
    if frame is None:
        return np.zeros((128, 128, 3), dtype=np.uint8)
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Histogram equalization
    if use_histogram_eq:
        gray = cv2.equalizeHist(gray)
    
    # Gamma correction for low-light
    if use_gamma:
        # Compute mean brightness
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 50:  # Low-light threshold
            gamma = 1.5
            inv_gamma = 1.0 / gamma
            gray = np.array([((i / 255.0) ** inv_gamma) * 255
                            for i in np.uint8(gray)]).astype("uint8")
    
    # Convert back to BGR
    normalized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    return normalized


if __name__ == "__main__":
    # Test feature engineering
    from app.utils.pose_extractor import extract_pose_from_video
    
    test_video = "uploads/8296233-hd_1080_1920_25fps.mp4"
    
    pose_seq, _ = extract_pose_from_video(test_video, max_frames=10)
    
    engineer = FeatureEngineer()
    feature_vector = engineer.create_feature_vector(pose_seq)
    
    print(f"Feature vector shape: {feature_vector.shape}")
    print(f"Feature vector: {feature_vector}")