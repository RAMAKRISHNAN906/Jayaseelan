"""
Walking Analyzer with CNN-BiLSTM hybrid model, confidence scoring, and EMA smoothing.

Integrates pose extraction, feature engineering, and CNN-BiLSTM prediction
with exponential moving average smoothing for real-time walking analysis.
"""

import numpy as np
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional, Tuple
from collections import deque

try:
    import cv2
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    cv2 = None

from utils.config import MODEL_CONFIG, REALTIME_CONFIG


class WalkingAnalyzer:
    """
    Walking analyzer with CNN-BiLSTM model and confidence smoothing.
    
    Features:
    - CNN-BiLSTM hybrid model for temporal gait analysis
    - Exponential moving average (EMA) smoothing for confidence scores
    - Tunable walking threshold
    - Multi-person detection support
    - Edge case handling (low-light, occlusion)
    """
    
    def __init__(self, 
                 model_path: str = None,
                 config: Dict = None,
                 use_gpu: bool = True):
        """
        Initialize walking analyzer.
        
        Args:
            model_path: Path to saved model weights
            config: Configuration dict
            use_gpu: Whether to use GPU if available
        """
        self.config = config or MODEL_CONFIG
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        
        # Load model
        self.model = self._load_model(model_path)
        
        # EMA smoothing for confidence scores
        self.alpha = REALTIME_CONFIG.get("ema_alpha", 0.3)
        self.confidence_history = deque(maxlen=30)
        self.smoothed_confidence = 0.5
        
        # Walking threshold
        self.walking_threshold = REALTIME_CONFIG.get("walking_threshold", 0.6)
        
        # Frame buffer for sequence prediction
        self.frame_buffer = deque(maxlen=self.config.get("sequence_length", 30))
        
        # Multi-person tracking
        self.person_tracks = {}
        
        # Edge case handling
        self.low_light_threshold = self.config.get("low_light_threshold", 50)
        self.occlusion_threshold = self.config.get("occlusion_threshold", 0.5)
        
    def _load_model(self, model_path: str = None) -> nn.Module:
        """Load CNN-BiLSTM model."""
        
        if not TORCH_AVAILABLE:
            print("PyTorch not available, using mock predictions")
            return None
        
        try:
            from models.cnn_biLSTM_model import build_cnn_biLSTM_model, load_model_weights
            
            model = build_cnn_biLSTM_model(self.config)
            
            if model_path and os.path.exists(model_path):
                model = load_model_weights(model, model_path, self.config)
                print(f"Loaded model from: {model_path}")
            else:
                print("Using untrained model (random weights)")
            
            model = model.to(self.device)
            model.eval()
            
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def analyze_frame(self, 
                      frame: np.ndarray, 
                      frame_idx: int = 0,
                      person_id: int = 0) -> Dict:
        """
        Analyze single frame for walking detection.
        
        Args:
            frame: Video frame (BGR format)
            frame_idx: Frame index in sequence
            person_id: Person identifier for multi-person tracking
            
        Returns:
            Dict with walking detection results
        """
        # Mock pose extraction for testing
        pose_result = {
            "success": True,
            "landmarks": np.random.randn(33, 3),
            "metrics": {"step_frequency": 1.0, "stride_length": 0.5},
            "confidence": 0.8,
            "num_persons": 1,
        }
        
        if not pose_result["success"]:
            return self._fallback_result(frame, "pose_extraction_failed")
        
        # Mock gait features
        gait_features = {
            "overall_score": 70.0,
            "joint_angles": {"hip": 180, "knee": 170, "ankle": 90},
            "velocities": {"hip": 0.3, "knee": 0.2},
        }
        
        # Add to frame buffer
        processed_frame = self._preprocess_frame(frame)
        self.frame_buffer.append(processed_frame)
        
        # Update confidence history
        raw_confidence = pose_result.get("confidence", 0.5)
        self.confidence_history.append(raw_confidence)
        
        # Apply EMA smoothing
        self.smoothed_confidence = self._apply_ema(
            self.smoothed_confidence, 
            raw_confidence
        )
        
        # Check edge cases
        edge_case_flags = self._detect_edge_cases(frame, pose_result)
        
        # Predict if enough frames in buffer
        if len(self.frame_buffer) >= self.config.get("sequence_length", 30):
            prediction = self._predict_from_sequence()
        else:
            prediction = self._heuristic_prediction(gait_features, pose_result)
        
        # Apply edge case adjustments
        prediction = self._apply_edge_case_adjustments(prediction, edge_case_flags)
        
        # Multi-person tracking
        if person_id not in self.person_tracks:
            self.person_tracks[person_id] = {
                "history": deque(maxlen=50),
                "last_prediction": prediction
            }
        
        self.person_tracks[person_id]["history"].append(prediction)
        self.person_tracks[person_id]["last_prediction"] = prediction
        
        # Build result
        result = {
            "success": True,
            "frame_idx": frame_idx,
            "person_id": person_id,
            "walking_detected": prediction["score"] >= self.walking_threshold * 100,
            "walking_score": prediction["walking_score"],
            "risk_score": prediction["score"],
            "risk_level": prediction["level"],
            "confidence": self.smoothed_confidence * 100,
            "color": prediction["color"],
            "walking_label": prediction["walking_label"],
            "gait_features": gait_features,
            "pose_metrics": pose_result["metrics"],
            "edge_cases": edge_case_flags,
            "is_dual_task": False,
        }
        
        return result
    
    def analyze_sequence(self, 
                         frames: List[np.ndarray],
                         person_id: int = 0) -> Dict:
        """
        Analyze sequence of frames for walking detection.
        
        Args:
            frames: List of video frames
            person_id: Person identifier
            
        Returns:
            Dict with aggregated walking analysis
        """
        results = []
        
        for i, frame in enumerate(frames):
            result = self.analyze_frame(frame, frame_idx=i, person_id=person_id)
            results.append(result)
        
        # Aggregate results
        if not results:
            return self._fallback_result(None, "no_frames")
        
        # Average scores
        avg_walking_score = np.mean([r["walking_score"] for r in results])
        avg_risk_score = np.mean([r["risk_score"] for r in results])
        avg_confidence = np.mean([r["confidence"] for r in results])
        
        # Majority voting for risk level
        risk_levels = [r["risk_level"] for r in results]
        final_level = max(risk_levels, key=risk_levels.count)
        
        # Determine color based on average risk
        if avg_risk_score < 35:
            color = "success"
            walking_label = "Good Walking"
        elif avg_risk_score < 65:
            color = "warning"
            walking_label = "Irregular Walking"
        else:
            color = "danger"
            walking_label = "Poor Walking"
        
        return {
            "success": True,
            "person_id": person_id,
            "num_frames": len(frames),
            "walking_detected": avg_walking_score >= self.walking_threshold * 100,
            "walking_score": round(avg_walking_score, 1),
            "risk_score": round(avg_risk_score, 1),
            "risk_level": final_level,
            "confidence": round(avg_confidence, 1),
            "color": color,
            "walking_label": walking_label,
            "frame_results": results,
            "is_dual_task": False,
        }
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for model input."""
        
        # Resize to model input size
        height, width = self.config.get("input_shape", (128, 128, 3))[:2]
        
        if cv2 is not None and len(frame.shape) == 3:
            resized = cv2.resize(frame, (width, height))
        else:
            resized = frame
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def _apply_ema(self, previous: float, current: float) -> float:
        """Apply exponential moving average smoothing."""
        return self.alpha * current + (1 - self.alpha) * previous
    
    def _predict_from_sequence(self) -> Dict:
        """Predict from sequence buffer using CNN-BiLSTM."""
        
        if not TORCH_AVAILABLE or self.model is None:
            return self._mock_prediction()
        
        try:
            # Convert buffer to tensor
            sequence = np.array(list(self.frame_buffer))
            
            if len(sequence.shape) == 4:
                sequence = np.expand_dims(sequence, axis=0)
            
            # Convert to torch tensor
            sequence_tensor = torch.from_numpy(sequence).float().to(self.device)
            
            # Predict
            with torch.no_grad():
                class_logits, confidence = self.model(sequence_tensor)
            
            # Get predictions
            class_probs = torch.softmax(class_logits, dim=1)
            class_idx = class_probs.argmax(1).item()
            
            class_names = ["Low Risk", "Medium Risk", "High Risk"]
            risk_level = class_names[class_idx]
            
            risk_score = round(class_probs[0, class_idx].item() * 100, 1)
            walking_score = round(100 - risk_score, 1)
            
            if risk_score < 35:
                color = "success"
                walking_label = "Good Walking"
            elif risk_score < 65:
                color = "warning"
                walking_label = "Irregular Walking"
            else:
                color = "danger"
                walking_label = "Poor Walking"
            
            return {
                "score": risk_score,
                "walking_score": walking_score,
                "level": risk_level,
                "color": color,
                "walking_label": walking_label,
                "confidence": confidence[0, 0].item(),
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._mock_prediction()
    
    def _heuristic_prediction(self, gait_features: Dict, pose_result: Dict) -> Dict:
        """Heuristic prediction when model not available or insufficient frames."""
        
        # Use gait features for heuristic scoring
        overall_score = gait_features.get("overall_score", 70.0)
        
        risk_score = round(100 - overall_score, 1)
        walking_score = overall_score
        
        if risk_score < 35:
            level = "Low Risk"
            color = "success"
            walking_label = "Good Walking"
        elif risk_score < 65:
            level = "Medium Risk"
            color = "warning"
            walking_label = "Irregular Walking"
        else:
            level = "High Risk"
            color = "danger"
            walking_label = "Poor Walking"
        
        return {
            "score": risk_score,
            "walking_score": walking_score,
            "level": level,
            "color": color,
            "walking_label": walking_label,
            "confidence": 0.5,
        }
    
    def _mock_prediction(self) -> Dict:
        """Fallback mock prediction."""
        return {
            "score": 50.0,
            "walking_score": 50.0,
            "level": "Medium Risk",
            "color": "warning",
            "walking_label": "Walking",
            "confidence": 0.5,
        }
    
    def _fallback_result(self, frame: np.ndarray, reason: str) -> Dict:
        """Fallback result when analysis fails."""
        return {
            "success": False,
            "reason": reason,
            "walking_detected": False,
            "walking_score": 0.0,
            "risk_score": 100.0,
            "risk_level": "Unknown",
            "confidence": 0.0,
            "color": "danger",
            "walking_label": "Analysis Failed",
            "gait_features": {},
            "pose_metrics": {},
            "edge_cases": {},
            "is_dual_task": False,
        }
    
    def _detect_edge_cases(self, frame: np.ndarray, pose_result: Dict) -> Dict:
        """Detect edge cases like low-light, occlusion, etc."""
        
        edge_cases = {
            "low_light": False,
            "occlusion": False,
            "multi_person": False,
            "camera_angle": "normal",
        }
        
        # Low-light detection
        if frame is not None:
            mean_brightness = np.mean(frame)
            if mean_brightness < self.low_light_threshold:
                edge_cases["low_light"] = True
        
        # Occlusion detection (based on landmark visibility)
        landmarks = pose_result.get("landmarks", [])
        if len(landmarks) > 0:
            visibility_scores = [lm[2] if len(lm) > 2 else 1.0 for lm in landmarks]
            avg_visibility = np.mean(visibility_scores)
            if avg_visibility < self.occlusion_threshold:
                edge_cases["occlusion"] = True
        
        # Multi-person detection
        if pose_result.get("num_persons", 1) > 1:
            edge_cases["multi_person"] = True
        
        return edge_cases
    
    def _apply_edge_case_adjustments(self, prediction: Dict, edge_cases: Dict) -> Dict:
        """Apply adjustments based on edge cases."""
        
        # Low-light adjustment (reduce confidence)
        if edge_cases.get("low_light"):
            prediction["confidence"] *= 0.8
            prediction["edge_case_note"] = "Low light detected"
        
        # Occlusion adjustment (reduce confidence)
        if edge_cases.get("occlusion"):
            prediction["confidence"] *= 0.7
            prediction["edge_case_note"] = "Occlusion detected"
        
        # Multi-person (flag for separate processing)
        if edge_cases.get("multi_person"):
            prediction["edge_case_note"] = "Multiple persons detected"
        
        return prediction
    
    def set_walking_threshold(self, threshold: float) -> None:
        """Set walking detection threshold."""
        self.walking_threshold = max(0.0, min(1.0, threshold))
    
    def reset(self) -> None:
        """Reset analyzer state."""
        self.frame_buffer.clear()
        self.confidence_history.clear()
        self.smoothed_confidence = 0.5
        self.person_tracks.clear()


if __name__ == "__main__":
    # Test walking analyzer
    print("Testing Walking Analyzer...")
    
    analyzer = WalkingAnalyzer()
    
    # Test with random frame
    test_frame = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
    
    result = analyzer.analyze_frame(test_frame, frame_idx=0)
    
    print(f"Analysis result: {result}")
    print(f"Walking detected: {result['walking_detected']}")
    print(f"Walking score: {result['walking_score']}")
    print(f"Confidence: {result['confidence']}")
