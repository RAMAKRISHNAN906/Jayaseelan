"""
Visualization Module for Walking Detection System.

Renders skeleton overlay with color-coded joints, walking status badge, bounding boxes,
gait metrics display, and FPS counter with modern minimalist design.
"""

import cv2
import numpy as np
import os
import sys
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.config import VISUALIZER_CONFIG, POSE_CONFIG


class Visualizer:
    """
    Modern minimalist visualizer for walking detection results.
    
    Features:
    - Skeleton overlay with color-coded joints (green=high confidence, red=low)
    - Walking status badge with confidence percentage
    - Bounding box around detected person(s)
    - Gait metrics display (step frequency, stride length, symmetry score)
    - Real-time FPS counter
    - Clean typography and smooth transitions
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration dict (uses VISUALIZER_CONFIG defaults)
        """
        self.config = config or VISUALIZER_CONFIG
        
        # Colors (BGR format)
        self.high_conf_color = self.config.get("high_confidence_color", (0, 255, 0))
        self.medium_conf_color = self.config.get("medium_confidence_color", (255, 255, 0))
        self.low_conf_color = self.config.get("low_confidence_color", (255, 0, 0))
        
        # Drawing settings
        self.line_thickness = self.config.get("line_thickness", 2)
        self.circle_radius = self.config.get("circle_radius", 3)
        self.font_scale = self.config.get("font_scale", 0.6)
        self.font_thickness = self.config.get("font_thickness", 2)
        
        # MediaPipe pose connections
        self.pose_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Head
            (0, 5), (0, 9),  # Neck to shoulders
            (5, 6), (6, 7), (7, 8),  # Left arm
            (9, 10), (10, 11), (11, 12),  # Right arm
            (5, 11), (9, 12),  # Shoulders to hips
            (11, 12),  # Hips
            (12, 13), (13, 14), (14, 15),  # Left leg
            (12, 16), (16, 17), (17, 18),  # Right leg
        ]
        
        # Badge settings
        self.badge_position = self.config.get("badge_position", "top_left")
        self.badge_size = self.config.get("badge_size", (150, 50))
        
        # Metrics display
        self.metrics_position = self.config.get("metrics_position", "bottom_left")
        self.metrics_font_scale = self.config.get("metrics_font_scale", 0.5)
    
    def draw_skeleton(self, 
                      frame: np.ndarray, 
                      landmarks: List[Tuple[float, float, float]],
                      confidence: float = 0.5) -> np.ndarray:
        """
        Draw skeleton overlay on frame.
        
        Args:
            frame: BGR image frame
            landmarks: List of (x, y, visibility) tuples (33 landmarks)
            confidence: Overall pose confidence
            
        Returns:
            Frame with skeleton overlay
        """
        if not landmarks or len(landmarks) < 33:
            return frame
        
        # Determine color based on confidence
        if confidence > 0.7:
            color = self.high_conf_color
        elif confidence > 0.4:
            color = self.medium_conf_color
        else:
            color = self.low_conf_color
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Draw landmarks (circles)
        for i, lm in enumerate(landmarks):
            x, y, visibility = lm
            
            # Convert to pixel coordinates
            px = int(x * width)
            py = int(y * height)
            
            # Only draw visible landmarks
            if visibility > 0.3:
                # Color intensity based on visibility
                lm_color = tuple(int(c * visibility) for c in color)
                cv2.circle(frame, (px, py), self.circle_radius, lm_color, -1)
        
        # Draw connections (lines)
        for connection in self.pose_connections:
            idx1, idx2 = connection
            
            if idx1 < len(landmarks) and idx2 < len(landmarks):
                lm1 = landmarks[idx1]
                lm2 = landmarks[idx2]
                
                # Check visibility
                if lm1[2] > 0.3 and lm2[2] > 0.3:
                    x1, y1 = int(lm1[0] * width), int(lm1[1] * height)
                    x2, y2 = int(lm2[0] * width), int(lm2[1] * height)
                    
                    cv2.line(frame, (x1, y1), (x2, y2), color, self.line_thickness)
        
        return frame
    
    def draw_walking_status_badge(self, 
                                   frame: np.ndarray,
                                   walking_detected: bool,
                                   confidence: float,
                                   walking_score: float = 0.0) -> np.ndarray:
        """
        Draw walking status badge with confidence percentage.
        
        Args:
            frame: BGR image frame
            walking_detected: Boolean walking detection result
            confidence: Confidence score (0-100)
            walking_score: Walking quality score (0-100)
            
        Returns:
            Frame with status badge
        """
        # Badge parameters
        badge_width, badge_height = self.badge_size
        padding = 10
        
        # Position
        if self.badge_position == "top_left":
            x, y = padding, padding
        elif self.badge_position == "top_right":
            x, y = frame.shape[1] - badge_width - padding, padding
        else:
            x, y = padding, padding
        
        # Background color based on walking status
        if walking_detected:
            bg_color = (0, 150, 0)  # Green
            status_text = "Walking"
        else:
            bg_color = (150, 0, 0)  # Red
            status_text = "Not Walking"
        
        # Draw badge background
        cv2.rectangle(frame, (x, y), (x + badge_width, y + badge_height), bg_color, -1)
        
        # Add border
        cv2.rectangle(frame, (x, y), (x + badge_width, y + badge_height), (255, 255, 255), 2)
        
        # Status text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(status_text, font, self.font_scale, self.font_thickness)[0][0]
        text_x = x + (badge_width - text_size) // 2
        text_y = y + badge_height // 2 + 5
        
        cv2.putText(frame, status_text, (text_x, text_y), 
                    font, self.font_scale, (255, 255, 255), self.font_thickness)
        
        # Confidence percentage (smaller text below)
        conf_text = f"{confidence:.1f}%"
        conf_font_scale = self.font_scale * 0.8
        conf_size = cv2.getTextSize(conf_text, font, conf_font_scale, 1)[0][0]
        conf_x = x + (badge_width - conf_size) // 2
        conf_y = y + badge_height - 8
        
        cv2.putText(frame, conf_text, (conf_x, conf_y), 
                    font, conf_font_scale, (255, 255, 255), 1)
        
        return frame
    
    def draw_bounding_box(self, 
                          frame: np.ndarray,
                          landmarks: List[Tuple[float, float, float]],
                          color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw bounding box around detected person.
        
        Args:
            frame: BGR image frame
            landmarks: List of (x, y, visibility) tuples
            color: Bounding box color
            
        Returns:
            Frame with bounding box
        """
        if not landmarks or len(landmarks) < 33:
            return frame
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Find min/max x,y from visible landmarks
        visible_landmarks = [lm for lm in landmarks if lm[2] > 0.3]
        
        if not visible_landmarks:
            return frame
        
        x_coords = [lm[0] * width for lm in visible_landmarks]
        y_coords = [lm[1] * height for lm in visible_landmarks]
        
        min_x = int(min(x_coords))
        max_x = int(max(x_coords))
        min_y = int(min(y_coords))
        max_y = int(max(y_coords))
        
        # Draw bounding box
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, 3)
        
        return frame
    
    def draw_gait_metrics(self, 
                          frame: np.ndarray,
                          metrics: Dict) -> np.ndarray:
        """
        Draw gait metrics display.
        
        Args:
            frame: BGR image frame
            metrics: Dict with gait metrics (step_frequency, stride_length, symmetry_score)
            
        Returns:
            Frame with metrics display
        """
        # Position
        if self.metrics_position == "bottom_left":
            x, y = 10, frame.shape[0] - 80
        else:
            x, y = 10, frame.shape[0] - 80
        
        # Metrics text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.metrics_font_scale
        
        # Background rectangle
        bg_height = 70
        bg_width = 220
        cv2.rectangle(frame, (x - 5, y - 5), (x + bg_width + 5, y + bg_height + 5), 
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (x - 5, y - 5), (x + bg_width + 5, y + bg_height + 5), 
                      (255, 255, 255), 1)
        
        # Step frequency
        step_freq = metrics.get("step_frequency", 0.0)
        text1 = f"Step Frequency: {step_freq:.2f} steps/s"
        cv2.putText(frame, text1, (x, y + 15), font, font_scale, (255, 255, 255), 1)
        
        # Stride length
        stride_length = metrics.get("stride_length", 0.0)
        text2 = f"Stride Length: {stride_length:.2f} m"
        cv2.putText(frame, text2, (x, y + 35), font, font_scale, (255, 255, 255), 1)
        
        # Symmetry score
        symmetry = metrics.get("symmetry_score", 0.0)
        text3 = f"Symmetry: {symmetry:.1f}%"
        cv2.putText(frame, text3, (x, y + 55), font, font_scale, (255, 255, 255), 1)
        
        return frame
    
    def draw_fps_counter(self, 
                         frame: np.ndarray,
                         fps: float) -> np.ndarray:
        """
        Draw real-time FPS counter.
        
        Args:
            frame: BGR image frame
            fps: Current FPS value
            
        Returns:
            Frame with FPS counter
        """
        # Position (top right)
        x, y = frame.shape[1] - 100, 30
        
        # FPS color based on performance
        if fps >= 20:
            color = (0, 255, 0)  # Green - good
        elif fps >= 15:
            color = (255, 255, 0)  # Yellow - acceptable
        else:
            color = (255, 0, 0)  # Red - slow
        
        # Background
        cv2.rectangle(frame, (x - 5, y - 15), (x + 95, y + 20), (0, 0, 0), -1)
        
        # FPS text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"FPS: {fps:.1f}"
        cv2.putText(frame, text, (x, y), font, self.font_scale, color, self.font_thickness)
        
        return frame
    
    def draw_full_overlay(self, 
                          frame: np.ndarray,
                          result: Dict) -> np.ndarray:
        """
        Draw complete visualization overlay.
        
        Args:
            frame: BGR image frame
            result: Dict with all analysis results
            
        Returns:
            Frame with all visual elements
        """
        # Extract data from result
        landmarks = result.get("landmarks", [])
        confidence = result.get("confidence", 50.0)
        walking_detected = result.get("walking_detected", False)
        walking_score = result.get("walking_score", 0.0)
        gait_metrics = result.get("gait_features", {})
        fps = result.get("fps", 0.0)
        
        # 1. Draw skeleton
        if landmarks:
            frame = self.draw_skeleton(frame, landmarks, confidence / 100.0)
        
        # 2. Draw walking status badge
        frame = self.draw_walking_status_badge(
            frame, 
            walking_detected, 
            confidence,
            walking_score
        )
        
        # 3. Draw bounding box
        if landmarks:
            box_color = self.high_conf_color if walking_detected else self.low_conf_color
            frame = self.draw_bounding_box(frame, landmarks, box_color)
        
        # 4. Draw gait metrics
        if gait_metrics:
            frame = self.draw_gait_metrics(frame, gait_metrics)
        
        # 5. Draw FPS counter
        frame = self.draw_fps_counter(frame, fps)
        
        return frame
    
    def create_summary_frame(self, 
                             frames: List[np.ndarray],
                             results: List[Dict]) -> np.ndarray:
        """
        Create summary frame with aggregated results.
        
        Args:
            frames: List of frames
            results: List of analysis results
            
        Returns:
            Summary frame with aggregated visualization
        """
        if not frames or not results:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Use last frame
        frame = frames[-1].copy()
        
        # Aggregate results
        avg_confidence = np.mean([r.get("confidence", 0) for r in results])
        avg_walking_score = np.mean([r.get("walking_score", 0) for r in results])
        walking_detected = avg_walking_score > 60.0
        
        # Create aggregated result
        agg_result = {
            "landmarks": results[-1].get("landmarks", []),
            "confidence": avg_confidence,
            "walking_detected": walking_detected,
            "walking_score": avg_walking_score,
            "gait_features": results[-1].get("gait_features", {}),
            "fps": results[-1].get("fps", 0.0),
        }
        
        return self.draw_full_overlay(frame, agg_result)


def visualize_video_results(video_path: str, 
                            results: List[Dict],
                            output_path: str = None) -> str:
    """
    Create visualization video from results.
    
    Args:
        video_path: Path to original video
        results: List of analysis results
        output_path: Path for output video
        
    Returns:
        Path to output video
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None
    
    visualizer = Visualizer()
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Output path
    if output_path is None:
        output_path = video_path.replace(".mp4", "_visualized.mp4")
    
    # Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    
    while cap.isOpened() and frame_idx < len(results):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize for consistent processing
        frame = cv2.resize(frame, (640, 480))
        
        # Get result for this frame
        if frame_idx < len(results):
            result = results[frame_idx]
            
            # Draw overlay
            visualized = visualizer.draw_full_overlay(frame, result)
            
            out.write(visualized)
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    print(f"Visualization saved to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    # Test visualizer
    print("Testing Visualizer...")
    
    visualizer = Visualizer()
    
    # Test with mock frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Mock landmarks
    mock_landmarks = [(0.5 + 0.1 * np.sin(i), 0.5 + 0.1 * np.cos(i), 0.8) for i in range(33)]
    
    # Mock result
    mock_result = {
        "landmarks": mock_landmarks,
        "confidence": 85.0,
        "walking_detected": True,
        "walking_score": 75.0,
        "gait_features": {
            "step_frequency": 1.2,
            "stride_length": 0.65,
            "symmetry_score": 82.0,
        },
        "fps": 25.0,
    }
    
    # Draw full overlay
    visualized = visualizer.draw_full_overlay(test_frame, mock_result)
    
    print(f"Visualized frame shape: {visualized.shape}")
    print("✓ Visualizer test completed")
    
    # Save test frame
    test_output = "app/static/processed/test_visualized.jpg"
    cv2.imwrite(test_output, visualized)
    print(f"Test output saved to: {test_output}")
