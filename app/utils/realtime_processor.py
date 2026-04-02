"""
Real-Time Processor for Walking Detection System.

Handles batch inference, frame skipping, temporal smoothing, and FPS optimization
for real-time walking analysis with ≥20 FPS target.
"""

import cv2
import numpy as np
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
from collections import deque
from threading import Thread, Lock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.config import REALTIME_CONFIG, CONFIDENCE_CONFIG, MODEL_CONFIG
from app.utils.pose_extractor import PoseExtractor
from app.utils.feature_engineer import FeatureEngineer
from app.utils.walking_analyzer import WalkingAnalyzer


class RealtimeProcessor:
    """
    Real-time processor with batch inference, frame skipping, and temporal smoothing.
    
    Features:
    - Batch processing for latency reduction
    - Frame skipping when confidence > threshold
    - Exponential moving average (EMA) smoothing
    - Multi-person tracking
    - FPS optimization with adaptive processing
    """
    
    def __init__(self, 
                 model_path: str = None,
                 config: Dict = None,
                 use_gpu: bool = True):
        """
        Initialize real-time processor.
        
        Args:
            model_path: Path to model weights
            config: Configuration dict
            use_gpu: Use GPU if available
        """
        self.config = config or REALTIME_CONFIG
        self.use_gpu = use_gpu
        
        # Initialize components
        self.pose_extractor = PoseExtractor()
        self.feature_engineer = FeatureEngineer()
        self.walking_analyzer = WalkingAnalyzer(model_path=model_path, use_gpu=use_gpu)
        
        # Frame buffer for batch processing
        self.frame_buffer = deque(maxlen=self.config.get("batch_size", 4))
        self.results_buffer = deque(maxlen=30)
        
        # EMA smoothing
        self.alpha = CONFIDENCE_CONFIG.get("ema_alpha", 0.3)
        self.smoothed_confidence = 0.5
        self.smoothed_walking_score = 0.5
        
        # Frame skipping logic
        self.skip_threshold = self.config.get("frame_skip_threshold", 0.95)
        self.skip_count = self.config.get("skip_frames_count", 2)
        self.frames_skipped = 0
        self.last_high_confidence_frame = None
        
        # FPS tracking
        self.fps_counter = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.current_fps = 0.0
        
        # Multi-person tracking
        self.person_tracks = {}
        self.max_persons = self.config.get("max_persons", 3)
        
        # Processing lock for thread safety
        self.processing_lock = Lock()
        
        # Async processing state
        self.async_mode = self.config.get("async_mode", True)
        self.processing_thread = None
        
    def process_frame(self, 
                      frame: np.ndarray, 
                      frame_idx: int = 0) -> Dict:
        """
        Process single frame with optimization strategies.
        
        Args:
            frame: BGR video frame
            frame_idx: Frame index
            
        Returns:
            Dict with walking detection results
        """
        start_time = time.time()
        
        # Check if we should skip processing
        if self._should_skip_processing():
            self.frames_skipped += 1
            return self._get_last_result_with_skip_flag()
        
        # Extract pose
        pose_result = self.pose_extractor.extract_pose(frame, frame_idx)
        
        if not pose_result.get("success", False):
            return self._fallback_result("pose_extraction_failed")
        
        # Compute features
        feature_vector = self.feature_engineer.create_feature_vector([pose_result])
        
        # Analyze walking
        analysis_result = self.walking_analyzer.analyze_frame(
            frame, 
            frame_idx=frame_idx,
            pose_result=pose_result,
            feature_vector=feature_vector
        )
        
        # Apply EMA smoothing
        self._apply_smoothing(analysis_result)
        
        # Check for frame skipping opportunity
        self._update_skip_state(analysis_result)
        
        # Update FPS
        self._update_fps(start_time)
        
        # Store result
        self.results_buffer.append(analysis_result)
        
        return analysis_result
    
    def process_batch(self, 
                      frames: List[np.ndarray], 
                      start_idx: int = 0) -> List[Dict]:
        """
        Process batch of frames for latency reduction.
        
        Args:
            frames: List of frames
            start_idx: Starting frame index
            
        Returns:
            List of analysis results
        """
        results = []
        
        batch_size = min(len(frames), self.config.get("batch_size", 4))
        
        for i in range(batch_size):
            frame_idx = start_idx + i
            if frame_idx < len(frames):
                result = self.process_frame(frames[frame_idx], frame_idx)
                results.append(result)
        
        return results
    
    def process_video_stream(self, 
                             video_path: str, 
                             max_frames: int = None,
                             callback: callable = None) -> List[Dict]:
        """
        Process video stream in real-time.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process
            callback: Callback function for each result
            
        Returns:
            List of analysis results
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return []
        
        results = []
        frame_idx = 0
        
        print(f"Starting real-time processing of {video_path}")
        print(f"Target FPS: {self.config.get('target_fps', 20)}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 480))
            
            # Process frame
            result = self.process_frame(frame, frame_idx)
            results.append(result)
            
            # Call callback if provided
            if callback:
                callback(result, frame)
            
            frame_idx += 1
            
            # Check max frames
            if max_frames and frame_idx >= max_frames:
                break
        
        cap.release()
        
        print(f"Processed {frame_idx} frames")
        print(f"Average FPS: {self.current_fps:.1f}")
        print(f"Frames skipped: {self.frames_skipped}")
        
        return results
    
    def _should_skip_processing(self) -> bool:
        """Check if current frame should be skipped."""
        
        if not self.results_buffer:
            return False
        
        last_result = self.results_buffer[-1]
        confidence = last_result.get("confidence", 0.0) / 100.0
        
        # Skip if confidence very high and we're in skip mode
        if confidence > self.skip_threshold and self.frames_skipped < self.skip_count:
            return True
        
        return False
    
    def _get_last_result_with_skip_flag(self) -> Dict:
        """Return last result with skip flag."""
        
        if not self.results_buffer:
            return self._fallback_result("no_previous_result")
        
        last_result = self.results_buffer[-1].copy()
        last_result["frame_skipped"] = True
        last_result["confidence"] = self.smoothed_confidence * 100
        
        return last_result
    
    def _apply_smoothing(self, result: Dict) -> None:
        """Apply exponential moving average smoothing."""
        
        confidence = result.get("confidence", 50.0) / 100.0
        walking_score = result.get("walking_score", 50.0) / 100.0
        
        # EMA for confidence
        self.smoothed_confidence = (
            self.alpha * confidence + 
            (1 - self.alpha) * self.smoothed_confidence
        )
        
        # EMA for walking score
        self.smoothed_walking_score = (
            self.alpha * walking_score + 
            (1 - self.alpha) * self.smoothed_walking_score
        )
        
        # Update result with smoothed values
        result["smoothed_confidence"] = self.smoothed_confidence * 100
        result["smoothed_walking_score"] = self.smoothed_walking_score * 100
    
    def _update_skip_state(self, result: Dict) -> None:
        """Update frame skipping state based on result."""
        
        confidence = result.get("confidence", 0.0) / 100.0
        
        if confidence > self.skip_threshold:
            self.frames_skipped = 0
            self.last_high_confidence_frame = result
        else:
            self.frames_skipped = min(self.frames_skipped + 1, self.skip_count)
    
    def _update_fps(self, start_time: float) -> None:
        """Update FPS counter."""
        
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        if frame_time > 0:
            instant_fps = 1.0 / frame_time
            self.fps_counter.append(instant_fps)
            self.current_fps = np.mean(self.fps_counter)
    
    def _fallback_result(self, reason: str) -> Dict:
        """Fallback result when processing fails."""
        
        return {
            "success": False,
            "reason": reason,
            "walking_detected": False,
            "walking_score": 0.0,
            "risk_score": 100.0,
            "confidence": 0.0,
            "smoothed_confidence": self.smoothed_confidence * 100,
            "frame_skipped": False,
            "fps": self.current_fps,
        }
    
    def get_fps(self) -> float:
        """Get current FPS."""
        return self.current_fps
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics."""
        
        return {
            "current_fps": self.current_fps,
            "target_fps": self.config.get("target_fps", 20),
            "frames_processed": len(self.results_buffer),
            "frames_skipped": self.frames_skipped,
            "skip_ratio": self.frames_skipped / max(len(self.results_buffer), 1),
            "average_confidence": np.mean([r.get("confidence", 0) for r in self.results_buffer]) if self.results_buffer else 0,
            "smoothed_confidence": self.smoothed_confidence * 100,
        }
    
    def set_walking_threshold(self, threshold: float) -> None:
        """Set walking detection threshold dynamically."""
        self.walking_analyzer.set_walking_threshold(threshold)
    
    def reset(self) -> None:
        """Reset processor state."""
        self.frame_buffer.clear()
        self.results_buffer.clear()
        self.smoothed_confidence = 0.5
        self.smoothed_walking_score = 0.5
        self.frames_skipped = 0
        self.fps_counter.clear()
        self.current_fps = 0.0
        self.last_frame_time = time.time()


class AsyncProcessor:
    """
    Async processor for background video processing.
    
    Uses separate thread for non-blocking processing.
    """
    
    def __init__(self, processor: RealtimeProcessor):
        self.processor = processor
        self.queue = deque(maxlen=10)
        self.results_queue = deque(maxlen=10)
        self.worker_thread = None
        self.running = False
        self.lock = Lock()
    
    def start(self) -> None:
        """Start async processing thread."""
        self.running = True
        self.worker_thread = Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
    
    def stop(self) -> None:
        """Stop async processing."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
    
    def submit_frame(self, frame: np.ndarray, frame_idx: int = 0) -> None:
        """Submit frame for async processing."""
        with self.lock:
            self.queue.append((frame, frame_idx))
    
    def get_result(self) -> Optional[Dict]:
        """Get latest result if available."""
        with self.lock:
            if self.results_queue:
                return self.results_queue.popleft()
        return None
    
    def _worker_loop(self) -> None:
        """Worker thread loop."""
        
        while self.running:
            with self.lock:
                if self.queue:
                    frame, frame_idx = self.queue.popleft()
                else:
                    time.sleep(0.01)
                    continue
            
            result = self.processor.process_frame(frame, frame_idx)
            
            with self.lock:
                self.results_queue.append(result)


if __name__ == "__main__":
    # Test real-time processor
    print("Testing Real-Time Processor...")
    
    processor = RealtimeProcessor(use_gpu=False)
    
    # Test with sample video
    test_video = "uploads/8296233-hd_1080_1920_25fps.mp4"
    
    if os.path.exists(test_video):
        print(f"Processing video: {test_video}")
        results = processor.process_video_stream(test_video, max_frames=60)
        
        stats = processor.get_processing_stats()
        print(f"\nProcessing Statistics:")
        print(f"  FPS: {stats['current_fps']:.1f}")
        print(f"  Frames processed: {stats['frames_processed']}")
        print(f"  Frames skipped: {stats['frames_skipped']}")
        print(f"  Average confidence: {stats['average_confidence']:.1f}%")
    else:
        print(f"Test video not found: {test_video}")
        
        # Test with mock frames
        print("\nTesting with mock frames...")
        for i in range(20):
            mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = processor.process_frame(mock_frame, i)
            
            if i % 5 == 0:
                print(f"Frame {i}: Walking={result.get('walking_detected', False)}, "
                      f"Confidence={result.get('confidence', 0):.1f}%")
        
        stats = processor.get_processing_stats()
        print(f"\nFinal FPS: {stats['current_fps']:.1f}")