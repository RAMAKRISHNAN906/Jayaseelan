"""Test script for realtime_processor and visualizer modules."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.utils.realtime_processor import RealtimeProcessor
from app.utils.visualizer import Visualizer

# Test with sample video
test_video = "uploads/8296233-hd_1080_1920_25fps.mp4"

print(f"Testing with video: {test_video}")
print(f"Video exists: {os.path.exists(test_video)}")

# Test realtime processor
print("\n=== Testing RealtimeProcessor ===")
processor = RealtimeProcessor(use_gpu=False)
results = processor.process_video_stream(test_video, max_frames=30)

stats = processor.get_processing_stats()
print(f"\nProcessing Statistics:")
print(f"  FPS: {stats['current_fps']:.1f}")
print(f"  Frames processed: {stats['frames_processed']}")
print(f"  Frames skipped: {stats['frames_skipped']}")
print(f"  Average confidence: {stats['average_confidence']:.1f}%")

# Test visualizer
print("\n=== Testing Visualizer ===")
visualizer = Visualizer()

if results:
    # Create visualization for first frame
    test_frame = results[0]
    print(f"Sample result keys: {list(test_frame.keys())}")
    
    # Test drawing functions
    import numpy as np
    mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    landmarks = test_frame.get('landmarks', [])
    if landmarks:
        print(f"  Landmarks available: {len(landmarks)}")
        
        # Draw skeleton
        visualized = visualizer.draw_skeleton(mock_frame, landmarks, 0.8)
        print("  [OK] Skeleton drawn")
        
        # Draw bounding box
        visualized = visualizer.draw_bounding_box(visualized, landmarks)
        print("  [OK] Bounding box drawn")
    
    # Draw status badge
    walking_detected = test_frame.get('walking_detected', False)
    confidence = test_frame.get('confidence', 50.0)
    visualized = visualizer.draw_walking_status_badge(mock_frame, walking_detected, confidence)
    print("  [OK] Status badge drawn")
    
    # Draw metrics
    gait_metrics = test_frame.get('gait_features', {})
    if gait_metrics:
        visualized = visualizer.draw_gait_metrics(mock_frame, gait_metrics)
        print("  [OK] Gait metrics drawn")
    
    # Draw FPS
    fps = test_frame.get('fps', 20.0)
    visualized = visualizer.draw_fps_counter(mock_frame, fps)
    print("  [OK] FPS counter drawn")
    
    # Save test output
    output_path = "app/static/processed/test_visualized.jpg"
    cv2 = None
    try:
        import cv2
        cv2.imwrite(output_path, visualized)
        print(f"  [OK] Test output saved to: {output_path}")
    except Exception as e:
        print(f"  [ERROR] Could not save image: {e}")

print("\n[OK] All module tests completed!")