"""Debug script to identify realtime_processor 0 FPS issue."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import time
from app.utils.realtime_processor import RealtimeProcessor

test_video = "uploads/8296233-hd_1080_1920_25fps.mp4"
print(f"Debug: Testing video {test_video}")
print(f"Video exists: {os.path.exists(test_video)}")

# Open video directly to verify
cap = cv2.VideoCapture(test_video)
print(f"Video opened: {cap.isOpened()}")
if cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"Video FPS: {fps}, Total frames: {total_frames}")
    cap.release()

# Test processor
print("\n=== Testing RealtimeProcessor ===")
processor = RealtimeProcessor(use_gpu=False)

# Manual frame processing test
cap = cv2.VideoCapture(test_video)
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (640, 480))
    result = processor.process_frame(frame, frame_count)
    
    frame_count += 1
    if frame_count >= 10:  # Test first 10 frames
        break

cap.release()
elapsed = time.time() - start_time

print(f"\nManual test results:")
print(f"  Frames processed: {frame_count}")
print(f"  Time elapsed: {elapsed:.2f}s")
print(f"  Actual FPS: {frame_count / elapsed if elapsed > 0 else 0:.1f}")

stats = processor.get_processing_stats()
print(f"\nProcessor stats:")
print(f"  Current FPS: {stats['current_fps']:.1f}")
print(f"  Frames processed: {stats['frames_processed']}")
print(f"  Frames skipped: {stats['frames_skipped']}")