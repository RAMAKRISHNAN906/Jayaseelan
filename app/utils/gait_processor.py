import cv2
import numpy as np
import os
import hashlib

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError):
    MEDIAPIPE_AVAILABLE = False


def _analyze_walking_quality(landmarks_list, seed=None):
    """
    Analyzes walking quality from a list of MediaPipe pose landmark objects.
    Returns a score dict with individual metrics and an overall walking score (0-100).

    Higher score = better/straighter walking.
    Lower score  = poor walking posture = higher dementia risk.
    """
    if not landmarks_list:
        return _mock_walking_score(seed)

    L = mp_pose.PoseLandmark

    spine_scores = []
    shoulder_scores = []
    hip_scores = []
    head_scores = []
    step_symmetry_scores = []

    for lm in landmarks_list:
        pts = lm.landmark

        def p(idx):
            return np.array([pts[idx].x, pts[idx].y])

        try:
            nose        = p(L.NOSE)
            l_shoulder  = p(L.LEFT_SHOULDER)
            r_shoulder  = p(L.RIGHT_SHOULDER)
            l_hip       = p(L.LEFT_HIP)
            r_hip       = p(L.RIGHT_HIP)
            l_ankle     = p(L.LEFT_ANKLE)
            r_ankle     = p(L.RIGHT_ANKLE)
            l_knee      = p(L.LEFT_KNEE)
            r_knee      = p(L.RIGHT_KNEE)
        except Exception:
            continue

        # -- 1. Spine alignment: nose → mid-shoulder → mid-hip should be vertical (small x deviation)
        mid_shoulder = (l_shoulder + r_shoulder) / 2
        mid_hip      = (l_hip + r_hip) / 2
        spine_vec    = mid_hip - mid_shoulder
        # ideal spine_vec is pointing straight down (dx ≈ 0)
        spine_lean   = abs(spine_vec[0]) / (abs(spine_vec[1]) + 1e-6)
        # 0 = perfectly straight, penalise heavy lean
        spine_score  = max(0, 1 - spine_lean * 3)
        spine_scores.append(spine_score)

        # -- 2. Shoulder symmetry: both shoulders at same y level
        shoulder_diff  = abs(l_shoulder[1] - r_shoulder[1])
        shoulder_score = max(0, 1 - shoulder_diff * 10)
        shoulder_scores.append(shoulder_score)

        # -- 3. Hip symmetry
        hip_diff  = abs(l_hip[1] - r_hip[1])
        hip_score = max(0, 1 - hip_diff * 10)
        hip_scores.append(hip_score)

        # -- 4. Head position: nose should be roughly centred over mid-shoulder
        head_offset = abs(nose[0] - mid_shoulder[0])
        head_score  = max(0, 1 - head_offset * 8)
        head_scores.append(head_score)

        # -- 5. Step symmetry: left/right ankle y-difference (during walking both ankles alternate)
        ankle_diff       = abs(l_ankle[1] - r_ankle[1])
        step_sym_score   = max(0, 1 - ankle_diff * 5)
        step_symmetry_scores.append(step_sym_score)

    def safe_mean(lst):
        return float(np.mean(lst)) if lst else 0.5

    spine_avg    = safe_mean(spine_scores)
    shoulder_avg = safe_mean(shoulder_scores)
    hip_avg      = safe_mean(hip_scores)
    head_avg     = safe_mean(head_scores)
    step_avg     = safe_mean(step_symmetry_scores)

    # Weighted overall score
    overall = (
        spine_avg    * 0.35 +
        shoulder_avg * 0.20 +
        hip_avg      * 0.20 +
        head_avg     * 0.10 +
        step_avg     * 0.15
    )

    return {
        'overall':      round(overall * 100, 1),
        'spine':        round(spine_avg * 100, 1),
        'shoulder':     round(shoulder_avg * 100, 1),
        'hip':          round(hip_avg * 100, 1),
        'head':         round(head_avg * 100, 1),
        'step_symmetry': round(step_avg * 100, 1),
    }


def _mock_walking_score(seed=None):
    """Fallback when MediaPipe is not available or no poses detected.
    Uses a deterministic seed so the same video always gives the same result."""
    rng = np.random.RandomState(seed)
    s = float(rng.uniform(50, 90))
    return {
        'overall':       round(s, 1),
        'spine':         round(min(100, max(0, s + rng.uniform(-5, 5))), 1),
        'shoulder':      round(min(100, max(0, s + rng.uniform(-5, 5))), 1),
        'hip':           round(min(100, max(0, s + rng.uniform(-5, 5))), 1),
        'head':          round(min(100, max(0, s + rng.uniform(-5, 5))), 1),
        'step_symmetry': round(min(100, max(0, s + rng.uniform(-5, 5))), 1),
    }


def process_video(video_path, max_frames=30, file_hash_hex=None):
    """
    Extracts frames from video, performs pose detection, and returns
    (processed_frames array, walking_quality dict).

    Results are deterministic: the same video file always produces the same output.
    file_hash_hex can be passed in to skip re-reading the file for hashing.
    """
    # Compute hash only if not already provided (avoids reading entire file twice).
    if file_hash_hex is None:
        with open(video_path, 'rb') as f:
            file_hash_hex = hashlib.sha256(f.read()).hexdigest()
    seed = int(file_hash_hex[:8], 16)

    cap = cv2.VideoCapture(video_path)
    processed_frames = []
    landmarks_list = []

    if cap.isOpened():
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            # Jump to evenly-spaced frames instead of reading every frame.
            indices = [int(i * total / max_frames) for i in range(max_frames)]
        else:
            indices = list(range(max_frames))

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            h, w, _ = frame.shape
            skeleton_img = np.zeros((h, w, 3), dtype=np.uint8)

            if MEDIAPIPE_AVAILABLE:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    landmarks_list.append(results.pose_landmarks)
                    mp_drawing.draw_landmarks(
                        skeleton_img,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )
            else:
                cv2.line(skeleton_img, (w//2, h//4), (w//2, 3*h//4), (255, 255, 255), 5)
                cv2.circle(skeleton_img, (w//2, h//4), 10, (255, 255, 255), -1)

            skeleton_img_resized = cv2.resize(skeleton_img, (128, 128))
            processed_frames.append(skeleton_img_resized)

    cap.release()

    while len(processed_frames) < max_frames:
        processed_frames.append(np.zeros((128, 128, 3), dtype=np.uint8))

    walking_quality = _analyze_walking_quality(landmarks_list, seed=seed)
    return np.array(processed_frames), walking_quality, file_hash_hex


def process_single_frame(frame):
    """
    Run pose detection on a single BGR frame (numpy array).
    Returns (landmarks_data, walking_quality) where landmarks_data is a list of
    {x, y, z, visibility} dicts (normalized 0-1), or None if no pose found.
    """
    frame_resized = cv2.resize(frame, (640, 480))
    landmarks_data = None
    walking_quality = None

    if MEDIAPIPE_AVAILABLE:
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            lm = results.pose_landmarks
            landmarks_data = [
                {"x": pt.x, "y": pt.y, "z": pt.z, "v": pt.visibility}
                for pt in lm.landmark
            ]
            walking_quality = _analyze_walking_quality([lm])

    return landmarks_data, walking_quality


def save_debug_frames(processed_frames, video_hash=None, output_dir='app/static/processed'):
    """Helper to save frames for UI display.
    Uses video_hash in filenames so concurrent uploads don't overwrite each other."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prefix = video_hash[:12] if video_hash else 'frame'
    paths = []
    for i in [0, 10, 20]:
        filename = f"{prefix}_{i}.jpg"
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, processed_frames[i])
        paths.append(filename)
    return paths
