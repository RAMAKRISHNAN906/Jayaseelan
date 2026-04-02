"""
Configuration module for Walking Detection System Enhancement.

Centralized configuration for all hyperparameters, thresholds, and deployment settings.
"""

# =============================================================================
# MODEL ARCHITECTURE (CNN-BiLSTM)
# =============================================================================
MODEL_CONFIG = {
    # CNN Backbone
    "cnn_layers": 4,
    "cnn_filters": [32, 64, 128, 256],
    "cnn_kernel_size": (3, 3),
    "cnn_activation": "relu",
    "use_mobilenetv2": False,  # Set True for MobileNetV2 backbone
    
    # BiLSTM Layers
    "lstm_units": 128,
    "lstm_layers": 2,
    "bidirectional": True,
    "lstm_dropout": 0.3,
    "recurrent_dropout": 0.2,
    
    # Dense Layers
    "dense_units": [256, 128],
    "dense_activation": "relu",
    "dropout_rate": 0.4,
    
    # Output
    "output_classes": 3,  # Low, Medium, High dementia risk
    "output_activation": "softmax",
    
    # Input
    "input_shape": (128, 128, 3),
    "sequence_length": 30,
}

# =============================================================================
# POSE EXTRACTION (MediaPipe)
# =============================================================================
POSE_CONFIG = {
    # MediaPipe Settings
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "static_image_mode": False,
    "model_complexity": 1,  # 0, 1, 2 (higher = more accurate)
    
    # Kalman Filter Smoothing
    "kalman_alpha": 0.3,  # Smoothing factor (0.3-0.5 recommended)
    "kalman_process_noise": 0.01,
    "kalman_measurement_noise": 0.1,
    
    # Visibility Filtering
    "visibility_threshold": 0.5,  # Ignore landmarks with visibility < threshold
    
    # Multi-Person Detection
    "max_persons": 3,
    "person_iou_threshold": 0.3,
    
    # Occlusion Handling
    "occlusion_threshold": 0.3,
    "visibility_weighted": True,
    
    # Camera Normalization
    "normalize_camera_angle": True,
    "height_normalization": True,
}

# =============================================================================
# REAL-TIME PROCESSING
# =============================================================================
REALTIME_CONFIG = {
    # Performance Targets
    "target_fps": 20,
    "max_latency_ms": 50,
    
    # Batch Processing
    "batch_size": 4,
    "batch_inference": True,
    
    # Frame Skipping
    "frame_skip_threshold": 0.95,  # Skip frames when confidence > threshold
    "skip_frames_count": 2,
    
    # Async Processing
    "async_mode": True,
    "queue_size": 10,
    "worker_threads": 2,
    
    # CPU Optimization
    "cpu_threads": 4,
    "use_opencv_optimization": True,
}

# =============================================================================
# CONFIDENCE SCORING
# =============================================================================
CONFIDENCE_CONFIG = {
    # EMA Smoothing
    "ema_alpha": 0.3,  # Exponential moving average factor
    
    # Thresholds
    "walking_threshold": 0.6,  # Default walking detection threshold
    "high_confidence_threshold": 0.8,
    "low_confidence_threshold": 0.4,
    
    # Calibration
    "calibration_samples": 100,
    "brier_score_target": 0.15,
}

# =============================================================================
# DATA PROCESSING & AUGMENTATION
# =============================================================================
DATA_CONFIG = {
    # Frame Processing
    "frame_size": (640, 480),
    "skeleton_size": (128, 128),
    "max_frames": 30,
    
    # Noise Reduction
    "use_kalman_filter": True,
    "kalman_denoise": True,
    
    # Frame Normalization
    "histogram_equalization": True,
    "gamma_correction": True,
    "gamma_low_light_threshold": 0.4,  # Apply gamma if brightness < threshold
    "gamma_value": 1.5,
    
    # Background Removal
    "use_mog2": False,  # Mixture of Gaussians
    "mog2_history": 500,
    
    # Data Augmentation
    "augmentation_enabled": True,
    "temporal_jittering": True,
    "jitter_range": 0.1,  # 10% timing variation
    "spatial_flipping": True,
    "scale_variation": True,
    "scale_range": [0.9, 1.1],
    "rotation": True,
    "rotation_range": [-10, 10],  # degrees
    "noise_injection": True,
    "noise_std": 0.01,
}

# =============================================================================
# TRAINING PIPELINE
# =============================================================================
TRAINING_CONFIG = {
    # Optimizer
    "optimizer": "adamw",
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "epsilon": 1e-8,
    
    # LR Scheduler
    "scheduler": "cosine_annealing",
    "t_max": 50,  # Maximum epochs
    "eta_min": 1e-5,
    
    # Early Stopping
    "early_stopping": True,
    "patience": 10,
    "min_delta": 0.001,
    "restore_best_weights": True,
    
    # Batch Size
    "batch_size": 32,
    "validation_split": 0.15,
    
    # Epochs
    "max_epochs": 100,
    
    # Hyperparameter Tuning
    "hyperparameter_tuning": True,
    "lstm_units_search": [64, 128, 256],
    "dropout_search": [0.2, 0.3, 0.4, 0.5],
    "lr_search": [1e-3, 1e-4, 1e-5],
    
    # Checkpoints
    "checkpoint_dir": "app/models/artifacts/",
    "save_best_only": True,
    "save_freq_epochs": 5,
}

# =============================================================================
# METRICS & EVALUATION
# =============================================================================
METRICS_CONFIG = {
    "primary_metrics": ["precision", "recall", "f1_score", "roc_auc"],
    "secondary_metrics": ["accuracy", "confusion_matrix"],
    "class_names": ["Low Risk", "Medium Risk", "High Risk"],
    "average_method": "weighted",  # For multi-class metrics
}

# =============================================================================
# VISUALIZATION
# =============================================================================
VISUALIZER_CONFIG = {
    # Skeleton Colors (BGR format for OpenCV)
    "high_confidence_color": (0, 255, 0),  # Green
    "medium_confidence_color": (255, 255, 0),  # Yellow
    "low_confidence_color": (255, 0, 0),  # Red
    
    # Drawing Settings
    "line_thickness": 2,
    "circle_radius": 3,
    "font_scale": 0.6,
    "font_thickness": 2,
    
    # Display Elements
    "show_skeleton": True,
    "show_walking_status": True,
    "show_bounding_box": True,
    "show_gait_metrics": True,
    "show_fps_counter": True,
    "show_confidence_score": True,
    
    # Badge Settings
    "badge_position": "top_left",
    "badge_size": (150, 50),
    
    # Metrics Display
    "metrics_position": "bottom_left",
    "metrics_font_scale": 0.5,
}

# =============================================================================
# EDGE CASE HANDLING
# =============================================================================
EDGE_CASE_CONFIG = {
    # Multi-Person
    "handle_multi_person": True,
    "aggregation_method": "weighted_average",  # weighted_average, max, mean
    
    # Low-Light Handling
    "auto_gamma_correction": True,
    "brightness_threshold": 50,  # Apply gamma if mean brightness < threshold
    
    # Occlusion
    "handle_occlusion": True,
    "min_visible_landmarks": 15,  # Minimum landmarks for valid pose
    
    # Camera Angle
    "normalize_angle": True,
    "reference_angle": 90,  # Ideal camera angle (degrees)
}

# =============================================================================
# DEPLOYMENT
# =============================================================================
DEPLOYMENT_CONFIG = {
    # Web App
    "web_async_processing": True,
    "websocket_enabled": True,
    "gunicorn_workers": 4,
    
    # Mobile (TFLite)
    "tflite_enabled": True,
    "tflite_quantization": "float16",  # float16, int8, dynamic_range
    "tflite_model_path": "app/models/artifacts/walking_detector.tflite",
    
    # Edge (Raspberry Pi)
    "edge_enabled": True,
    "edge_device": "raspberry_pi_4",
    "coral_tpu": False,
    "edge_batch_size": 1,
    "edge_threads": 2,
    
    # Docker
    "docker_gpu_support": True,
    "docker_image": "walking_detection:latest",
    "docker_port": 5001,
}

# =============================================================================
# PATHS & FILES
# =============================================================================
PATHS_CONFIG = {
    "model_dir": "app/models/",
    "artifacts_dir": "app/models/artifacts/",
    "uploads_dir": "uploads/",
    "processed_dir": "app/static/processed/",
    "plans_dir": "plans/",
    "deployment_dir": "deployment/",
}

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
GLOBAL_CONFIG = {
    "debug_mode": False,
    "log_level": "INFO",
    "random_seed": 42,
    "deterministic": True,  # For reproducibility
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_config(section):
    """Get configuration for a specific section."""
    config_map = {
        "model": MODEL_CONFIG,
        "pose": POSE_CONFIG,
        "realtime": REALTIME_CONFIG,
        "confidence": CONFIDENCE_CONFIG,
        "data": DATA_CONFIG,
        "training": TRAINING_CONFIG,
        "metrics": METRICS_CONFIG,
        "visualizer": VISUALIZER_CONFIG,
        "edge_case": EDGE_CASE_CONFIG,
        "deployment": DEPLOYMENT_CONFIG,
        "paths": PATHS_CONFIG,
        "global": GLOBAL_CONFIG,
    }
    return config_map.get(section, {})


def update_config(section, key, value):
    """Update a configuration value dynamically."""
    config_map = {
        "model": MODEL_CONFIG,
        "pose": POSE_CONFIG,
        "realtime": REALTIME_CONFIG,
        "confidence": CONFIDENCE_CONFIG,
        "data": DATA_CONFIG,
        "training": TRAINING_CONFIG,
        "metrics": METRICS_CONFIG,
        "visualizer": VISUALIZER_CONFIG,
        "edge_case": EDGE_CASE_CONFIG,
        "deployment": DEPLOYMENT_CONFIG,
        "paths": PATHS_CONFIG,
        "global": GLOBAL_CONFIG,
    }
    if section in config_map and key in config_map[section]:
        config_map[section][key] = value
        return True
    return False


def print_config_summary():
    """Print summary of all configurations."""
    print("=" * 60)
    print("WALKING DETECTION SYSTEM - CONFIGURATION SUMMARY")
    print("=" * 60)
    
    sections = [
        ("Model Architecture", MODEL_CONFIG),
        ("Pose Extraction", POSE_CONFIG),
        ("Real-Time Processing", REALTIME_CONFIG),
        ("Confidence Scoring", CONFIDENCE_CONFIG),
        ("Data Processing", DATA_CONFIG),
        ("Training Pipeline", TRAINING_CONFIG),
        ("Metrics", METRICS_CONFIG),
        ("Visualization", VISUALIZER_CONFIG),
        ("Edge Cases", EDGE_CASE_CONFIG),
        ("Deployment", DEPLOYMENT_CONFIG),
    ]
    
    for name, config in sections:
        print(f"\n{name}:")
        for key, value in config.items():
            if not isinstance(value, (dict, list)):
                print(f"  - {key}: {value}")
    
    print("=" * 60)