try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
import numpy as np

def build_cnn_model():
    """Defines a simple CNN architecture for Gait Analysis."""
    if not TENSORFLOW_AVAILABLE:
        print("Warning: TensorFlow not found. Using mock model for demo.")
        return None
        
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax') # 3 classes: Low, Medium, High Risk
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def load_dementia_model():
    """Builds and returns the model. In production, we'd load weights here."""
    model = build_cnn_model()
    # model.load_weights('app/models/gait_cnn_weights.h5') # Placeholder for weights
    return model

def predict_risk(model, processed_frames, task_type, walking_quality=None):
    """
    Computes dementia risk from walking quality metrics.

    walking_quality is a dict produced by gait_processor._analyze_walking_quality():
      { 'overall', 'spine', 'shoulder', 'hip', 'head', 'step_symmetry' }

    A high walking quality score → Low Risk (good, straight walking).
    A low walking quality score  → High Risk (poor, irregular walking).
    """
    if walking_quality is None:
        walking_quality = {'overall': 50.0, 'spine': 50.0, 'shoulder': 50.0,
                           'hip': 50.0, 'head': 50.0, 'step_symmetry': 50.0}

    overall = walking_quality['overall']   # 0-100, higher = better walking

    # Dual-task adds cognitive load penalty (realistic dementia indicator)
    if task_type == 'dual':
        overall = overall * 0.85

    # Risk score is the inverse: bad walking → high risk number
    risk_score = round(100 - overall, 1)

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
        'score':          risk_score,          # risk score (higher = worse)
        'walking_score':  round(overall, 1),   # walking quality (higher = better)
        'level':          level,
        'color':          color,
        'walking_label':  walking_label,
        'metrics':        walking_quality,
    }
