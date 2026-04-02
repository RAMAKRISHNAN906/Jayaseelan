"""
CNN-BiLSTM Hybrid Model for Temporal Gait Analysis (PyTorch).

Replaces basic CNN with CNN-BiLSTM architecture for sequential gait pattern learning.
Includes MobileNetV2 backbone option, dropout regularization, and confidence scoring.
"""

import numpy as np
import os
import sys
from typing import Dict, Tuple, Optional, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import models
    TORCH_AVAILABLE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
    nn = None
    optim = None
    device = "cpu"

from app.utils.config import MODEL_CONFIG, TRAINING_CONFIG


class CNNBiLSTMModel(nn.Module):
    """CNN-BiLSTM hybrid architecture for gait analysis."""
    
    def __init__(self, config: Optional[Dict] = None):
        super(CNNBiLSTMModel, self).__init__()
        
        config = config or MODEL_CONFIG
        
        # CNN backbone parameters
        use_mobilenet = config.get("use_mobilenetv2", False)
        input_channels = config.get("input_channels", 3)
        
        # Build CNN backbone
        if use_mobilenet:
            self.cnn_backbone = self._build_mobilenet_backbone()
            cnn_output_dim = 1280  # MobileNetV2 output dim
        else:
            self.cnn_backbone = self._build_custom_cnn_backbone(config)
            cnn_output_dim = 256
        
        # BiLSTM layers
        lstm_units = config.get("lstm_units", 128)
        lstm_layers = config.get("lstm_layers", 2)
        lstm_dropout = config.get("lstm_dropout", 0.3)
        bidirectional = config.get("bidirectional", True)
        
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=lstm_units,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0
        )
        
        # Dense layers
        lstm_output_dim = lstm_units * 2 if bidirectional else lstm_units
        dense_units = config.get("dense_units", [256, 128])
        dropout_rate = config.get("dropout_rate", 0.4)
        
        self.dense_layers = nn.ModuleList()
        input_dim = lstm_output_dim
        for units in dense_units:
            self.dense_layers.append(nn.Linear(input_dim, units))
            self.dense_layers.append(nn.ReLU())
            self.dense_layers.append(nn.Dropout(dropout_rate))
            input_dim = units
        
        # Output layers
        output_classes = config.get("output_classes", 3)
        self.classifier = nn.Linear(input_dim, output_classes)
        self.confidence_head = nn.Linear(input_dim, 1)
        
    def _build_custom_cnn_backbone(self, config: Dict) -> nn.Module:
        """Build custom CNN backbone."""
        
        cnn_layers = config.get("cnn_layers", 4)
        filters = config.get("cnn_filters", [32, 64, 128, 256])
        kernel_size = config.get("cnn_kernel_size", 3)
        
        layers = []
        in_channels = config.get("input_channels", 3)
        
        for i in range(cnn_layers):
            out_channels = filters[i] if i < len(filters) else filters[-1]
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            layers.append(nn.Dropout(0.2))
            in_channels = out_channels
        
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        
        return nn.Sequential(*layers)
    
    def _build_mobilenet_backbone(self) -> nn.Module:
        """Build MobileNetV2 backbone."""
        
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Remove classifier
        backbone = nn.Sequential(*list(mobilenet.children())[:-1])
        
        # Freeze backbone
        for param in backbone.parameters():
            param.requires_grad = False
        
        return backbone
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, channels, height, width)
            
        Returns:
            (class_logits, confidence)
        """
        batch_size, seq_len = x.shape[:2]
        
        # Reshape for CNN: (batch * seq_len, channels, height, width)
        x = x.view(batch_size * seq_len, *x.shape[2:])
        
        # CNN feature extraction
        cnn_features = self.cnn_backbone(x)
        
        # Reshape for LSTM: (batch, seq_len, features)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        
        # BiLSTM
        lstm_out, _ = self.lstm(cnn_features)
        
        # Use last output
        final_hidden = lstm_out[:, -1, :]
        
        # Dense layers
        x = final_hidden
        for layer in self.dense_layers:
            x = layer(x)
        
        # Outputs
        class_logits = self.classifier(x)
        confidence = torch.sigmoid(self.confidence_head(x))
        
        return class_logits, confidence


def build_cnn_biLSTM_model(config: Optional[Dict] = None) -> nn.Module:
    """Build CNN-BiLSTM model."""
    
    if not TORCH_AVAILABLE:
        print("Warning: PyTorch not available. Returning None.")
        return None
    
    model = CNNBiLSTMModel(config)
    model = model.to(device)
    
    return model


def build_adamw_optimizer(model: nn.Module, config: Dict) -> optim.Optimizer:
    """Build AdamW optimizer with weight decay."""
    
    lr = config.get("learning_rate", 1e-3)
    weight_decay = config.get("weight_decay", 0.01)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    return optimizer


def cosine_annealing_scheduler(optimizer: optim.Optimizer, 
                                T_max: int, 
                                eta_min: float = 0.0) -> optim.lr_scheduler.CosineAnnealingLR:
    """Create cosine annealing learning rate scheduler."""
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=T_max,
        eta_min=eta_min
    )
    
    return scheduler


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
        
    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
        elif self.mode == "min" and value < self.best_value - self.min_delta:
            self.best_value = value
            self.counter = 0
        elif self.mode == "max" and value > self.best_value + self.min_delta:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def predict_risk(model: nn.Module, 
                 processed_frames: np.ndarray, 
                 task_type: str = "single",
                 walking_quality: Dict = None) -> Dict:
    """
    Predict dementia risk from gait sequence.
    
    Args:
        model: CNN-BiLSTM model
        processed_frames: Sequence of frames
        task_type: "single" or "dual" task
        walking_quality: Dict with walking quality metrics
        
    Returns:
        Dict with risk score, level, confidence, and metrics
    """
    
    if not TORCH_AVAILABLE or model is None:
        return _mock_prediction(task_type, walking_quality)
    
    # Convert to tensor
    if len(processed_frames.shape) == 3:
        processed_frames = np.expand_dims(processed_frames, axis=0)
    
    if len(processed_frames.shape) == 4:
        processed_frames = np.expand_dims(processed_frames, axis=0)
    
    # Convert to torch tensor
    frames_tensor = torch.from_numpy(processed_frames).float().to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        class_logits, confidence = model(frames_tensor)
    
    # Get class probabilities
    class_probs = torch.softmax(class_logits, dim=1)
    class_idx = class_probs.argmax(1).item()
    
    # Map to risk level
    class_names = ["Low Risk", "Medium Risk", "High Risk"]
    risk_level = class_names[class_idx]
    
    # Risk score (0-100, higher = worse)
    risk_score = round(class_probs[0, class_idx].item() * 100, 1)
    
    # Walking score (inverse of risk)
    walking_score = round(100 - risk_score, 1)
    
    # Dual-task penalty
    if task_type == "dual":
        walking_score = round(walking_score * 0.85, 1)
        risk_score = round(100 - walking_score, 1)
    
    # Confidence score (0-100%)
    confidence_score = round(confidence[0, 0].item() * 100, 1)
    
    # Determine color
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
        "confidence": confidence_score,
        "class_probs": class_probs[0].tolist(),
        "metrics": walking_quality or {},
    }


def _mock_prediction(task_type: str = "single", walking_quality: Dict = None) -> Dict:
    """Fallback prediction when model not available."""
    
    walking_quality = walking_quality or {"overall": 70.0}
    overall = walking_quality.get("overall", 70.0)
    
    if task_type == "dual":
        overall = overall * 0.85
    
    risk_score = round(100 - overall, 1)
    
    if risk_score < 35:
        level = "Low Risk"
        color = "success"
    elif risk_score < 65:
        level = "Medium Risk"
        color = "warning"
    else:
        level = "High Risk"
        color = "danger"
    
    return {
        "score": risk_score,
        "walking_score": round(overall, 1),
        "level": level,
        "color": color,
        "walking_label": "Walking",
        "confidence": 50.0,
        "class_probs": [0.33, 0.33, 0.34],
        "metrics": walking_quality,
    }


def save_model(model: nn.Module, save_path: str) -> bool:
    """Save model weights."""
    
    try:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False


def load_model_weights(model: nn.Module, load_path: str, config: Dict = None) -> nn.Module:
    """Load model weights."""
    
    if not os.path.exists(load_path):
        print(f"Weights file not found: {load_path}")
        return model
    
    try:
        state_dict = torch.load(load_path, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Model loaded from: {load_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return model


if __name__ == "__main__":
    # Test model building
    print("Building CNN-BiLSTM model...")
    
    if TORCH_AVAILABLE:
        model = build_cnn_biLSTM_model()
        
        if model:
            print(f"Model built successfully!")
            print(f"Device: {device}")
            
            # Print model structure
            print(model)
            
            # Test prediction with random data
            print("\nTesting prediction...")
            test_input = np.random.randn(1, 30, 3, 128, 128)
            result = predict_risk(model, test_input)
            print(f"Prediction result: {result}")
    else:
        print("PyTorch not available, testing mock prediction...")
        result = predict_risk(None, np.random.randn(1, 30, 3, 128, 128))
        print(f"Mock prediction: {result}")