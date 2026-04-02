# Walking Detection System Enhancement Plan

## Current State Analysis
- **Model**: Basic CNN (3 conv layers) for dementia risk classification
- **Pose Extraction**: MediaPipe Pose with 33 landmarks (basic implementation)
- **Processing**: 30 frames, simple skeleton visualization
- **Walking Quality**: Heuristic scoring (spine, shoulder, hip, head, step symmetry)
- **Framework**: Flask web app with Firebase authentication
- **GPU**: NVIDIA RTX 3050 6GB available

## Enhancement Objectives

### Phase 1: Architecture Upgrade (CNN → CNN-BiLSTM)
**Goal**: Temporal gait analysis with sequence modeling

**Changes**:
1. Replace flat CNN with CNN-BiLSTM hybrid
   - CNN layers for spatial feature extraction from skeletal frames
   - BiLSTM layers (64-128 units) for temporal gait pattern learning
   - Attention mechanism for key gait events
2. Input shape: (sequence_length, 128, 128, 3) → Output: 3-class risk probability
3. Add dropout (0.3-0.5) for regularization

**Files**: `app/models/cnn_biLSTM_model.py`

---

### Phase 2: Enhanced Pose & Motion Analysis
**Goal**: Comprehensive gait cycle metrics

**Changes**:
1. Full 33 MediaPipe landmarks extraction
2. Kalman filter smoothing for landmark trajectories
3. Visibility filtering (ignore low-confidence landmarks)
4. Gait metrics computation:
   - Step frequency (steps/second)
   - Stride length (normalized by height)
   - Joint angles (hip, knee, ankle)
   - Velocity profiles (joint velocities)
   - Gait cycle phase detection

**Files**: `app/utils/pose_extractor.py`, `app/utils/gait_metrics.py`

---

### Phase 3: Data Processing Pipeline
**Goal**: Robust preprocessing and augmentation

**Changes**:
1. Noise reduction: Kalman filter on landmark sequences
2. Frame normalization:
   - Histogram equalization
   - Gamma correction (auto-adjust for low-light)
3. Background removal: MOG2 (Mixture of Gaussians)
4. Data augmentation:
   - Temporal jittering (frame timing variation)
   - Spatial flipping (horizontal)
   - Scale variation
   - Rotation (small angles)

**Files**: `app/utils/frame_processor.py`, `app/utils/augmentation.py`

---

### Phase 4: Real-Time Performance Optimization
**Goal**: ≥20 FPS on CPU, edge deployment ready

**Changes**:
1. MobileNetV2 backbone for feature extraction
2. Batch inference (process 4-8 frames together)
3. Frame skipping when confidence >0.95
3. TensorFlow Lite conversion for edge deployment
4. Async processing pipeline

**Files**: `app/utils/realtime_processor.py`, `app/models/tflite_converter.py`

---

### Phase 5: Confidence Score System
**Goal**: Reliable confidence metrics with smoothing

**Changes**:
1. Output 0-100% confidence score
2. Exponential moving average smoothing (alpha=0.3)
3. Tunable walking threshold (default 0.6)
4. Confidence calibration on validation set

**Files**: `app/utils/confidence_tracker.py`

---

### Phase 6: Visual Output Enhancement
**Goal**: Rich visual feedback for users

**Changes**:
1. Skeleton overlay with color-coded joints:
   - Green: high confidence
   - Yellow: medium confidence
   - Red: low confidence
2. Walking status badge (Walking/Not Walking/Uncertain)
3. Bounding boxes around detected persons
4. Gait metrics display (step frequency, stride length)
5. FPS counter for real-time mode

**Files**: `app/utils/visualizer.py`

---

### Phase 7: Model Training Pipeline
**Goal**: Production-ready training with hyperparameter tuning

**Changes**:
1. AdamW optimizer (weight decay=0.01)
2. Cosine annealing learning rate scheduler
3. Early stopping (patience=10, min_delta=0.001)
4. Metrics: Precision, Recall, F1-score, ROC-AUC
5. Hyperparameter tuning:
   - LSTM units: [64, 128, 256]
   - Dropout: [0.2, 0.3, 0.5]
   - Learning rate: [1e-3, 1e-4, 1e-5]
6. Checkpoint saving (best model)

**Files**: `model/src/train_cnn_biLSTM.py`, `model/src/hyperparameter_search.py`

---

### Phase 8: Edge Case Handling
**Goal**: Robust performance in challenging conditions

**Changes**:
1. Multi-person detection:
   - Process each person separately
   - Aggregate predictions (weighted average)
2. Low-light handling:
   - Auto gamma correction (detect low brightness)
   - Adaptive histogram equalization
3. Occlusion handling:
   - Visibility-weighted feature computation
   - Partial pose reconstruction
4. Camera angle normalization:
   - Perspective correction
   - Height normalization

**Files**: `app/utils/multi_person_detector.py`, `app/utils/lighting_adapter.py`

---

### Phase 9: Code Refactoring
**Goal**: Modular, maintainable architecture

**New Module Structure**:
```
app/
├── models/
│   ├── cnn_biLSTM_model.py      # Main architecture
│   ├── tflite_converter.py      # Edge conversion
│   └── model_loader.py          # Unified loading
├── utils/
│   ├── pose_extractor.py        # MediaPipe extraction
│   ├── feature_engineer.py      # Gait features
│   ├── walking_analyzer.py      # Walking classification
│   ├── realtime_processor.py   # Real-time pipeline
│   ├── visualizer.py            # Visualization
│   ├── confidence_tracker.py    # Confidence smoothing
│   ├── frame_processor.py       # Preprocessing
│   ├── augmentation.py          # Data augmentation
│   ├── gait_metrics.py          # Metric computation
│   └── config.py                # Configuration
└── templates/                   # Updated UI
```

**Files**: All new modular files, refactor main.py imports

---

### Phase 10: Deployment Configuration
**Goal**: Production deployment ready

**Changes**:
1. Web app deployment:
   - Async processing queue
   - WebSocket for real-time updates
2. Mobile deployment:
   - TensorFlow Lite model
   - Android/iOS integration guide
3. Edge deployment:
   - Raspberry Pi configuration
   - Coral TPU optimization
4. Docker containerization:
   - Dockerfile with GPU support
   - .env.example with all configs
5. CI/CD pipeline config

**Files**: `Dockerfile`, `.env.example`, `deployment/README.md`, `deployment/mobile/`, `deployment/edge/`

---

## Implementation Sequence

1. **Week 1**: Phase 1 (CNN-BiLSTM), Phase 9 (Refactoring foundation)
2. **Week 2**: Phase 2 (Pose enhancement), Phase 3 (Data processing)
3. **Week 3**: Phase 7 (Training pipeline), Phase 5 (Confidence system)
4. **Week 4**: Phase 6 (Visualization), Phase 8 (Edge cases)
5. **Week 5**: Phase 4 (Real-time optimization), Phase 10 (Deployment)

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Accuracy | ~70% (estimated) | ≥85% |
| FPS (CPU) | ~10 | ≥20 |
| Confidence Calibration | N/A | Brier score <0.15 |
| False Positive Rate | ~15% | <8% |
| Model Size | ~5MB | <10MB (TFLite <3MB) |

## Technical Dependencies

- TensorFlow 2.15+ (for BiLSTM, TFLite)
- MediaPipe Pose 0.10+
- OpenCV 4.8+
- scikit-learn (metrics)
- NumPy, Pandas

## Risk Mitigation

1. **GPU Memory**: Use gradient accumulation for large batches
2. **Overfitting**: Heavy augmentation, dropout, early stopping
3. **Real-time Performance**: Model quantization, batch optimization
4. **Data Scarsity**: Transfer learning from pose estimation datasets

---

*Plan created: 2026-04-01*
*Next Action: Begin Phase 1 implementation*