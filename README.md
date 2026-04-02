# Vision-Based Contactless Dementia Detection Framework

College final year project focused on CNN-driven single and dual-task gait analysis.

## Features
- **User Authentication**: Secure login and registration using Firebase.
- **Gait Analysis**: 
  - Single-task (Normal walking).
  - Dual-task (Walking + mentally demanding task).
- **Processing**: Pose detection via MediaPipe, converted to skeletal silhouettes.
- **Deep Learning**: CNN model to classify risk levels (Low/Medium/High).
- **Dashboard**: Easy upload and result visualization.

## Setup Instructions

### 1. Requirements
Ensure you have Python 3.8+ installed. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Firebase Configuration
1. Go to [Firebase Console](https://console.firebase.google.com/).
2. Create a new project.
3. Enable **Authentication** (Email/Password).
4. Enable **Realtime Database**.
5. Go to Project Settings > Service Accounts.
6. Generate a new private key and save it as `serviceAccountKey.json` in the root folder.
7. Update the `databaseURL` in `app/utils/firebase_config.py`.

### 3. Run the Application
```bash
python main.py
```

## How it works
1. **User Login**: Users register and sign in.
2. **Video Upload**: A video of a person walking is uploaded.
3. **Gait Extraction**: MediaPipe extracts 30 frames of skeletal landmarks.
4. **CNN Prediction**: The skeletal sequence is passed through a CNN to determine dementia risk.
5. **Results**: Dashboard displays the risk score and processed silhouettes.

## Technical Stack
- **Backend**: Flask
- **Vision**: OpenCV, MediaPipe
- **Deep Learning**: TensorFlow/Keras
- **Database**: Firebase Admin SDK
- **Frontend**: Bootstrap 5
