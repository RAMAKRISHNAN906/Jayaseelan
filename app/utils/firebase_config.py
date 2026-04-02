import firebase_admin
from firebase_admin import credentials
import os
import json

FIREBASE_PROJECT_ID = 'walking-analysis'
DEFAULT_DATABASE_URL = f'https://{FIREBASE_PROJECT_ID}-default-rtdb.firebaseio.com/'

def initialize_firebase():
    """Initializes the Firebase Admin SDK.
    Only initializes if the service account key matches the configured project."""
    if firebase_admin._apps:
        return firebase_admin.get_app()

    cred_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH', os.path.join(os.getcwd(), 'serviceAccountKey.json'))
    if not os.path.exists(cred_path):
        print("Warning: serviceAccountKey.json not found. Firebase Admin SDK is disabled.")
        return None

    # Validate the service account belongs to the correct Firebase project
    try:
        with open(cred_path) as f:
            key_data = json.load(f)
        key_project = key_data.get('project_id', '')
        if key_project != FIREBASE_PROJECT_ID:
            print(f"Warning: serviceAccountKey.json is for project '{key_project}' "
                  f"but app is configured for '{FIREBASE_PROJECT_ID}'. "
                  f"Admin SDK disabled — using Identity Toolkit fallback.")
            return None
    except Exception as exc:
        print(f"Warning: Could not read serviceAccountKey.json: {exc}")
        return None

    database_url = os.getenv('FIREBASE_DATABASE_URL', DEFAULT_DATABASE_URL)

    try:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {'databaseURL': database_url})
        print(f"Firebase Admin SDK initialized for project: {FIREBASE_PROJECT_ID}")
        return firebase_admin.get_app()
    except Exception as exc:
        print(f"Warning: Firebase initialization failed: {exc}")
        return None
