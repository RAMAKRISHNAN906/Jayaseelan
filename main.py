import base64
import hashlib
import json
import os
import threading
import time
import traceback
import uuid
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from dotenv import load_dotenv
from firebase_admin import auth, db
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user

import cv2

from app.models.cnn_model import load_dementia_model, predict_risk
from app.utils.firebase_config import initialize_firebase
from app.utils.gait_processor import process_video, save_debug_frames, process_single_frame

# In-memory cache: key = "sha256hash_tasktype" -> {prediction, debug_frames}
# Guarantees identical results for the same video uploaded multiple times.
_prediction_cache = {}
_prediction_cache_lock = threading.Lock()

# In-memory background extraction jobs.
_background_jobs = {}
_background_jobs_lock = threading.Lock()
JOB_TTL_SECONDS = 2 * 60 * 60

load_dotenv()

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
app.secret_key = 'dementia_detection_secret_key'


def _now_iso():
    """UTC timestamp in ISO format."""
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


def _cleanup_old_jobs():
    """Trim stale in-memory jobs to avoid unlimited growth."""
    cutoff = time.time() - JOB_TTL_SECONDS
    with _background_jobs_lock:
        stale_job_ids = [
            job_id
            for job_id, job in _background_jobs.items()
            if job.get("updated_ts", 0) < cutoff
        ]
        for job_id in stale_job_ids:
            _background_jobs.pop(job_id, None)


def _get_user_job(job_id, user_id):
    with _background_jobs_lock:
        job = _background_jobs.get(job_id)
        if not job or job.get("user_id") != user_id:
            return None
        return dict(job)


def _set_job_fields(job_id, **kwargs):
    with _background_jobs_lock:
        job = _background_jobs.get(job_id)
        if not job:
            return
        job.update(kwargs)
        job["updated_at"] = _now_iso()
        job["updated_ts"] = time.time()


def _build_video_url(video_bytes, video_sha256, original_filename):
    ext = os.path.splitext(original_filename or "")[1] or ".mp4"
    ext = "".join(ch for ch in ext if ch.isalnum() or ch == ".")
    if not ext.startswith("."):
        ext = f".{ext}"
    if len(ext) > 12:
        ext = ".mp4"

    static_video_name = f"{video_sha256[:16]}{ext.lower()}"
    static_video_path = os.path.join('app', 'static', 'uploads', static_video_name)
    os.makedirs(os.path.join('app', 'static', 'uploads'), exist_ok=True)

    if not os.path.exists(static_video_path):
        with open(static_video_path, 'wb') as f:
            f.write(video_bytes)

    return f"uploads/{static_video_name}"


def _safe_upload_filename(original_filename, video_sha256):
    base_name = os.path.basename(original_filename or "upload.mp4")
    name_root, ext = os.path.splitext(base_name)
    if not ext:
        ext = ".mp4"

    safe_root = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in name_root).strip("_")
    safe_root = safe_root[:40] or "video"

    return f"{video_sha256[:12]}_{safe_root}{ext}"


def _run_prediction_job(job_id, user_id, task_type, original_filename, video_bytes):
    _set_job_fields(job_id, status="processing")

    try:
        video_sha256 = hashlib.sha256(video_bytes).hexdigest()
        cache_key = f"{video_sha256}_{task_type}"
        video_url = _build_video_url(video_bytes, video_sha256, original_filename)

        with _prediction_cache_lock:
            cached = _prediction_cache.get(cache_key)

        if cached:
            _set_job_fields(
                job_id,
                status="done",
                result={
                    "prediction": cached["prediction"],
                    "task_type": task_type,
                    "debug_frames": cached["debug_frames"],
                    "video_url": video_url,
                },
                error=None,
            )
            return

        os.makedirs('uploads', exist_ok=True)
        upload_filename = _safe_upload_filename(original_filename, video_sha256)
        video_path = os.path.join('uploads', upload_filename)
        with open(video_path, 'wb') as f:
            f.write(video_bytes)

        # Pass the already-computed hash to skip re-reading the file for hashing.
        processed_frames, walking_quality, _ = process_video(video_path, file_hash_hex=video_sha256)
        debug_frames = save_debug_frames(processed_frames, video_hash=video_sha256)

        model = load_dementia_model()
        prediction = predict_risk(model, processed_frames, task_type, walking_quality)

        with _prediction_cache_lock:
            _prediction_cache[cache_key] = {
                "prediction": prediction,
                "debug_frames": debug_frames,
            }

        # Store in Firebase DB (if initialized).
        try:
            ref = db.reference(f'results/{user_id}')
            ref.push({
                'task_type': task_type,
                'risk_score': prediction['score'],
                'risk_level': prediction['level'],
                'timestamp': str(np.datetime64('now')),
            })
        except Exception:
            print("Firebase DB not initialized, skipping storage.")

        _set_job_fields(
            job_id,
            status="done",
            result={
                "prediction": prediction,
                "task_type": task_type,
                "debug_frames": debug_frames,
                "video_url": video_url,
            },
            error=None,
        )
    except Exception as exc:
        print(f"[PredictJob] Job {job_id} failed: {exc}")
        traceback.print_exc()
        _set_job_fields(
            job_id,
            status="error",
            error=f"Video extraction failed: {exc}",
        )


def build_firebase_web_config():
    return {
        "apiKey": os.getenv("FIREBASE_API_KEY", "AIzaSyCJww8tw7222PKCLWQxf7Iy1Ptjh_rn_Eo"),
        "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", "walking-analysis.firebaseapp.com"),
        "projectId": os.getenv("FIREBASE_PROJECT_ID", "walking-analysis"),
        "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET", "walking-analysis.firebasestorage.app"),
        "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID", "622747970386"),
        "appId": os.getenv("FIREBASE_APP_ID", "1:622747970386:web:1d8dd5e2e1a0b29f3fa88b"),
        "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID", "G-GYT1W9PQJE"),
    }


FIREBASE_WEB_CONFIG = build_firebase_web_config()

# Initialize Firebase
firebase_app = initialize_firebase()
FIREBASE_READY = firebase_app is not None

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin):
    def __init__(self, uid, email):
        self.id = uid
        self.email = email


def verify_token_with_identity_toolkit(id_token):
    """Verifies ID token through Firebase Identity Toolkit when Admin SDK isn't configured."""
    api_key = FIREBASE_WEB_CONFIG.get("apiKey")
    if not api_key:
        return None

    endpoint = f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={api_key}"
    payload = json.dumps({"idToken": id_token}).encode("utf-8")
    request_obj = Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request_obj, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        print(f"[Auth] Identity Toolkit HTTP error {e.code}: {body}")
        return None
    except (URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"[Auth] Identity Toolkit error: {e}")
        return None

    users = data.get("users", [])
    if not users:
        return None

    user_info = users[0]
    uid = user_info.get("localId")
    if not uid:
        return None

    return {"uid": uid, "email": user_info.get("email", "")}


@login_manager.user_loader
def load_user(user_id):
    if FIREBASE_READY:
        try:
            user = auth.get_user(user_id)
            return User(user.uid, user.email or session.get("user_email", ""))
        except Exception:
            return None

    # Fallback path for environments without Admin SDK key.
    email = session.get("user_email")
    if email:
        return User(user_id, email)

    return None


@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login')
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('login.html', firebase_config=FIREBASE_WEB_CONFIG)


@app.route('/register')
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('register.html', firebase_config=FIREBASE_WEB_CONFIG)


@app.route('/session_login', methods=['POST'])
def session_login():
    payload = request.get_json(silent=True) or {}
    id_token = payload.get('idToken')

    if not id_token:
        return jsonify({"success": False, "message": "Missing Firebase ID token."}), 400

    verified_user = None

    if FIREBASE_READY:
        try:
            decoded_token = auth.verify_id_token(id_token)
            uid = decoded_token.get("uid")
            user_record = auth.get_user(uid)
            verified_user = {"uid": user_record.uid, "email": user_record.email or decoded_token.get("email", "")}
            print(f"[Auth] Admin SDK verified: {verified_user['email']}")
        except Exception as e:
            print(f"[Auth] Admin SDK failed: {e}")
            verified_user = None

    if verified_user is None:
        print("[Auth] Trying Identity Toolkit fallback...")
        verified_user = verify_token_with_identity_toolkit(id_token)
        if verified_user:
            print(f"[Auth] Identity Toolkit verified: {verified_user.get('email')}")
        else:
            print("[Auth] Identity Toolkit also failed.")

    if not verified_user:
        return jsonify({"success": False, "message": "Invalid or expired Firebase token."}), 401

    session["user_email"] = verified_user.get("email", "")
    login_user(User(verified_user["uid"], verified_user.get("email", "")))
    print(f"[Auth] Session created for: {verified_user.get('email')}")

    return jsonify({"success": True, "redirect": url_for("dashboard")})


@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop("user_email", None)
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user)


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    task_type = (request.form.get('task_type') or 'single').strip().lower()
    if task_type not in ('single', 'dual'):
        task_type = 'single'

    video_file = request.files.get('video')
    if not video_file:
        flash("No video uploaded")
        return redirect(url_for('dashboard'))

    video_bytes = video_file.read()
    if not video_bytes:
        flash("Uploaded video is empty. Please upload a valid file.")
        return redirect(url_for('dashboard'))

    _cleanup_old_jobs()
    job_id = uuid.uuid4().hex
    now = _now_iso()
    file_name = video_file.filename or "uploaded_video.mp4"

    with _background_jobs_lock:
        _background_jobs[job_id] = {
            "id": job_id,
            "user_id": current_user.id,
            "status": "queued",
            "task_type": task_type,
            "file_name": file_name,
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
            "updated_ts": time.time(),
        }

    worker = threading.Thread(
        target=_run_prediction_job,
        args=(job_id, current_user.id, task_type, file_name, video_bytes),
        daemon=True,
        name=f"predict-{job_id[:8]}",
    )
    worker.start()

    return render_template(
        'processing.html',
        job_id=job_id,
        task_type=task_type,
        file_name=file_name,
    )


@app.route('/predict_status/<job_id>')
@login_required
def predict_status(job_id):
    _cleanup_old_jobs()
    job = _get_user_job(job_id, current_user.id)
    if not job:
        return jsonify({"success": False, "message": "Job not found or expired."}), 404

    payload = {
        "success": True,
        "status": job.get("status"),
        "updated_at": job.get("updated_at"),
    }

    if job.get("status") == "done":
        payload["result_url"] = url_for('predict_result', job_id=job_id)
    elif job.get("status") == "error":
        payload["error"] = job.get("error") or "Video extraction failed."

    return jsonify(payload)


@app.route('/predict_result/<job_id>')
@login_required
def predict_result(job_id):
    _cleanup_old_jobs()
    job = _get_user_job(job_id, current_user.id)
    if not job:
        flash("Result not found or expired. Please upload again.")
        return redirect(url_for('dashboard'))

    status = job.get("status")
    if status in ("queued", "processing"):
        return render_template(
            'processing.html',
            job_id=job_id,
            task_type=job.get("task_type", "single"),
            file_name=job.get("file_name", "uploaded_video.mp4"),
        )

    if status == "error":
        flash(job.get("error") or "Video extraction failed.")
        return redirect(url_for('dashboard'))

    result = job.get("result") or {}
    prediction = result.get("prediction")
    if not prediction:
        flash("Result payload is unavailable. Please upload again.")
        return redirect(url_for('dashboard'))

    return render_template(
        'results.html',
        prediction=prediction,
        task_type=result.get("task_type", "single"),
        debug_frames=result.get("debug_frames", []),
        video_url=result.get("video_url"),
    )


@app.route('/memory_test')
@login_required
def memory_test():
    return render_template('memory_test.html', user=current_user)


@app.route('/realtime')
@login_required
def realtime():
    return render_template('realtime.html', user=current_user)


@app.route('/realtime_frame', methods=['POST'])
@login_required
def realtime_frame():
    payload = request.get_json(silent=True) or {}
    frame_b64 = payload.get('frame', '')
    if not frame_b64:
        return jsonify({"success": False, "error": "No frame data"}), 400

    try:
        img_bytes = base64.b64decode(frame_b64.split(',')[-1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

    landmarks_data, walking_quality = process_single_frame(frame)
    return jsonify({
        "success": True,
        "landmarks": landmarks_data,
        "walking_quality": walking_quality,
    })


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('RAILWAY_ENVIRONMENT') is None
    app.run(debug=debug, host='0.0.0.0', port=port)
