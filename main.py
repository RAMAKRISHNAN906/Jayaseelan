import base64
import hashlib
import json
import os
import threading
import time
import traceback
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

# Persistent disk cache: results survive server restarts.
# key = "sha256[:16]_tasktype"  value = {prediction, debug_frames}
_prediction_cache = {}
_prediction_cache_lock = threading.Lock()
_CACHE_DIR = os.path.join('app', 'static', 'cache')

def _cache_path(cache_key):
    safe = cache_key.replace('/', '_')
    return os.path.join(_CACHE_DIR, f"{safe}.json")

def _load_cache(cache_key):
    """Load result from disk cache. Returns None if not found."""
    try:
        p = _cache_path(cache_key)
        if os.path.exists(p):
            with open(p, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return None

def _save_cache(cache_key, data):
    """Save result to disk cache."""
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_cache_path(cache_key), 'w') as f:
            json.dump(data, f)
    except Exception:
        pass

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
    _set_job_fields(job_id, status="processing", progress=5, step="Preparing video…")
    print(f"[PredictJob] {job_id[:8]} started for user {user_id}")

    try:
        video_sha256 = hashlib.sha256(video_bytes).hexdigest()
        cache_key = f"{video_sha256}_{task_type}"
        video_url = _build_video_url(video_bytes, video_sha256, original_filename)

        with _prediction_cache_lock:
            cached = _prediction_cache.get(cache_key)

        if cached:
            print(f"[PredictJob] {job_id[:8]} served from cache")
            _set_job_fields(
                job_id,
                status="done",
                progress=100,
                step="Complete",
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

        _set_job_fields(job_id, progress=15, step="Reading video frames…")
        print(f"[PredictJob] {job_id[:8]} extracting frames…")

        # Pass the already-computed hash to skip re-reading the file for hashing.
        processed_frames, walking_quality, _ = process_video(video_path, file_hash_hex=video_sha256)
        print(f"[PredictJob] {job_id[:8]} frames done — walking_quality={walking_quality.get('overall')}")

        _set_job_fields(job_id, progress=60, step="Scoring biomechanics…")
        debug_frames = save_debug_frames(processed_frames, video_hash=video_sha256)

        _set_job_fields(job_id, progress=75, step="Running risk model…")
        print(f"[PredictJob] {job_id[:8]} running CNN…")
        model = load_dementia_model()
        prediction = predict_risk(model, processed_frames, task_type, walking_quality)
        print(f"[PredictJob] {job_id[:8]} prediction={prediction.get('level')} score={prediction.get('score')}")

        _set_job_fields(job_id, progress=90, step="Saving results…")

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
            progress=100,
            step="Complete",
            result={
                "prediction": prediction,
                "task_type": task_type,
                "debug_frames": debug_frames,
                "video_url": video_url,
            },
            error=None,
        )
        print(f"[PredictJob] {job_id[:8]} DONE")
    except Exception as exc:
        print(f"[PredictJob] Job {job_id} failed: {exc}")
        traceback.print_exc()
        _set_job_fields(
            job_id,
            status="error",
            progress=0,
            step="Failed",
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
    return redirect(url_for('dashboard_clinical'))


@app.route('/technology')
@login_required
def dashboard_technology():
    return render_template('dashboard.html', user=current_user, active_tab='technology')


@app.route('/clinical-data')
@login_required
def dashboard_clinical():
    return render_template('dashboard.html', user=current_user, active_tab='clinical')


@app.route('/gait-analysis')
@login_required
def dashboard_gait():
    return render_template('dashboard.html', user=current_user, active_tab='gait')


@app.route('/recall-iq')
@login_required
def dashboard_recall():
    return render_template('dashboard.html', user=current_user, active_tab='recall')


@app.route('/feedback')
@login_required
def dashboard_feedback():
    return render_template('dashboard.html', user=current_user, active_tab='feedback')


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Synchronous prediction — processes the video on the request thread.
    No background jobs, no polling, no threading issues.
    The browser waits ~15-30 s then redirects straight to results."""
    task_type = (request.form.get('task_type') or 'single').strip().lower()
    if task_type not in ('single', 'dual'):
        task_type = 'single'

    video_file = request.files.get('video')
    if not video_file:
        flash("No video file received. Please select a file and try again.")
        return redirect(url_for('dashboard_clinical'))

    video_bytes = video_file.read()
    if not video_bytes:
        flash("Uploaded video is empty. Please upload a valid video file.")
        return redirect(url_for('dashboard_clinical'))

    file_name = video_file.filename or "uploaded_video.mp4"
    print(f"[Predict] Starting sync processing: {file_name} ({len(video_bytes)//1024} KB), task={task_type}")

    try:
        video_sha256 = hashlib.sha256(video_bytes).hexdigest()
        cache_key = f"{video_sha256}_{task_type}"
        video_url = _build_video_url(video_bytes, video_sha256, file_name)

        # 1. Memory cache (fastest)
        with _prediction_cache_lock:
            cached = _prediction_cache.get(cache_key)

        # 2. Firebase persistent cache (survives restarts/redeploys)
        if not cached:
            try:
                fb_key = cache_key[:40].replace('.', '_')
                snap = db.reference(f'video_cache/{fb_key}').get()
                if snap and snap.get('prediction'):
                    cached = snap
                    print(f"[Predict] Served from Firebase cache: {fb_key}")
                    with _prediction_cache_lock:
                        _prediction_cache[cache_key] = cached
            except Exception:
                pass

        # 3. Disk cache (local fallback)
        if not cached:
            cached = _load_cache(cache_key)
            if cached:
                with _prediction_cache_lock:
                    _prediction_cache[cache_key] = cached

        if cached:
            return render_template(
                'results.html',
                prediction=cached["prediction"],
                task_type=task_type,
                debug_frames=cached.get("debug_frames", []),
                video_url=video_url,
            )

        # Not cached — compute now
        os.makedirs('uploads', exist_ok=True)
        upload_filename = _safe_upload_filename(file_name, video_sha256)
        video_path = os.path.join('uploads', upload_filename)
        with open(video_path, 'wb') as f:
            f.write(video_bytes)

        print(f"[Predict] Extracting frames from {video_path}")
        processed_frames, walking_quality, _ = process_video(video_path, file_hash_hex=video_sha256)
        print(f"[Predict] Frames done. walking_quality overall={walking_quality.get('overall')}")

        debug_frames = save_debug_frames(processed_frames, video_hash=video_sha256)

        model = load_dementia_model()
        prediction = predict_risk(model, processed_frames, task_type, walking_quality)
        print(f"[Predict] Prediction: level={prediction.get('level')}, score={prediction.get('score')}")

        result_data = {"prediction": prediction, "debug_frames": debug_frames}

        # Store in all caches
        with _prediction_cache_lock:
            _prediction_cache[cache_key] = result_data
        _save_cache(cache_key, result_data)

        # Fire-and-forget: save to Firebase cache + user results log
        def _fb_write(uid, tt, pred, rdata, ckey):
            try:
                fb_key = ckey[:40].replace('.', '_')
                db.reference(f'video_cache/{fb_key}').set(rdata)
            except Exception:
                pass
            try:
                db.reference(f'results/{uid}').push({
                    'task_type': tt,
                    'risk_score': pred['score'],
                    'risk_level': pred['level'],
                    'timestamp': str(np.datetime64('now')),
                })
            except Exception:
                pass
        threading.Thread(
            target=_fb_write,
            args=(current_user.id, task_type, prediction, result_data, cache_key),
            daemon=True
        ).start()

        print(f"[Predict] Done — rendering results.")
        return render_template(
            'results.html',
            prediction=prediction,
            task_type=task_type,
            debug_frames=debug_frames,
            video_url=video_url,
        )

    except Exception as exc:
        print(f"[Predict] ERROR: {exc}")
        traceback.print_exc()
        # Return error directly — don't silently redirect to homepage
        return render_template('error.html', error=str(exc)), 500


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
    # use_reloader=False is critical — the reloader spawns a child process and
    # kills background threads (video processing jobs), causing silent failures.
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)
