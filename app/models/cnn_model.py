"""
NeuroCare — Gait-Based Dementia Risk Prediction Engine
Pure biomechanical scoring — no TensorFlow needed.
"""

import numpy as np

_RISK_THRESHOLDS = {'low': 65, 'medium': 40}
_DUAL_TASK_PENALTY = 0.82


def load_dementia_model():
    """No TF model needed — scoring is purely biomechanical."""
    return None


def predict_risk(model, processed_frames, task_type, walking_quality=None):
    """
    Comprehensive dementia risk scoring from gait biomechanics.

    Combines:
      • Primary  — multi-metric walking quality score (MediaPipe landmarks)
      • Penalty  — dual-task cognitive load correction
      • Bonus    — detection confidence weighting

    walking_quality keys (all 0-100, higher = healthier):
      overall, spine, shoulder, hip, head, step_symmetry,
      balance, trunk_sway, arm_swing, knee_flex, cadence,
      detection_rate, frames_analyzed
    """
    # ── Defaults ─────────────────────────────────────────────────────────────
    if walking_quality is None:
        walking_quality = {
            'overall': 50.0, 'spine': 50.0, 'shoulder': 50.0,
            'hip': 50.0, 'head': 50.0, 'step_symmetry': 50.0,
            'balance': 50.0, 'trunk_sway': 50.0, 'arm_swing': 50.0,
            'knee_flex': 50.0, 'cadence': 50.0,
            'detection_rate': 0.0, 'frames_analyzed': 0,
        }

    # ── Pull metrics ──────────────────────────────────────────────────────────
    overall       = float(walking_quality.get('overall', 50.0))
    spine         = float(walking_quality.get('spine', 50.0))
    hip           = float(walking_quality.get('hip', 50.0))
    balance       = float(walking_quality.get('balance', 50.0))
    step_sym      = float(walking_quality.get('step_symmetry', 50.0))
    shoulder      = float(walking_quality.get('shoulder', 50.0))
    knee          = float(walking_quality.get('knee_flex', 50.0))
    trunk         = float(walking_quality.get('trunk_sway', 50.0))
    head          = float(walking_quality.get('head', 50.0))
    cadence       = float(walking_quality.get('cadence', 50.0))
    detect_rate   = float(walking_quality.get('detection_rate', 0.0))
    frames_used   = int(walking_quality.get('frames_analyzed', 0))

    # ── Composite quality score (clinically weighted) ─────────────────────────
    composite = (
        overall  * 0.30 +
        spine    * 0.15 +
        hip      * 0.15 +
        balance  * 0.14 +
        step_sym * 0.10 +
        shoulder * 0.06 +
        knee     * 0.05 +
        trunk    * 0.03 +
        head     * 0.01 +
        cadence  * 0.01
    )

    # ── Dual-task cognitive penalty ───────────────────────────────────────────
    if task_type == 'dual':
        composite *= _DUAL_TASK_PENALTY

    # ── Detection confidence boost / penalty ──────────────────────────────────
    # High detection rate = more reliable; low = noisier estimate
    if detect_rate >= 80:
        confidence_adj = +1.5
    elif detect_rate >= 50:
        confidence_adj = 0.0
    else:
        confidence_adj = -2.5   # penalise noisy estimates slightly

    composite = max(0.0, min(100.0, composite + confidence_adj))

    # ── Risk score (inverse of walking quality) ───────────────────────────────
    risk_score = round(100.0 - composite, 1)

    # ── Risk classification ───────────────────────────────────────────────────
    walking_score = round(composite, 1)

    if walking_score >= _RISK_THRESHOLDS['low']:
        level          = "Low Risk"
        color          = "success"
        walking_label  = "Good Walking"
        confidence     = min(98, int(60 + walking_score * 0.38))
    elif walking_score >= _RISK_THRESHOLDS['medium']:
        level          = "Medium Risk"
        color          = "warning"
        walking_label  = "Irregular Walking"
        confidence     = min(95, int(55 + walking_score * 0.4))
    else:
        level          = "High Risk"
        color          = "danger"
        walking_label  = "Poor Walking"
        confidence     = min(96, int(50 + (100 - walking_score) * 0.45))

    # ── Clinical recommendations ──────────────────────────────────────────────
    if color == 'success':
        recommendation = (
            "Gait analysis indicates normal walking patterns with low markers for "
            "cognitive impairment. Posture metrics are within expected healthy ranges. "
            "Routine monitoring is recommended; no immediate clinical action required."
        )
    elif color == 'warning':
        recommendation = (
            "Gait analysis reveals mild irregularities that may correspond to early-stage "
            "cognitive changes. Some postural deviations detected. A follow-up clinical "
            "assessment is advisable within the next 4–8 weeks."
        )
    else:
        recommendation = (
            "Gait analysis shows significant postural and symmetry deviations associated "
            "with elevated cognitive risk. Immediate referral to a neurologist or geriatric "
            "specialist is strongly recommended for comprehensive evaluation."
        )

    return {
        'score':          risk_score,
        'walking_score':  walking_score,
        'level':          level,
        'color':          color,
        'walking_label':  walking_label,
        'confidence':     confidence,
        'task_type':      task_type,
        'recommendation': recommendation,
        'metrics':        walking_quality,
        'frames_analyzed': frames_used,
        'detection_rate': detect_rate,
    }
