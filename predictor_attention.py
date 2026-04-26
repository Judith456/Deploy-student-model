# predictor_attention.py

import cv2
import mediapipe as mp
import numpy as np
import keras
import os
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode)
from keras.models import load_model

# ============================================================
# CONSTANTS
# ============================================================
LABELS   = ["Poor", "Average", "Good"]
EMOJIS   = ["🔴", "🟡", "🟢"]

FEATURES = [
    "balance", "posture", "confidence",
    "speed_consistency", "pedal_symmetry",
    "upper_body_stability", "pedal_rhythm"
]

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "student_attention.keras")
POSE_PATH  = os.path.join(BASE_DIR, "models", "pose_landmarker.task")

# ============================================================
# LOAD MODEL
# ============================================================
# ============================================================
# LOAD MODEL
# ============================================================
from keras.layers import MultiHeadAttention as _MHA

class _CompatMHA(_MHA):
    def __init__(self, **kwargs):
        kwargs.pop('seed', None)
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, config):
        config.pop('seed', None)
        return cls(**config)

print("Loading Attention Student...")
attention_model = load_model(
    MODEL_PATH,
    custom_objects={'MultiHeadAttention': _CompatMHA}
)
print(f"✅ Loaded — {attention_model.count_params():,} params")

# ============================================================
# LOAD MEDIAPIPE
# ============================================================
options = PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=POSE_PATH),
    running_mode=VisionTaskRunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.3,
    min_pose_presence_confidence=0.3
)
landmarker = PoseLandmarker.create_from_options(options)
print("✅ MediaPipe loaded")

# ============================================================
# FEATURE FUNCTIONS
# ============================================================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def normalize_landmarks(lm):
    scale = abs(lm[11].y - lm[23].y) + 1e-6

    class LM:
        def __init__(self, x, y, z, v):
            self.x = x
            self.y = y
            self.z = z
            self.visibility = v

    return [LM(l.x / scale, l.y / scale, l.z, l.presence) for l in lm]


def extract_features(lm, history=None):
    balance = float(np.clip(1 - abs(lm[23].x - lm[24].x) * 3, 0, 1))

    sh_x = (lm[11].x + lm[12].x) / 2
    hip_x = (lm[23].x + lm[24].x) / 2
    sh_y = (lm[11].y + lm[12].y) / 2
    hip_y = (lm[23].y + lm[24].y) / 2
    tlen = abs(hip_y - sh_y) + 1e-6
    posture = float(np.clip(1 - abs(sh_x - hip_x) / tlen, 0, 1))

    confidence = float(np.mean([l.visibility for l in lm]))

    pts = [
        lm[15].x, lm[16].x, lm[27].x, lm[28].x,
        lm[15].y, lm[16].y, lm[27].y, lm[28].y
    ]
    speed = float(np.clip(1 - np.std(pts) * 2, 0, 1))

    la = calculate_angle(
        [lm[23].x, lm[23].y],
        [lm[25].x, lm[25].y],
        [lm[27].x, lm[27].y]
    )
    ra = calculate_angle(
        [lm[24].x, lm[24].y],
        [lm[26].x, lm[26].y],
        [lm[28].x, lm[28].y]
    )
    symmetry = float(np.clip(1 - abs(la - ra) / 180, 0, 1))

    sc = (lm[11].x + lm[12].x) / 2
    hc = (lm[23].x + lm[24].x) / 2
    stab = float(np.clip(1 - abs(sc - hc) * 4, 0, 1))

    if history and len(history) >= 5:
        la_l, ra_l = [], []
        for p in history:
            la_l.append(calculate_angle(
                [p[23].x, p[23].y],
                [p[25].x, p[25].y],
                [p[27].x, p[27].y]))
            ra_l.append(calculate_angle(
                [p[24].x, p[24].y],
                [p[26].x, p[26].y],
                [p[28].x, p[28].y]))
        rhythm = float(np.clip(
            1 - (np.std(np.diff(la_l)) + np.std(np.diff(ra_l))) / 180,
            0, 1))
    else:
        rhythm = 0.5

    return [
        balance, posture, confidence,
        speed, symmetry, stab, rhythm
    ]


def smooth_features(arr, window=5):
    out = []
    for i in range(arr.shape[1]):
        out.append(np.convolve(arr[:, i],
                               np.ones(window) / window,
                               mode='valid'))
    return np.array(out).T


def create_sequences(feats, seq_len=16):
    return np.array([
        feats[i:i + seq_len]
        for i in range(len(feats) - seq_len)
    ])

# ============================================================
# COACHING + OVERRIDES
# ============================================================
TIP_RULES = [
    (0, 0.35, 0.55, "Keep both hips level"),
    (1, 0.35, 0.55, "Reduce lateral lean"),
    (3, 0.40, 0.45, "Maintain steady movement"),
    (4, 0.55, 0.70, "Equal effort from both legs"),
    (5, 0.40, 0.55, "Engage your core"),
    (6, 0.50, 0.60, "Smooth cadence"),
]

def build_coaching_tips(feat_means, class_idx, confidence):
    tips = []
    for idx, crit, minor, tip in TIP_RULES:
        val = feat_means[idx]
        if val < crit:
            tips.append(tip)
        elif val < minor and confidence < 88:
            tips.append(tip)
    return tips[:4]


def apply_hard_overrides(class_idx, feat_means, tips):
    if feat_means[0] < 0.28:
        if class_idx == 1:
            class_idx = 0
        priority = "Critical: severe balance issue"
        if priority not in tips:
            tips.insert(0, priority)
    return class_idx, tips[:4]

# ============================================================
# RELIABILITY
# ============================================================
def score_reliability(result):
    if not result.get("success"):
        return "failed"

    issues = 0
    if result["detection_rate"] < 0.4:
        issues += 1
    if result["sequences_analysed"] < 30:
        issues += 1
    if result["confidence"] < 70:
        issues += 1

    if issues == 0:
        return "high"
    elif issues == 1:
        return "medium"
    else:
        return "low"

# ============================================================
# FALLBACK (sampled)
# ============================================================
def _predict_sampled(video_path, dur, total, fps):
    cap = cv2.VideoCapture(video_path)
    feats, history = [], []
    frame_n = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_n % 3 != 0:
            frame_n += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb)

        result = landmarker.detect(mp_img)

        if result.pose_landmarks:
            lm = normalize_landmarks(result.pose_landmarks[0])
            history.append(lm)
            if len(history) > 10:
                history.pop(0)
            feats.append(extract_features(lm, history))

        frame_n += 1

    cap.release()

    if len(feats) < 20:
        return {"success": False, "error": "Pose undetectable"}

    feats = np.array(feats)
    seqs = create_sequences(feats)

    if len(seqs) == 0:
        return {"success": False, "error": "Not enough sequences"}

    preds = attention_model.predict(seqs, verbose=0)
    avg = preds.mean(axis=0)
    cls = int(np.argmax(avg))
    means = feats.mean(axis=0)

    result = {
        "success": True,
        "prediction": LABELS[cls],
        "emoji": EMOJIS[cls],
        "confidence": round(float(avg[cls]) * 100, 1),
        "features": {
            FEATURES[i]: round(float(means[i]), 3)
            for i in range(7)
        },
        "coaching_tips": [],
        "detection_rate": round(len(feats) / max(total, 1), 2),
        "sequences_analysed": len(seqs),
        "reliability": "low"
    }

    return result

# ============================================================
# MAIN PREDICT
# ============================================================
def predict(video_path, conf_threshold=0.3, retry=True):
    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    dur = total / fps if fps > 0 else 0

    feats, history = [], []
    detected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb)

        result = landmarker.detect(mp_img)

        if result.pose_landmarks:
            lm = normalize_landmarks(result.pose_landmarks[0])
            history.append(lm)

            if len(history) > 10:
                history.pop(0)

            f = extract_features(lm, history)

            if f[2] >= conf_threshold:
                feats.append(f)
                detected += 1

    cap.release()

    detection_rate = detected / max(total, 1)

    # Retry logic
    if len(feats) < 20 and retry:
        return predict(video_path, 0.15, False)

    # Fallback
    if len(feats) < 20 and not retry:
        return _predict_sampled(video_path, dur, total, fps)

    feats = np.array(feats)

    if len(feats) > 5:
        feats = smooth_features(feats)

    seqs = create_sequences(feats)

    if len(seqs) == 0:
        return {"success": False, "error": "Not enough sequences"}

    preds = attention_model.predict(seqs, verbose=0)
    avg = preds.mean(axis=0)
    cls = int(np.argmax(avg))
    means = feats.mean(axis=0)

    confidence = float(avg[cls]) * 100

    tips = build_coaching_tips(means, cls, confidence)
    cls, tips = apply_hard_overrides(cls, means, tips)

    result = {
        "success": True,
        "prediction": LABELS[cls],
        "emoji": EMOJIS[cls],
        "confidence": round(confidence, 1),
        "probabilities": {
            "Poor": round(float(avg[0]) * 100, 1),
            "Average": round(float(avg[1]) * 100, 1),
            "Good": round(float(avg[2]) * 100, 1),
        },
        "features": {
            FEATURES[i]: round(float(means[i]), 3)
            for i in range(7)
        },
        "coaching_tips": tips,
        "video_info": {
            "total_frames": total,
            "detected_frames": detected,
            "detection_rate": round(detection_rate, 2),
            "duration_seconds": round(dur, 1),
            "sequences_analysed": len(seqs)
        }
    }

    result["reliability"] = score_reliability({
        "success": True,
        "detection_rate": detection_rate,
        "sequences_analysed": len(seqs),
        "confidence": confidence
    })

    return result