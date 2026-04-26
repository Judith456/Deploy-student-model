# predictor_attention.py
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions)
from mediapipe.tasks.python.vision.core\
    .vision_task_running_mode import (
    VisionTaskRunningMode)

LABELS   = ["Poor", "Average", "Good"]
EMOJIS   = ["🔴", "🟡", "🟢"]
FEATURES = [
    "balance", "posture", "confidence",
    "speed_consistency", "pedal_symmetry",
    "upper_body_stability", "pedal_rhythm"
]

BASE_DIR    = os.path.dirname(
    os.path.abspath(__file__))
MODEL_PATH  = os.path.join(
    BASE_DIR, "models",
    "student_attention.keras")
POSE_PATH   = os.path.join(
    BASE_DIR, "models",
    "pose_landmarker.task")

print("Loading Attention Student...")
attention_model = tf.keras.models.load_model(
    MODEL_PATH)
print(f"✅ Loaded — "
      f"{attention_model.count_params():,} params")

options    = PoseLandmarkerOptions(
    base_options = python.BaseOptions(
        model_asset_path=POSE_PATH),
    running_mode = VisionTaskRunningMode.IMAGE,
    num_poses    = 1,
    min_pose_detection_confidence = 0.3,
    min_pose_presence_confidence  = 0.3
)
landmarker = PoseLandmarker\
    .create_from_options(options)
print("✅ MediaPipe loaded")


def calculate_angle(a, b, c):
    a,b,c = np.array(a),np.array(b),np.array(c)
    ba    = a-b
    bc    = c-b
    cos   = np.dot(ba,bc)/(
        np.linalg.norm(ba)*
        np.linalg.norm(bc)+1e-6)
    return np.degrees(
        np.arccos(np.clip(cos,-1,1)))


def normalize_landmarks(lm):
    scale = abs(lm[11].y-lm[23].y)+1e-6
    class LM:
        def __init__(self,x,y,z,v):
            self.x=x; self.y=y
            self.z=z; self.visibility=v
    return [LM(l.x/scale,l.y/scale,
               l.z,l.presence) for l in lm]


def extract_features(lm, history=None):
    balance  = float(np.clip(
        1-abs(lm[23].x-lm[24].x)*3,0,1))
    sh_x=(lm[11].x+lm[12].x)/2
    hx  =(lm[23].x+lm[24].x)/2
    sh_y=(lm[11].y+lm[12].y)/2
    hy  =(lm[23].y+lm[24].y)/2
    tlen=abs(hy-sh_y)+1e-6
    posture = float(np.clip(
        1-abs(sh_x-hx)/tlen,0,1))
    conf    = float(np.mean(
        [l.visibility for l in lm]))
    pts=[lm[15].x,lm[16].x,
         lm[27].x,lm[28].x,
         lm[15].y,lm[16].y,
         lm[27].y,lm[28].y]
    speed   = float(np.clip(
        1-np.std(pts)*2,0,1))
    la=calculate_angle(
        [lm[23].x,lm[23].y],
        [lm[25].x,lm[25].y],
        [lm[27].x,lm[27].y])
    ra=calculate_angle(
        [lm[24].x,lm[24].y],
        [lm[26].x,lm[26].y],
        [lm[28].x,lm[28].y])
    symm    = float(np.clip(
        1-abs(la-ra)/180,0,1))
    sc=(lm[11].x+lm[12].x)/2
    hc=(lm[23].x+lm[24].x)/2
    stab    = float(np.clip(
        1-abs(sc-hc)*4,0,1))
    if history and len(history)>=5:
        ll,rl=[],[]
        for p in history:
            ll.append(calculate_angle(
                [p[23].x,p[23].y],
                [p[25].x,p[25].y],
                [p[27].x,p[27].y]))
            rl.append(calculate_angle(
                [p[24].x,p[24].y],
                [p[26].x,p[26].y],
                [p[28].x,p[28].y]))
        rhythm=float(np.clip(
            1-(np.std(np.diff(ll))+
               np.std(np.diff(rl)))/180,
            0,1))
    else:
        rhythm=0.5
    return [balance,posture,conf,
            speed,symm,stab,rhythm]


def smooth_features(arr, w=5):
    out=[]
    for i in range(arr.shape[1]):
        out.append(np.convolve(
            arr[:,i],
            np.ones(w)/w,
            mode='valid'))
    return np.array(out).T


def create_sequences(f, n=16):
    return np.array([
        f[i:i+n]
        for i in range(len(f)-n)])


def predict(video_path: str) -> dict:
    cap     = cv2.VideoCapture(video_path)
    total   = int(cap.get(
        cv2.CAP_PROP_FRAME_COUNT))
    fps     = cap.get(cv2.CAP_PROP_FPS)
    dur     = total/fps if fps>0 else 0
    feats   = []
    history = []
    detected= 0

    while True:
        ret,frame = cap.read()
        if not ret: break
        rgb    = cv2.cvtColor(
            frame,cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb)
        res    = landmarker.detect(mp_img)
        if res.pose_landmarks and \
                len(res.pose_landmarks)>0:
            lm = normalize_landmarks(
                res.pose_landmarks[0])
            history.append(lm)
            if len(history)>10:
                history.pop(0)
            f = extract_features(lm,history)
            if f[2]>=0.3:
                feats.append(f)
                detected+=1
    cap.release()

    if len(feats)<20:
        return {
            "success": False,
            "error":   "Pose not detected "
                       "in enough frames"}

    feats = np.array(feats)
    if len(feats)>5:
        feats = smooth_features(feats)
    seqs  = create_sequences(feats)
    if len(seqs)==0:
        return {
            "success": False,
            "error":   "Not enough sequences"}

    preds     = attention_model.predict(
        seqs, verbose=0)
    avg       = preds.mean(axis=0)
    cls       = int(np.argmax(avg))
    means     = feats.mean(axis=0)

    tips = []
    thresh = [(0,0.5,"Level your hips"),
              (1,0.5,"Reduce lateral lean"),
              (4,0.7,"Equal leg effort"),
              (5,0.5,"Stabilise upper body"),
              (6,0.6,"Smooth your cadence")]
    for i,t,tip in thresh:
        if means[i]<t:
            tips.append(tip)

    return {
        "success":     True,
        "model":       "Attention Student",
        "prediction":  LABELS[cls],
        "emoji":       EMOJIS[cls],
        "confidence":  round(
            float(avg[cls])*100,1),
        "probabilities":{
            "Poor":   round(
                float(avg[0])*100,1),
            "Average":round(
                float(avg[1])*100,1),
            "Good":   round(
                float(avg[2])*100,1)},
        "features":{
            FEATURES[i]:round(
                float(means[i]),3)
            for i in range(7)},
        "coaching_tips":   tips,
        "video_info":{
            "total_frames":    total,
            "detected_frames": detected,
            "detection_rate":  round(
                detected/max(total,1),2),
            "duration_seconds":round(dur,1),
            "sequences_analysed":len(seqs)}
    }