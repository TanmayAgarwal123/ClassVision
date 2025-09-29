import os, time, math, requests
import cv2
import numpy as np
import mediapipe as mp
from dotenv import load_dotenv


load_dotenv()
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
SESSION_ID = os.getenv("SESSION_ID", "demo-101")
POST_URL = f"{API_BASE}/v1/sessions/{SESSION_ID}/signals"


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


# 3D model points (rough) for head pose from 2D landmarks
MODEL_POINTS_3D = np.array([
[0.0, 0.0, 0.0], # nose tip (landmark 1)
[0.0, -63.6, -12.5], # chin
[-43.3, 32.7, -26.0], # left eye corner
[43.3, 32.7, -26.0], # right eye corner
[-28.9, -28.9, -24.1], # left mouth corner
[28.9, -28.9, -24.1], # right mouth corner
], dtype=np.float64)


# FaceMesh landmark indices: nose=1, chin=152, left eye outer=33, right eye outer=263, mouth left=61, mouth right=291
LMS = [1, 152, 33, 263, 61, 291]


def head_pose_from_landmarks(landmarks, w, h):
points_2d = []
for idx in LMS:
lm = landmarks[idx]
points_2d.append([lm.x * w, lm.y * h])
points_2d = np.array(points_2d, dtype=np.float64)


focal_length = w
center = (w / 2, h / 2)
main()