import os, time, math, requests
import cv2
import numpy as np
from dotenv import load_dotenv
from loguru import logger


# InsightFace for detection + 5 keypoints (eyes, nose, mouth corners)
from insightface.app import FaceAnalysis


load_dotenv()
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
SESSION_ID = os.getenv("SESSION_ID", "demo-101")
POST_URL = f"{API_BASE}/v1/sessions/{SESSION_ID}/signals"


# 3D model points corresponding to: nose tip, left eye outer, right eye outer, left mouth corner, right mouth corner
MODEL_POINTS_3D = np.array([
    [0.0, 0.0, 0.0], # nose tip
    [-30.0, 32.0, -30.0], # left eye outer
    [30.0, 32.0, -30.0], # right eye outer
    [-25.0, -28.0, -24.0], # left mouth corner
    [25.0, -28.0, -24.0], # right mouth corner
], dtype=np.float64)


# indices for InsightFace 5-point kps: [left_eye, right_eye, nose, left_mouth, right_mouth]
KPS_IDX = {"left_eye":0, "right_eye":1, "nose":2, "left_mouth":3, "right_mouth":4}




def head_pose_from_5kps(kps, w, h):
# Map to 2D points in image coordinates with same order as MODEL_POINTS_3D
    nose = kps[KPS_IDX["nose"]]
    le = kps[KPS_IDX["left_eye"]]
    re = kps[KPS_IDX["right_eye"]]
    lm = kps[KPS_IDX["left_mouth"]]
    rm = kps[KPS_IDX["right_mouth"]]

    points_2d = np.array([
        [nose[0], nose[1]],
        [le[0], le[1]],
        [re[0], re[1]],
        [lm[0], lm[1]],
        [rm[0], rm[1]],
    ], dtype=np.float64)

    focal_length = w
    center = (w/2, h/2)
    cam_matrix = np.array([[focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4,1))

    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS_3D, points_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None
    rmat, _ = cv2.Rodrigues(rvec)
    yaw = math.degrees(math.atan2(rmat[1,0], rmat[0,0]))
    pitch = math.degrees(math.atan2(-rmat[2,0], math.sqrt(rmat[2,1]**2 + rmat[2,2]**2)))
    return float(yaw), float(pitch)

def gaze_to_board_prob(yaw_deg, pitch_deg):
    if yaw_deg is None or pitch_deg is None:
        return 0.0
    yaw_score = max(0.0, 1.0 - abs(yaw_deg)/30.0)
    pitch_score = max(0.0, 1.0 - abs(pitch_deg)/35.0)
    return max(0.0, min(1.0, 0.6*yaw_score + 0.4*pitch_score))

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open camera")
        return

    app = FaceAnalysis(name=None, providers=["CPUExecutionProvider"], allowed_modules=["detection"]) # 5-point kps available on detection
    app.prepare(ctx_id=0, det_size=(640, 640))

    batch = []
    last_post = 0.0

    while True:
        ok, frame = cap.read()
        if not ok: break
        h, w = frame.shape[:2]

        faces = app.get(frame)
        for i, f in enumerate(faces or []):
            if f.kps is None or len(f.kps) < 5:
                continue
            yaw, pitch = head_pose_from_5kps(f.kps, w, h)
            att = gaze_to_board_prob(yaw, pitch)
            batch.append({
                "ts": time.time(),
                "zone_id": f"row{i//6}_col{i%6}",
                "head_pose_yaw": yaw,
                "head_pose_pitch": pitch,
                "gaze_to_board_prob": att,
            })
            # draw for viz (optional)
            for (x, y) in f.kps:
                cv2.circle(frame, (int(x), int(y)), 2, (255,255,255), -1)

        cv2.putText(frame, f"batch={len(batch)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Edge Agent (InsightFace)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        now = time.time()
        if (now - last_post) > 0.7 and batch:
            try:
                payload = {"session_id": SESSION_ID, "batch": batch}
                r = requests.post(POST_URL, json=payload, timeout=2)
                if r.ok:
                    logger.info(f"posted {len(batch)} records")
                    batch = []
                    last_post = now
            except Exception as e:
                logger.warning(f"post failed: {e}")
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()