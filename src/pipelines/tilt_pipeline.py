import cv2
import math
import numpy as np
import json
from pathlib import Path
from collections import deque
from scipy.signal import savgol_filter
import mediapipe as mp

# ---------- 설정 기본값 ----------
CONF_VIS = 0.6
INIT_FRAMES = 30
MED_WIN = 5
SAVGOL_WIN = 11
SAVGOL_POLY = 2
OUTLIER_MAX_DEG = 45.0
MIN_AXIS_PIX = 0.10

mp_pose = mp.solutions.pose

def shoulder_points(lm, w, h):
    L = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    R = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    pL = np.array([L.x * w, L.y * h], dtype=np.float32)
    pR = np.array([R.x * w, R.y * h], dtype=np.float32)
    return pL, pR

def shoulder_center(lm, w, h):
    pL, pR = shoulder_points(lm, w, h)
    return (pL + pR) / 2.0

def hip_center(lm, w, h):
    L = lm[mp_pose.PoseLandmark.LEFT_HIP]
    R = lm[mp_pose.PoseLandmark.RIGHT_HIP]
    return np.array([ (L.x+R.x)/2 * w, (L.y+R.y)/2 * h ], dtype=np.float32)

def landmarks_visible(lm, conf=CONF_VIS):
    idxs = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]
    return all(lm[i].visibility >= conf for i in idxs)

def body_tilt_signed_deg(neck, pelvis):
    v = neck - pelvis  # (dx, dy), y는 아래로 증가
    norm = np.linalg.norm(v)
    if norm < 1e-6: return None
    cos_th = np.clip(-(v[1]) / norm, -1.0, 1.0)  # 수직 u=(0,-1)과의 각
    ang = math.degrees(math.acos(cos_th))        # 0~180
    sgn = 1.0 if v[0] > 0 else (-1.0 if v[0] < 0 else 0.0)
    return sgn * ang

def moving_median(arr, k=MED_WIN):
    if k < 3 or k % 2 == 0: return np.array(arr, dtype=float)
    half = k // 2
    out = []
    for i in range(len(arr)):
        s = max(0, i-half); e = min(len(arr), i+half+1)
        out.append(float(np.median(arr[s:e])))
    return np.array(out, dtype=float)

def safe_mean_abs(x):
    
    if x.size == 0:
        return 0.0
    return float(np.mean(np.abs(x)))

def process_video(video_path, threshold_deg=3.0,
                  do_median=True, do_savgol=True,
                  cross_scale=0.6, cross_min=30, cross_max_ratio=0.4,
                  thick_mul=0.06, font_mul=0.025):
    video_path = str(Path(video_path).resolve())
    p = Path(video_path)
    out_video = p.with_name(p.stem + "_degree_final" + p.suffix)
    out_json  = p.with_name(p.stem + "_degree_final.json")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(str(out_video),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        enable_segmentation=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    init_signed = deque(maxlen=INIT_FRAMES)
    signed_series = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        tilt_signed = None
        neck = pelvis = None

        if res.pose_landmarks and landmarks_visible(res.pose_landmarks.landmark, CONF_VIS):
            lm = res.pose_landmarks.landmark
            neck = shoulder_center(lm, w, h)
            pelvis = hip_center(lm, w, h)
            # 축 길이 체크
            if np.linalg.norm(neck - pelvis) >= (MIN_AXIS_PIX * h):
                tilt_signed = body_tilt_signed_deg(neck, pelvis)

            # --- 십자가 크기 계산 (어깨폭 기반) ---
            pL, pR = shoulder_points(lm, w, h)
            shoulder_width = float(np.linalg.norm(pR - pL))  # px
            # 반길이 계산
            cross_half = int(max(cross_min,
                                 min(shoulder_width * cross_scale,
                                     h * cross_max_ratio)))
            # 두께/폰트 스케일
            thickness = max(2, int(shoulder_width * thick_mul))
            font_scale = max(0.5, shoulder_width * font_mul)

            # 오버레이: 몸축
            cv2.line(frame, (int(pelvis[0]), int(pelvis[1])),
                            (int(neck[0]),   int(neck[1])), (255, 200, 0), 2)
            # 오버레이: 십자가(어깨 중심)
            cx, cy = int(neck[0]), int(neck[1])
            cv2.line(frame, (cx - cross_half, cy), (cx + cross_half, cy), (0,255,0), thickness, cv2.LINE_AA)
            cv2.line(frame, (cx, cy - cross_half), (cx, cy + cross_half), (0,255,0), thickness, cv2.LINE_AA)

            # 각도 텍스트
            if tilt_signed is not None:
                cv2.putText(frame, f"Tilt: {tilt_signed:+.2f} deg",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255,255,255), max(2, thickness-1), cv2.LINE_AA)

        # 초기 기준축 수집
        if tilt_signed is not None and frame_idx <= INIT_FRAMES:
            init_signed.append(tilt_signed)

        signed_series.append(tilt_signed)
        writer.write(frame)

    cap.release()
    writer.release()
    pose.close()

    # 유효 프레임 추출
    raw = np.array([x for x in signed_series if x is not None], dtype=float)

    # 초기 기준축 보정(오프셋 제거)
    offset = float(np.median(init_signed)) if len(init_signed) >= max(5, INIT_FRAMES//3) else 0.0
    corrected = raw - offset

    # 이상치 제거
    corrected = corrected[np.abs(corrected) <= OUTLIER_MAX_DEG]

    # 노이즈 필터
    if do_median and len(corrected) >= 3:
        corrected = moving_median(corrected, MED_WIN)
    if do_savgol and len(corrected) >= SAVGOL_WIN:
        corrected = savgol_filter(corrected, SAVGOL_WIN, SAVGOL_POLY)

    # ----- 통계 (좌/우 별 추가) -----
    if len(corrected) > 0:
        avg_abs = float(np.mean(np.abs(corrected)))
        avg_signed = float(np.mean(corrected))
        used_count = int(len(corrected))

        left_vals  = corrected[corrected < 0.0]
        right_vals = corrected[corrected > 0.0]

        avg_left_abs  = safe_mean_abs(left_vals)
        avg_right_abs = safe_mean_abs(right_vals)

        left_ratio  = float(left_vals.size)  / used_count
        right_ratio = float(right_vals.size) / used_count

        if avg_signed > 0.05:
            dominant = "right"
        elif avg_signed < -0.05:
            dominant = "left"
        else:
            dominant = "balanced"
    else:
        avg_abs = 0.0
        avg_signed = 0.0
        used_count = 0
        avg_left_abs = 0.0
        avg_right_abs = 0.0
        left_ratio = 0.0
        right_ratio = 0.0
        dominant = "balanced"

    payload = {
        "average_abs_tilt_deg": round(avg_abs, 2),
        "average_signed_tilt_deg": round(avg_signed, 2),
        "avg_left_abs_tilt_deg": round(avg_left_abs, 2),
        "avg_right_abs_tilt_deg": round(avg_right_abs, 2),
        "left_ratio": round(left_ratio, 3),     # 0~1
        "right_ratio": round(right_ratio, 3),   # 0~1
        "dominant_side": dominant,              # left/right/balanced
        "used_frame_count": used_count,
        "total_frame_count": frame_idx,
        "threshold_deg": threshold_deg,
        "possible_parkinson": (avg_abs >= threshold_deg),
        "offset_deg": round(offset, 2)          # 디버깅용: 초기 오프셋
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 결과 영상 저장: {out_video}")
    print(f"[INFO] JSON 저장: {out_json}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    return payload
