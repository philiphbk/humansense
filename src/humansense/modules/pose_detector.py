"""
Stage 4 — Pose Analysis Module
Uses MediaPipe Tasks API (PoseLandmarker) — compatible with mediapipe >= 0.10.30.
Detects:
  - Raised hand (gesture of attention/need)
  - Fall risk (unusual body angle)
  - Protective/defensive posture (arms crossed, hunched)
  - Standing vs. seated
"""

import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from ..models import ensure_model
from ..utils.tracking import Debouncer

# ── Model path ───────────────────────────────────────────────────────────────
POSE_MODEL_NAME = "pose_landmarker_lite.task"

# ── Pose landmark indices (unchanged from legacy API) ────────────────────────
_NOSE          = 0
_L_SHOULDER    = 11
_R_SHOULDER    = 12
_L_ELBOW       = 13
_R_ELBOW       = 14
_L_WRIST       = 15
_R_WRIST       = 16
_L_HIP         = 23
_R_HIP         = 24
_L_KNEE        = 25
_R_KNEE        = 26
_L_ANKLE       = 27
_R_ANKLE       = 28

# ── Skeleton connections for manual drawing ──────────────────────────────────
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]

# Singleton landmarker
_landmarker = None

# A cue must persist for 2 processed frames and clear for 3 to stop one-frame flicker.
_debouncer = Debouncer(on_frames=2, off_frames=3)


def reset_state():
    """Call when a new video / camera session starts."""
    _debouncer.reset()


def get_landmarker():
    global _landmarker
    if _landmarker is None:
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model(POSE_MODEL_NAME)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    return _landmarker


def analyze_pose(frame: np.ndarray, detections=None, calibration=None, draw: bool = True):
    """
    Analyze the pose of every detected person.

    `detections` is the crowd detector's list of {"bbox", "id", ...}. Each person is
    cropped and analysed on their own, so cues are attributed to the right individual.
    With no detections the whole frame is analysed as a single person (legacy mode).

    Returns a dict with annotated_frame, cues (list of strings, tagged "[#id]" when
    several people are present), pose_data and, if the model is missing, "error".
    """
    try:
        landmarker = get_landmarker()
    except (OSError, RuntimeError) as e:
        annotated = frame.copy() if draw else None
        if draw:
            cv2.putText(annotated, "Pose model unavailable — run: humansense download-models",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1, cv2.LINE_AA)
        return {"annotated_frame": annotated, "cues": [], "pose_data": {}, "error": str(e)}

    h, w = frame.shape[:2]
    annotated = frame.copy() if draw else None

    if detections:
        people = [(d.get("id", i), d["bbox"]) for i, d in enumerate(detections)]
    else:
        people = [(0, (0, 0, w, h))]
    multi = len(people) > 1

    found = []          # (pid, cue, draw_fn)
    pose_data = {}
    skeletons = []

    for pid, (x1, y1, x2, y2) in people:
        # Pad the crop so limbs just outside the box are still included
        pw, ph = int((x2 - x1) * 0.15), int((y2 - y1) * 0.10)
        cx1, cy1 = max(x1 - pw, 0), max(y1 - ph, 0)
        cx2, cy2 = min(x2 + pw, w), min(y2 + ph, h)
        if cx2 - cx1 < 20 or cy2 - cy1 < 20:
            continue
        crop = np.ascontiguousarray(cv2.cvtColor(frame[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB))
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
        if not result.pose_landmarks:
            continue

        cw, ch = cx2 - cx1, cy2 - cy1
        # Landmarks in full-frame pixels: (x, y, visibility, z)
        pts = [(cx1 + p.x * cw, cy1 + p.y * ch, getattr(p, "visibility", 1.0), p.z * cw)
               for p in result.pose_landmarks[0]]
        skeletons.append(pts)
        in_bed_zone = (calibration is not None and
                       calibration.in_no_fall_zone(((x1 + x2) / 2, y2), w, h))
        for cue, fn in _person_cues(pts, (x1, y1, x2, y2)):
            # Beds / gurney bays: lying down there is expected, not a fall
            if in_bed_zone and cue.upper().startswith("FALL"):
                continue
            found.append((pid, cue, fn))
        if pid == people[0][0] or not pose_data:
            pose_data = _pose_summary(pts)

    confirmed = _debouncer.update({(pid, cue) for pid, cue, _ in found})
    cues, records = [], []
    for pid, cue, fn in found:
        if (pid, cue) not in confirmed:
            continue
        if draw:
            fn(annotated)
        cues.append(f"{cue} [#{pid}]" if multi else cue)
        records.append({"id": pid, "cue": cue})

    # Fall alerts first so they win the limited badge space
    cues.sort(key=lambda c: "FALL" not in c.upper())
    if draw:
        for pts in skeletons:
            _draw_skeleton(annotated, pts)
        _draw_pose_cues_badge(annotated, cues)

    return {"annotated_frame": annotated, "cues": cues, "cue_records": records,
            "pose_data": pose_data}


def _pose_summary(pts):
    def p(i): return (int(pts[i][0]), int(pts[i][1]))
    return {
        "nose": p(_NOSE),
        "shoulder_mid": ((p(_L_SHOULDER)[0] + p(_R_SHOULDER)[0]) // 2,
                         (p(_L_SHOULDER)[1] + p(_R_SHOULDER)[1]) // 2),
        "hip_mid": ((p(_L_HIP)[0] + p(_R_HIP)[0]) // 2,
                    (p(_L_HIP)[1] + p(_R_HIP)[1]) // 2),
        "left_wrist": p(_L_WRIST),
        "right_wrist": p(_R_WRIST),
    }


def _person_cues(pts, bbox):
    """
    Rule-based cues for one person. All thresholds are relative to that person's own
    torso length, so they do not change with resolution or distance from the camera.
    Returns a list of (cue_text, draw_fn(frame)).
    """
    def xy(i): return (pts[i][0], pts[i][1])
    def vis(i): return pts[i][2]

    cues = []
    bx1, by1, bx2, by2 = bbox
    box_w, box_h = max(bx2 - bx1, 1), max(by2 - by1, 1)

    shoulders_ok = vis(_L_SHOULDER) > 0.5 and vis(_R_SHOULDER) > 0.5
    hips_ok = vis(_L_HIP) > 0.4 and vis(_R_HIP) > 0.4
    ls, rs = xy(_L_SHOULDER), xy(_R_SHOULDER)
    lh, rh = xy(_L_HIP), xy(_R_HIP)
    lw, rw = xy(_L_WRIST), xy(_R_WRIST)

    sh_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    hip_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    sh_width = abs(ls[0] - rs[0])

    torso_len = None
    tilt = None                 # torso angle from vertical, 0 = upright, 90 = horizontal
    if shoulders_ok and hips_ok:
        dx, dy = hip_mid[0] - sh_mid[0], hip_mid[1] - sh_mid[1]
        torso_len = math.hypot(dx, dy)
        if torso_len > 1e-6:
            tilt = math.degrees(math.atan2(abs(dx), abs(dy)))
    # Reference length for everything below; falls back to a fraction of the box height
    ref = torso_len if torso_len and torso_len > 10 else box_h * 0.3

    # Facing the camera? Turned sideways, shoulders overlap in x and differ a lot in depth.
    frontal = not (shoulders_ok and abs(pts[_L_SHOULDER][3] - pts[_R_SHOULDER][3]) > sh_width)

    # ── Fall detection (body orientation, independent of distance/frame size) ────
    if tilt is not None and torso_len > 10:
        wide_box = box_w > 1.2 * box_h
        if tilt > 65 or (tilt > 50 and wide_box):
            cues.append(("FALL detected (horizontal posture)",
                         lambda f: _draw_alert_box(f, "FALL DETECTED", (0, 0, 220))))
        elif tilt > 40:
            cues.append(("Fall risk (leaning severely)",
                         lambda f: _draw_alert_box(f, "FALL RISK", (0, 100, 255))))
    elif wide_box_only(box_w, box_h):
        # Torso not visible (e.g. lying person seen end-on): fall back to the box shape
        cues.append(("Fall risk (lying posture)",
                     lambda f: _draw_alert_box(f, "FALL RISK", (0, 100, 255))))

    upright = tilt is None or tilt < 30

    # ── Raised hand (margin scales with torso length) ───────────────────────────
    margin = 0.15 * ref
    left_up = vis(_L_WRIST) > 0.5 and vis(_L_SHOULDER) > 0.5 and lw[1] < ls[1] - margin
    right_up = vis(_R_WRIST) > 0.5 and vis(_R_SHOULDER) > 0.5 and rw[1] < rs[1] - margin
    if left_up or right_up:
        side = ("both hands" if left_up and right_up
                else ("left hand" if left_up else "right hand"))
        target = (int(lw[0]), int(lw[1])) if left_up else (int(rw[0]), int(rw[1]))
        cues.append((f"Raised hand ({side})",
                     lambda f, t=target: _highlight_point(f, t, (50, 200, 50), "Hand raised!")))

    # ── Protective / hunched posture (only meaningful when facing the camera) ───
    if shoulders_ok and torso_len and frontal and upright:
        if sh_width < torso_len * 0.4 and torso_len > 30:
            cues.append(("Protective/hunched posture", lambda f: None))

    # ── Arms crossed (only meaningful when facing the camera) ───────────────────
    if (frontal and hips_ok and vis(_L_WRIST) > 0.5 and vis(_R_WRIST) > 0.5):
        # Each wrist has crossed the body midline to the side opposite its own shoulder
        mid_x = (sh_mid[0] + hip_mid[0]) / 2
        crossed = ((lw[0] - mid_x) * (ls[0] - mid_x) < 0 and
                   (rw[0] - mid_x) * (rs[0] - mid_x) < 0)
        near = (abs(lw[1] - hip_mid[1]) < ref * 0.8 and abs(rw[1] - hip_mid[1]) < ref * 0.8)
        if crossed and near:
            cues.append(("Arms crossed (defensive/closed posture)", lambda f: None))

    # ── Seated posture (upright torso, knees roughly level with hips) ───────────
    if (hips_ok and upright and vis(_L_KNEE) > 0.4 and vis(_R_KNEE) > 0.4):
        knee_y = (xy(_L_KNEE)[1] + xy(_R_KNEE)[1]) / 2
        if abs(knee_y - hip_mid[1]) < 0.5 * ref:
            cues.append(("Seated posture", lambda f: None))

    return cues


def wide_box_only(box_w, box_h):
    return box_w > 1.6 * box_h


# ── Drawing helpers ──────────────────────────────────────────────────────────

def _draw_skeleton(frame, pts):
    """Draw skeleton lines and joint dots from full-frame (x, y, visibility, z) points."""
    xy = [(int(p[0]), int(p[1])) for p in pts]
    vis = [p[2] for p in pts]

    for a, b in POSE_CONNECTIONS:
        if a < len(xy) and b < len(xy) and vis[a] > 0.3 and vis[b] > 0.3:
            cv2.line(frame, xy[a], xy[b], (0, 200, 100), 2, cv2.LINE_AA)

    for pt, v in zip(xy, vis):
        if v > 0.3:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1)
            cv2.circle(frame, pt, 4, (0, 180, 80), 1)


def _highlight_point(frame, point, color, label):
    cv2.circle(frame, point, 12, color, -1)
    cv2.circle(frame, point, 12, (255, 255, 255), 2)
    cv2.putText(frame, label, (point[0] + 14, point[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_alert_box(frame, text, color):
    h, w  = frame.shape[:2]
    cx, cy = w // 2, int(h * 0.1)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(frame,
                  (cx - tw // 2 - 12, cy - th - 8),
                  (cx + tw // 2 + 12, cy + 8), color, -1)
    cv2.putText(frame, text, (cx - tw // 2, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_pose_cues_badge(frame, cues):
    y_off = 72
    if not cues:
        _badge(frame, "  Pose: Normal  ", (50, 200, 50), y_off)
        return
    for cue in cues[:3]:
        color = (0, 0, 220) if "FALL" in cue.upper() else (0, 140, 255)
        _badge(frame, f"  {cue}  ", color, y_off)
        (_, th), _ = cv2.getTextSize(cue, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y_off += th + 14


def _badge(frame, text, color, y_off):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(frame, (8, y_off), (8 + tw + 4, y_off + th + 10), (20, 20, 20), -1)
    cv2.putText(frame, text, (10, y_off + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
