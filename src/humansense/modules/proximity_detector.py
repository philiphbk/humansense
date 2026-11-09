"""
Stage 3 — Proximity & Personal Space Module
Detects when people are too close to each other (or to a simulated robot position).
Uses person-height-normalised foot-point distances since we don't have depth cameras.
"""


import cv2
import numpy as np

from ..utils.tracking import Debouncer

# Personal space as a multiple of the (mean) person's on-screen height. Scaling by
# body height makes the threshold follow perspective: far-away people are small, so
# their pixel distances count for less than those of people near the camera.
PERSONAL_SPACE_RATIO = 0.8        # ~0.8 body heights ≈ 1.3 m for an adult
ROBOT_SPACE_RATIO    = 1.2        # robots get a larger bubble (1.5x)
# Two people whose heights differ by more than this are at very different depths
# even if their pixels line up, so they are not treated as close.
MAX_DEPTH_RATIO      = 1.8
# When a camera calibration (floor homography) is available, distances are real metres
PERSONAL_SPACE_M     = 1.2
ROBOT_SPACE_M        = 1.8

# A violation must persist for 2 processed frames and clear for 3 to stop flicker.
_debouncer = Debouncer(on_frames=2, off_frames=3)


def reset_state():
    """Call when a new video / camera session starts."""
    _debouncer.reset()


def _foot(det):
    """Ground-contact proxy: bottom-centre of the box (better than the box centre)."""
    x1, _y1, x2, y2 = det["bbox"]
    return ((x1 + x2) // 2, y2)


def _height(det):
    return max(det["bbox"][3] - det["bbox"][1], 1)


def check_proximity(frame: np.ndarray, detections: list[dict], robot_position: tuple | None = None,
                    calibration=None, draw: bool = True):
    """
    Given a list of person detections (with optional tracker 'id'), check:
      1. Person-to-person proximity violations
      2. Person-to-robot proximity violations (if robot_position is given)

    Returns a dict with annotated_frame, violations, alert_level
    ('none' | 'caution' | 'warning') and violation_count.
    """
    h, w = frame.shape[:2]
    annotated = frame.copy() if draw else None
    ids = [det.get("id", i) for i, det in enumerate(detections)]
    feet = [_foot(d) for d in detections]
    heights = [_height(d) for d in detections]

    candidates = {}   # debounce key -> violation dict
    metric = calibration is not None and calibration.has_ground_plane
    ground = [calibration.to_metres(f, w, h) for f in feet] if metric else None

    # --- Person-to-person proximity ---
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            hi, hj = heights[i], heights[j]
            if metric:
                dist, limit = _euclidean(ground[i], ground[j]), PERSONAL_SPACE_M
            else:
                if max(hi, hj) / min(hi, hj) > MAX_DEPTH_RATIO:
                    continue
                dist, limit = _euclidean(feet[i], feet[j]), PERSONAL_SPACE_RATIO * (hi + hj) / 2
            if dist < limit:
                threshold_px = limit
                candidates[("pp", min(ids[i], ids[j]), max(ids[i], ids[j]))] = {
                    "type": "person_to_person",
                    "pair": (i, j),
                    "ids": (ids[i], ids[j]),
                    "distance_px": _euclidean(feet[i], feet[j]),
                    "distance_m": dist if metric else None,
                    "severity": "warning" if dist < threshold_px * 0.5 else "caution",
                    "_draw": (feet[i], feet[j], dist, threshold_px),
                }

    # --- Person-to-robot proximity ---
    robot_center = None
    if robot_position is not None:
        robot_center = (int(robot_position[0] * w), int(robot_position[1] * h))
        robot_ground = calibration.to_metres(robot_center, w, h) if metric else None
        for i in range(len(detections)):
            if metric:
                dist, bubble = _euclidean(ground[i], robot_ground), ROBOT_SPACE_M
            else:
                dist, bubble = _euclidean(feet[i], robot_center), ROBOT_SPACE_RATIO * heights[i]
            if dist < bubble:
                candidates[("pr", ids[i])] = {
                    "type": "person_to_robot",
                    "person_index": i,
                    "id": ids[i],
                    "distance_px": _euclidean(feet[i], robot_center),
                    "distance_m": dist if metric else None,
                    "severity": "warning" if dist < bubble * 0.5 else "caution",
                    "_draw": (feet[i], robot_center, dist, bubble),
                }

    confirmed = _debouncer.update(candidates.keys())
    violations = []
    for key, v in candidates.items():
        if key not in confirmed:
            continue
        a, b, dist, thr = v.pop("_draw")
        if draw:
            if v["type"] == "person_to_person":
                _draw_proximity_line(annotated, a, b, dist, thr)
            else:
                _draw_robot_proximity(annotated, a, b, dist, thr, v["severity"])
        violations.append(v)

    if draw and robot_center is not None:
        cv2.circle(annotated, robot_center, 18, (255, 100, 0), -1)
        cv2.circle(annotated, robot_center, 18, (255, 255, 255), 2)
        cv2.putText(annotated, "ROBOT", (robot_center[0] - 22, robot_center[1] + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Determine overall alert level
    if any(v["severity"] == "warning" for v in violations):
        alert_level = "warning"
    elif violations:
        alert_level = "caution"
    else:
        alert_level = "none"

    if draw:
        _draw_proximity_badge(annotated, violations, alert_level)

    return {
        "annotated_frame": annotated,
        "violations": violations,
        "alert_level": alert_level,
        "violation_count": len(violations),
    }


def _euclidean(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def _draw_proximity_line(frame, c1, c2, dist, threshold):
    """Draw a line between two people who are too close."""
    ratio = 1 - (dist / threshold)  # 0=just touching threshold, 1=perfectly overlapping
    r = int(255 * min(ratio * 2, 1))
    g = int(255 * max(1 - ratio * 2, 0))
    color = (0, g, r)  # BGR: green → red

    cv2.line(frame, c1, c2, color, 2)
    mid = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
    cv2.circle(frame, mid, 6, color, -1)
    cv2.putText(frame, "Too close", (mid[0] + 8, mid[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def _draw_robot_proximity(frame, person_center, robot_center, dist, bubble, severity):
    color = (0, 0, 220) if severity == "warning" else (0, 140, 255)
    cv2.line(frame, person_center, robot_center, color, 2)
    label = "⚠ Near robot"
    cv2.putText(frame, label, (person_center[0] + 8, person_center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def _draw_proximity_badge(frame, violations, alert_level):
    COLOR_MAP = {"none": (50, 200, 50), "caution": (0, 165, 255), "warning": (0, 0, 220)}
    color = COLOR_MAP[alert_level]

    if alert_level == "none":
        text = "  Proximity: OK  "
    elif alert_level == "caution":
        text = f"  Proximity: CAUTION ({len(violations)} violation{'s' if len(violations)>1 else ''})  "
    else:
        text = f"  Proximity: WARNING ({len(violations)} violation{'s' if len(violations)>1 else ''})  "

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_off = 40
    cv2.rectangle(frame, (8, y_off), (8 + tw + 4, y_off + th + 10), (20, 20, 20), -1)
    cv2.putText(frame, text, (10, y_off + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
