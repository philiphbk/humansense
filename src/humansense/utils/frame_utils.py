"""
Shared frame utilities — blending, compositing, and event logging.
"""

import cv2
import numpy as np
from datetime import datetime
from collections import deque

from .event_store import save_event

# Rolling event log (last 50 events)
event_log = deque(maxlen=50)


def blend_heatmap(frame: np.ndarray, heatmap_rgba: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap (BGRA) onto a BGR frame using per-pixel alpha."""
    if heatmap_rgba is None:
        return frame
    heatmap_bgr = heatmap_rgba[:, :, :3]
    mask = heatmap_rgba[:, :, 3].astype(np.float32) / 255.0 * alpha
    mask3 = np.stack([mask, mask, mask], axis=-1)
    result = (frame.astype(np.float32) * (1 - mask3) + heatmap_bgr.astype(np.float32) * mask3)
    return result.clip(0, 255).astype(np.uint8)


def composite_detections(base: np.ndarray, results: dict, show_heatmap: bool = True,
                          show_pose: bool = True, show_emotions: bool = True,
                          show_proximity: bool = True) -> np.ndarray:
    """
    Combine all detection layers onto a single frame.
    Layers are applied in order: heatmap → crowd boxes → proximity → pose → emotions.
    Each module's annotated frame carries its own overlays.
    """
    frame = base.copy()

    # Layer 1: Heatmap (blended underneath)
    if show_heatmap and results.get("crowd") and results["crowd"].get("heatmap") is not None:
        frame = blend_heatmap(frame, results["crowd"]["heatmap"])

    # Layer 2: Crowd bounding boxes
    if results.get("crowd") and results["crowd"].get("annotated_frame") is not None:
        crowd_ann = results["crowd"]["annotated_frame"]
        frame = _soft_overlay(frame, crowd_ann, base)

    # Layer 3: Proximity overlays
    if show_proximity and results.get("proximity") and results["proximity"].get("annotated_frame") is not None:
        prox_ann = results["proximity"]["annotated_frame"]
        frame = _soft_overlay(frame, prox_ann, base)

    # Layer 4: Pose skeleton
    if show_pose and results.get("pose") and results["pose"].get("annotated_frame") is not None:
        pose_ann = results["pose"]["annotated_frame"]
        frame = _soft_overlay(frame, pose_ann, base)

    # Layer 5: Emotion face boxes (top layer)
    if show_emotions and results.get("emotions") and results["emotions"].get("annotated_frame") is not None:
        emo_ann = results["emotions"]["annotated_frame"]
        frame = _soft_overlay(frame, emo_ann, base)

    # Draw status bar at bottom
    frame = draw_status_bar(frame, results)

    return frame


def _soft_overlay(target: np.ndarray, annotated: np.ndarray, base: np.ndarray) -> np.ndarray:
    """
    Extract only the pixels that changed in 'annotated' vs 'base',
    and apply them onto 'target'. This lets multiple annotation layers stack cleanly.
    """
    diff_mask = np.any(annotated != base, axis=2)
    result = target.copy()
    result[diff_mask] = annotated[diff_mask]
    return result


def draw_status_bar(frame: np.ndarray, results: dict) -> np.ndarray:
    """Draw a dark status bar at the bottom of the frame with key metrics."""
    h, w = frame.shape[:2]
    bar_h = 36
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    bar[:] = (25, 25, 25)

    # Metrics
    person_count = results.get("crowd", {}).get("person_count", 0)
    density = results.get("crowd", {}).get("density_level", "—")
    prox_alert = results.get("proximity", {}).get("alert_level", "—")
    pose_cues = results.get("pose", {}).get("cues", [])
    top_emotion = results.get("emotions", {}).get("highest_priority_cue")

    emotion_text = top_emotion["label"] if top_emotion else "—"

    risk = results.get("risk")
    items = [
        ("RISK", f"{risk['level'].upper()} {risk['score']}" if risk else "—"),
        ("PEOPLE", str(person_count)),
        ("DENSITY", density.upper()),
        ("PROXIMITY", prox_alert.upper()),
        ("POSE", pose_cues[0][:18] if pose_cues else "Normal"),
        ("EMOTION", emotion_text),
    ]

    x = 12
    for key, val in items:
        COLOR_KEY = (130, 130, 130)
        COLOR_VAL_DEFAULT = (200, 200, 200)

        # Color-code values
        if key == "DENSITY":
            val_color = {"LOW": (50, 200, 50), "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 220)}.get(val, COLOR_VAL_DEFAULT)
        elif key == "PROXIMITY":
            val_color = {"NONE": (50, 200, 50), "CAUTION": (0, 165, 255), "WARNING": (0, 0, 220)}.get(val, COLOR_VAL_DEFAULT)
        elif key == "RISK":
            val_color = (0, 0, 220) if "CRITICAL" in val else (0, 165, 255) if "ELEVATED" in val else (50, 200, 50)
        elif key == "POSE" and val != "Normal":
            val_color = (0, 165, 255)
        elif key == "EMOTION" and val not in ("—", "Comfortable", "Neutral"):
            val_color = (0, 140, 255)
        else:
            val_color = COLOR_VAL_DEFAULT

        cv2.putText(bar, key + ":", (x, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_KEY, 1, cv2.LINE_AA)
        cv2.putText(bar, val, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, val_color, 1, cv2.LINE_AA)

        (kw, _), _ = cv2.getTextSize(key + ":", cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        (vw, _), _ = cv2.getTextSize(val, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        x += max(kw, vw) + 24

        if x > w - 80:
            break

    # Timestamp
    ts = datetime.now().strftime("%H:%M:%S")
    (tsw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.putText(bar, ts, (w - tsw - 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1, cv2.LINE_AA)

    # Append bar to frame
    return np.vstack([frame, bar])


_DEDUPE_SECONDS = 3.0
_last_logged = {}


def log_event(event_type: str, detail: str):
    # Skip an identical event logged moments ago (one per processed frame otherwise)
    now = datetime.now().timestamp()
    if now - _last_logged.get((event_type, detail), 0) < _DEDUPE_SECONDS:
        return
    _last_logged[(event_type, detail)] = now
    save_event(event_type, detail)
    event_log.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "type": event_type,
        "detail": detail,
    })


def get_event_log():
    return list(reversed(event_log))
