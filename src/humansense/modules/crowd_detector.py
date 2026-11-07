"""
Stage 2 — Crowd Detection Module
Detects people (pluggable backend, see humansense.detectors), estimate crowd density, and generate a heatmap overlay.
"""

import cv2
import numpy as np
from collections import deque

from ..detectors import get_detector
from ..utils.tracking import PersonTracker

_tracker = PersonTracker()
_recent_counts = deque(maxlen=5)   # smooths density level against single-frame misses


def reset_state():
    """Call when a new video / camera session starts."""
    _tracker.reset()
    _recent_counts.clear()

def detect_crowd(frame: np.ndarray, confidence_threshold: float = 0.4, draw: bool = True):
    """
    Detect people in a frame and return:
      - annotated_frame: frame with bounding boxes (None when draw=False)
      - detections: list of dicts with bbox, confidence, center
      - density_level: 'low' | 'medium' | 'high'
      - person_count: int
    """
    h, w = frame.shape[:2]

    detections = []
    centers = []
    for d in get_detector().detect(frame, confidence_threshold):
        x1, y1, x2, y2 = d["bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        detections.append({"bbox": (x1, y1, x2, y2), "confidence": d["confidence"],
                           "center": (cx, cy)})
        centers.append((cx, cy))

    _tracker.update(detections)
    person_count = len(detections)

    # Density classification (per 100k pixels of frame area), on a rolling median count
    _recent_counts.append(person_count)
    smoothed_count = float(np.median(_recent_counts))
    area = h * w
    density_ratio = smoothed_count / (area / 100_000)
    if density_ratio < 1.5:
        density_level = "low"
    elif density_ratio < 4.0:
        density_level = "medium"
    else:
        density_level = "high"

    annotated = heatmap = None
    if draw:
        annotated = draw_crowd_overlay(frame.copy(), detections, density_level, person_count)
        heatmap = generate_density_heatmap(frame.shape, centers)

    return {
        "annotated_frame": annotated,
        "heatmap": heatmap,
        "detections": detections,
        "density_level": density_level,
        "person_count": person_count,
        "centers": centers,
    }


def generate_density_heatmap(shape, centers, radius=80, blur=61):
    """Generate a Gaussian density heatmap from person center positions."""
    h, w = shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    for cx, cy in centers:
        # Draw a filled circle for each person
        cv2.circle(heat, (cx, cy), radius, 1.0, -1)

    # Smooth to make it look like a real heatmap
    heat = cv2.GaussianBlur(heat, (blur, blur), 0)
    heat = np.clip(heat, 0, 1)

    # Colorize: blue → green → yellow → red
    heat_uint8 = (heat * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    # Blend with a transparent background
    alpha_mask = (heat * 180).astype(np.uint8)
    heatmap_rgba = cv2.cvtColor(colored, cv2.COLOR_BGR2BGRA)
    heatmap_rgba[:, :, 3] = alpha_mask

    return heatmap_rgba


def draw_crowd_overlay(frame, detections, density_level, person_count):
    """Draw person bounding boxes and density badge on the frame."""
    COLOR_MAP = {
        "low":    (50, 205, 50),    # green
        "medium": (0, 165, 255),    # orange
        "high":   (0, 0, 220),      # red
    }
    color = COLOR_MAP[density_level]

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"person {conf:.0%}"
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Density badge (top-left corner)
    badge_text = f"  Crowd: {density_level.upper()}  |  {person_count} person{'s' if person_count != 1 else ''}  "
    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (8, 8), (8 + tw + 4, 8 + th + 10), (20, 20, 20), -1)
    cv2.putText(frame, badge_text, (10, 8 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    return frame
