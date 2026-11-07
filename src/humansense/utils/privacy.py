"""Privacy helpers."""

import cv2
import numpy as np


def blur_heads(frame: np.ndarray, detections, head_fraction: float = 0.2) -> np.ndarray:
    """Pixelate the top `head_fraction` of every person box so faces are not recognisable."""
    out = frame.copy()
    h, w = frame.shape[:2]
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        y_end = y1 + int((y2 - y1) * head_fraction)
        x1, x2, y1, y_end = max(x1, 0), min(x2, w), max(y1, 0), min(y_end, h)
        if x2 - x1 < 2 or y_end - y1 < 2:
            continue
        roi = out[y1:y_end, x1:x2]
        small = cv2.resize(roi, (max((x2 - x1) // 12, 1), max((y_end - y1) // 12, 1)),
                           interpolation=cv2.INTER_LINEAR)
        out[y1:y_end, x1:x2] = cv2.resize(small, (x2 - x1, y_end - y1),
                                          interpolation=cv2.INTER_NEAREST)
    return out
