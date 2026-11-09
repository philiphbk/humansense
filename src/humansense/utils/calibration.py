"""
Optional per-camera calibration, loaded from $HUMANSENSE_CALIBRATION or ./calibration.json.

{
  "floor_points_norm": [[x, y], [x, y], [x, y], [x, y]],   # 4 floor points in the image, 0-1
  "floor_points_m":    [[X, Y], [X, Y], [X, Y], [X, Y]],   # the same points in metres on the floor
  "no_fall_zones":     [[[x, y], [x, y], ...], ...]         # polygons (0-1) such as beds / gurney bays
}

With floor points, proximity is measured in real metres (no depth camera needed, but the
camera must be fixed). No-fall zones suppress fall / lying cues for people standing in them.
Every key is optional; without a file the app falls back to the height-based heuristics.
"""

import json
import os

import cv2
import numpy as np

CALIBRATION_PATH = os.environ.get("HUMANSENSE_CALIBRATION", os.path.join(os.getcwd(), "calibration.json"))


class Calibration:
    def __init__(self, data: dict):
        self._H = None
        src, dst = data.get("floor_points_norm"), data.get("floor_points_m")
        if src and dst and len(src) == len(dst) >= 4:
            self._H, _ = cv2.findHomography(np.array(src, np.float32), np.array(dst, np.float32))
        self.zones = [np.array(z, np.float32) for z in data.get("no_fall_zones", []) if len(z) >= 3]

    @property
    def has_ground_plane(self) -> bool:
        return self._H is not None

    def to_metres(self, point_px, frame_w, frame_h):
        """Image pixel -> (X, Y) metres on the floor."""
        p = np.array([[[point_px[0] / frame_w, point_px[1] / frame_h]]], np.float32)
        x, y = cv2.perspectiveTransform(p, self._H)[0, 0]
        return float(x), float(y)

    def in_no_fall_zone(self, point_px, frame_w, frame_h) -> bool:
        pt = (point_px[0] / frame_w, point_px[1] / frame_h)
        return any(cv2.pointPolygonTest(z, pt, False) >= 0 for z in self.zones)


def load_calibration(path: str = CALIBRATION_PATH) -> Calibration | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return Calibration(json.load(f))
    except Exception:  # noqa: BLE001 - a malformed calibration file should fall back, not crash
        return None
