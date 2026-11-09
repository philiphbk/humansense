"""
Lightweight temporal helpers: a greedy IoU person tracker and an alert debouncer.
Both are stepped once per *processed* frame, so they stay consistent with frame skipping.
"""

from collections.abc import Hashable, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class PersonTracker:
    """
    Assigns stable ids to detections. Each track predicts its next box from its last
    velocity, and detections are matched to predictions with the Hungarian algorithm
    (optimal assignment on IoU, with a centre-distance fallback for fast movers).
    Also records per-person motion: `speed` (body heights moved per processed frame)
    and `still_frames` (consecutive processed frames with almost no movement).
    """

    STILL_SPEED = 0.03     # body heights per processed frame below which a person is "still"

    def __init__(self, iou_threshold: float = 0.2, max_missed: int = 8):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._tracks: dict[int, dict] = {}
        self._next_id = 1

    def reset(self):
        self._tracks.clear()
        self._next_id = 1

    @staticmethod
    def _predict(t):
        vx, vy = t["vel"]
        x1, y1, x2, y2 = t["bbox"]
        k = 1 + t["missed"]
        return (x1 + vx * k, y1 + vy * k, x2 + vx * k, y2 + vy * k)

    @staticmethod
    def _centre_dist(a, b):
        h = max(a[3] - a[1], b[3] - b[1], 1)
        ca, cb = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2), ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5 / h

    def update(self, detections: list[dict]) -> list[dict]:
        """Adds 'id', 'speed' and 'still_frames' to each detection (in place)."""
        tids = list(self._tracks)
        used_det, used_trk = set(), set()

        if detections and tids:
            cost = np.ones((len(detections), len(tids)))
            for di, d in enumerate(detections):
                for ti, tid in enumerate(tids):
                    pred = self._predict(self._tracks[tid])
                    score = _iou(d["bbox"], pred)
                    if score < self.iou_threshold:
                        # Fast mover: accept a close centre (within ~0.6 body heights)
                        cd = self._centre_dist(d["bbox"], pred)
                        score = max(score, 0.3 * (1 - cd / 0.6)) if cd < 0.6 else score
                    cost[di, ti] = 1 - score
            rows, cols = linear_sum_assignment(cost)
            for di, ti in zip(rows, cols):
                if cost[di, ti] <= 1 - self.iou_threshold:
                    self._assign(detections[di], tids[ti])
                    used_det.add(di)
                    used_trk.add(tids[ti])

        for tid in tids:
            if tid not in used_trk:
                self._tracks[tid]["missed"] += 1
                if self._tracks[tid]["missed"] > self.max_missed:
                    del self._tracks[tid]

        for di, d in enumerate(detections):
            if di not in used_det:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"bbox": d["bbox"], "vel": (0.0, 0.0), "missed": 0, "still": 0}
                d["id"], d["speed"], d["still_frames"] = tid, 0.0, 0
        return detections

    def _assign(self, d, tid):
        t = self._tracks[tid]
        ob, nb = t["bbox"], d["bbox"]
        steps = 1 + t["missed"]
        new_vel = tuple((((nb[i] + nb[i + 2]) - (ob[i] + ob[i + 2])) / 2) / steps for i in (0, 1))
        # Smooth the velocity so one jittery box does not fling the prediction
        t["vel"] = tuple(0.5 * o + 0.5 * n for o, n in zip(t["vel"], new_vel))
        height = max(nb[3] - nb[1], 1)
        speed = (new_vel[0] ** 2 + new_vel[1] ** 2) ** 0.5 / height
        t["still"] = t["still"] + 1 if speed < self.STILL_SPEED else 0
        t["bbox"], t["missed"] = nb, 0
        d["id"], d["speed"], d["still_frames"] = tid, speed, t["still"]


class Debouncer:
    """
    Confirms a condition only after it holds for `on_frames` consecutive updates,
    and releases it only after it has been absent for `off_frames` updates.
    """

    def __init__(self, on_frames: int = 2, off_frames: int = 3):
        self.on_frames = on_frames
        self.off_frames = off_frames
        self._hits: dict[Hashable, int] = {}
        self._misses: dict[Hashable, int] = {}
        self._active: set[Hashable] = set()

    def reset(self):
        self._hits.clear()
        self._misses.clear()
        self._active.clear()

    def update(self, keys: Iterable[Hashable]) -> set[Hashable]:
        keys = set(keys)
        for k in keys:
            self._hits[k] = self._hits.get(k, 0) + 1
            self._misses[k] = 0
            if self._hits[k] >= self.on_frames:
                self._active.add(k)
        for k in list(self._hits):
            if k not in keys:
                self._misses[k] = self._misses.get(k, 0) + 1
                self._hits[k] = 0
                if self._misses[k] >= self.off_frames:
                    self._active.discard(k)
                    del self._hits[k], self._misses[k]
        return set(self._active)
