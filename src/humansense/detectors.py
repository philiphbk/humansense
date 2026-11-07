"""
Pluggable person detectors.

Two backends ship with HumanSense:

* ``yolo``       – Ultralytics YOLOv8 (accurate; the ``ultralytics`` package and its weights are
                   AGPL-3.0, so it is an *optional* extra: ``pip install humansense[yolo]``).
* ``mediapipe``  – MediaPipe EfficientDet-Lite object detector, "person" class only
                   (Apache-2.0 code and weights; installed with the base package).

Both return the same thing: ``[{"bbox": (x1, y1, x2, y2), "confidence": float}, ...]`` in
integer pixel coordinates of the frame passed in.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

Detection = Dict[str, object]

BACKENDS = ("yolo", "mediapipe")


class PersonDetector:
    name = "base"

    def detect(self, frame_bgr: np.ndarray, conf: float = 0.4) -> List[Detection]:  # pragma: no cover
        raise NotImplementedError


class YoloPersonDetector(PersonDetector):
    name = "yolo"

    def __init__(self, weights: str = "yolov8n.pt", imgsz: Optional[int] = None):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError(
                "The 'yolo' person-detector backend needs ultralytics (AGPL-3.0): "
                "pip install 'humansense[yolo]'  — or use backend='mediapipe'.") from e
        local = os.path.join(os.getcwd(), weights)
        self._model = YOLO(local if os.path.exists(local) else weights)
        self._imgsz = imgsz
        self.weights = weights

    def detect(self, frame_bgr, conf=0.4):
        kw = {"imgsz": self._imgsz} if self._imgsz else {}
        res = self._model(frame_bgr, classes=[0], conf=conf, verbose=False, **kw)[0]
        out = []
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            out.append({"bbox": (x1, y1, x2, y2), "confidence": float(box.conf[0])})
        return out


class MediaPipePersonDetector(PersonDetector):
    name = "mediapipe"

    def __init__(self, model: str = "efficientdet_lite0.tflite", min_score: float = 0.05):
        import mediapipe as mp  # noqa: F401  (imported here so the name `mp` is bound below)
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        from .models import ensure_model

        self._mp = mp
        self._min_score = min_score
        opts = mp_vision.ObjectDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model(model)),
            running_mode=mp_vision.RunningMode.IMAGE,
            category_allowlist=["person"],
            score_threshold=min_score,
            max_results=-1,
        )
        self._det = mp_vision.ObjectDetector.create_from_options(opts)
        self.weights = model

    def detect(self, frame_bgr, conf=0.4):
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        res = self._det.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        h, w = frame_bgr.shape[:2]
        out = []
        for d in res.detections:
            score = float(d.categories[0].score)
            if score < max(conf, self._min_score):
                continue
            b = d.bounding_box
            x1, y1 = max(int(b.origin_x), 0), max(int(b.origin_y), 0)
            x2, y2 = min(int(b.origin_x + b.width), w), min(int(b.origin_y + b.height), h)
            out.append({"bbox": (x1, y1, x2, y2), "confidence": score})
        return out


_current: Optional[PersonDetector] = None
_preferred = "auto"


def set_backend(name: str = "auto", **kwargs) -> PersonDetector:
    """Select (and load) the person-detector backend. ``auto`` = yolo if installed else mediapipe."""
    global _current, _preferred
    _preferred = name
    if name == "auto":
        try:
            import ultralytics  # noqa: F401
            name = "yolo"
        except ImportError:
            name = "mediapipe"
    if name == "yolo":
        _current = YoloPersonDetector(**kwargs)
    elif name == "mediapipe":
        _current = MediaPipePersonDetector(**kwargs)
    else:
        raise ValueError(f"unknown backend {name!r}; choose from {BACKENDS} or 'auto'")
    return _current


def get_detector() -> PersonDetector:
    return _current if _current is not None else set_backend(_preferred)
