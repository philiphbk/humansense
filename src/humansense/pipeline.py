"""
Headless HumanSense pipeline: one BGR frame in, one typed ``FrameResult`` out.

    from humansense import Pipeline, PipelineConfig
    pipe = Pipeline(PipelineConfig(backend="mediapipe"))
    result = pipe.process(frame_bgr)
    print(result.risk.level, [c.kind for c in result.cues])

Every frontend (CLI, Streamlit, ROS 2 node, benchmarks) goes through this class.

Limitation: the underlying trackers and debouncers are process-global, so run one ``Pipeline``
per process (use one ROS node/process per camera for multi-camera setups).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import detectors
from .result import CUE_SEVERITY, Cue, Face, FrameResult, Person, Risk, Violation, cue_kind

ALL_MODULES = ("crowd", "proximity", "pose", "emotion")


@dataclass
class PipelineConfig:
    modules: Sequence[str] = ALL_MODULES
    backend: str = "auto"                   # "auto" | "yolo" | "mediapipe"
    backend_kwargs: dict = field(default_factory=dict)
    conf: float = 0.40                      # person-detection confidence threshold
    max_width: int | None = 720          # downscale wider frames first (None = native size)
    robot_position: tuple[float, float] | None = None   # normalised (x, y) of the robot in view
    calibration_path: str | None = None  # floor-homography / no-fall-zone JSON (see utils.calibration)
    draw: bool = False                      # render overlays into result.annotated
    show_heatmap: bool = True

    def __post_init__(self):
        unknown = set(self.modules) - set(ALL_MODULES)
        if unknown:
            raise ValueError(f"unknown modules {sorted(unknown)}; choose from {ALL_MODULES}")


class Pipeline:
    def __init__(self, config: PipelineConfig | None = None):
        # Imported here so that `import humansense` stays cheap and doesn't load every model stack
        from .modules import crowd_detector, pose_detector, proximity_detector, risk
        from .utils.calibration import load_calibration

        self.cfg = config or PipelineConfig()
        self._mods = (crowd_detector, proximity_detector, pose_detector, risk)
        self._risk = risk
        self.calibration = (load_calibration(self.cfg.calibration_path)
                            if self.cfg.calibration_path else load_calibration())
        detectors.set_backend(self.cfg.backend, **self.cfg.backend_kwargs)
        self.reset()
        self._frame_id = 0

    # ------------------------------------------------------------------ public API
    def reset(self):
        """Clear trackers/debouncers (call when a new video or camera session starts)."""
        for m in self._mods:
            m.reset_state()
        self._frame_id = 0

    def warmup(self, shape=(480, 640, 3), n: int = 2):
        """Run a few blank frames so lazily-loaded models are ready before the first real frame."""
        mods = set(self.cfg.modules)
        if "pose" in mods:        # these are only touched when a person/face is present, so load them directly
            self._mods[2].get_landmarker()
        if "emotion" in mods:
            from .modules import emotion_detector
            emotion_detector.get_face_detector()
            emotion_detector.get_emotion_recognizer()
        blank = np.zeros(shape, np.uint8)
        for _ in range(n):
            self.process(blank)
        self.reset()

    @property
    def detector_name(self) -> str:
        return detectors.get_detector().name

    def process(self, frame_bgr: np.ndarray, timestamp: float | None = None) -> FrameResult:
        cfg, mods = self.cfg, set(self.cfg.modules)
        oh, ow = frame_bgr.shape[:2]
        scale = 1.0
        frame = frame_bgr
        if cfg.max_width and ow > cfg.max_width:
            scale = cfg.max_width / ow
            frame = cv2.resize(frame_bgr, (cfg.max_width, int(oh * scale)), interpolation=cv2.INTER_AREA)

        self._frame_id += 1
        res = FrameResult(frame_id=self._frame_id,
                          timestamp=time.time() if timestamp is None else timestamp,
                          frame_size=(ow, oh))
        raw = res.raw
        t_all = time.perf_counter()

        def timed(name, fn):
            t0 = time.perf_counter()
            try:
                return fn()
            except Exception as e:  # noqa: BLE001 - one failing module must never take the robot's sensing down
                res.errors.append(f"{name}: {type(e).__name__}: {e}")
                return None
            finally:
                res.timings_ms[name] = (time.perf_counter() - t0) * 1000.0

        # Proximity and pose need person detections even when the crowd overlay is hidden.
        # NB: the key is "crowd" to match v1 naming, but it is the person-detection stage.
        detections = []
        if mods & {"crowd", "proximity", "pose"}:
            cr = timed("detect", lambda: self._mods[0].detect_crowd(
                frame, confidence_threshold=cfg.conf, draw=cfg.draw))
            if cr is not None:
                raw["crowd"] = cr
                detections = cr["detections"]
                res.person_count = cr["person_count"]
                res.density_level = cr["density_level"]

        if "proximity" in mods:
            pr = timed("proximity", lambda: self._mods[1].check_proximity(
                frame, detections, robot_position=cfg.robot_position,
                calibration=self.calibration, draw=cfg.draw))
            if pr is not None:
                raw["proximity"] = pr
                res.proximity_level = pr["alert_level"]
                res.violations = [Violation(
                    type=v["type"], ids=tuple(v.get("ids", (v.get("id"),))),
                    severity=v["severity"], distance_px=float(v["distance_px"]),
                    distance_m=v.get("distance_m")) for v in pr["violations"]]

        if "pose" in mods:
            ps = timed("pose", lambda: self._mods[2].analyze_pose(
                frame, detections, self.calibration, draw=cfg.draw))
            if ps is not None:
                raw["pose"] = ps
                if ps.get("error"):
                    res.errors.append(f"pose: {ps['error']}")
                for rec in ps.get("cue_records", []):
                    k = cue_kind(rec["cue"])
                    res.cues.append(Cue(person_id=rec["id"], kind=k, label=rec["cue"],
                                        severity=CUE_SEVERITY.get(k, 0)))

        if "emotion" in mods:
            from .modules.emotion_detector import detect_emotions
            em = timed("emotion", lambda: detect_emotions(frame, draw=cfg.draw))
            if em is not None:
                raw["emotions"] = em
                res.faces = [Face(bbox=_unscale(f["bbox"], scale), emotion=f["emotion"],
                                  label=f["label"], confidence=f["confidence"],
                                  priority=f["priority"]) for f in em["face_results"]]

        rk = timed("risk", lambda: self._risk.assess(raw, detections))
        if rk is not None:
            raw["risk"] = rk
            res.risk = Risk(score=rk["score"], level=rk["level"], reasons=list(rk["reasons"]))

        res.people = [Person(id=d.get("id", i), bbox=_unscale(d["bbox"], scale),
                             confidence=float(d["confidence"]), speed=float(d.get("speed", 0.0)),
                             still_frames=int(d.get("still_frames", 0)))
                      for i, d in enumerate(detections)]

        if cfg.draw:
            from .utils.frame_utils import composite_detections
            res.annotated = composite_detections(frame, raw, show_heatmap=cfg.show_heatmap)
        res.timings_ms["total"] = (time.perf_counter() - t_all) * 1000.0
        res.raw["detections"] = detections
        return res


def _unscale(bbox, scale):
    if scale == 1.0:
        return tuple(int(v) for v in bbox)
    return tuple(round(v / scale) for v in bbox)
