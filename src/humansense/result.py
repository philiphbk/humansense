"""Typed, JSON-serialisable output of one pipeline step."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# Cue taxonomy. `kind` is stable and machine-readable; `label` is the human text.
CUE_KINDS = ("fall", "fall_risk", "raised_hand", "protective_posture", "arms_crossed", "seated")
# Severity per cue kind: 0 = informational, 1 = attention, 2 = urgent
CUE_SEVERITY = {"fall": 2, "fall_risk": 1, "raised_hand": 1, "protective_posture": 1,
                "arms_crossed": 0, "seated": 0}


def cue_kind(text: str) -> str:
    """Map a pose-module cue string to its machine-readable kind."""
    t = text.lower()
    if t.startswith("fall detected"):
        return "fall"
    if t.startswith("fall risk"):
        return "fall_risk"
    if t.startswith("raised hand"):
        return "raised_hand"
    if t.startswith("protective"):
        return "protective_posture"
    if t.startswith("arms crossed"):
        return "arms_crossed"
    if t.startswith("seated"):
        return "seated"
    return "other"


@dataclass
class Person:
    id: int
    bbox: tuple[int, int, int, int]        # x1, y1, x2, y2 in ORIGINAL frame pixels
    confidence: float
    speed: float = 0.0                      # body heights per processed frame
    still_frames: int = 0                   # consecutive processed frames with ~no movement


@dataclass
class Cue:
    person_id: int
    kind: str
    label: str
    severity: int


@dataclass
class Violation:
    type: str                               # "person_to_person" | "person_to_robot"
    ids: tuple[int, ...]
    severity: str                           # "caution" | "warning"
    distance_px: float
    distance_m: float | None = None      # only with a floor calibration


@dataclass
class Face:
    bbox: tuple[int, int, int, int]
    emotion: str
    label: str
    confidence: float
    priority: int


@dataclass
class Risk:
    score: int
    level: str                              # "ok" | "elevated" | "critical"
    reasons: list[str] = field(default_factory=list)


@dataclass
class FrameResult:
    frame_id: int
    timestamp: float                        # seconds; capture time if the source provides it
    frame_size: tuple[int, int]             # (width, height) of the ORIGINAL frame
    people: list[Person] = field(default_factory=list)
    person_count: int = 0
    density_level: str = "low"
    proximity_level: str = "none"           # "none" | "caution" | "warning"
    violations: list[Violation] = field(default_factory=list)
    cues: list[Cue] = field(default_factory=list)
    faces: list[Face] = field(default_factory=list)
    risk: Risk = field(default_factory=lambda: Risk(0, "ok"))
    timings_ms: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    # Not serialised: rendering for UIs. Only present when the pipeline was built with draw=True.
    annotated: np.ndarray | None = field(default=None, repr=False, compare=False)
    # Raw module outputs, kept for the Streamlit UI and for debugging.
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("annotated", None)
        d.pop("raw", None)
        return d

    @property
    def has_fall(self) -> bool:
        return any(c.kind == "fall" for c in self.cues)
