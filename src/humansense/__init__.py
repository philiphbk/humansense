"""HumanSense: non-verbal human cue sensing for robots in hospitals."""

__version__ = "2.0.0"

from .pipeline import ALL_MODULES, Pipeline, PipelineConfig
from .result import Cue, Face, FrameResult, Person, Risk, Violation
from .sources import CameraSource, Frame, ImageSequenceSource, VideoFileSource, open_source

__all__ = [
    "ALL_MODULES",
    "CameraSource",
    "Cue",
    "Face",
    "Frame",
    "FrameResult",
    "ImageSequenceSource",
    "Person",
    "Pipeline",
    "PipelineConfig",
    "Risk",
    "VideoFileSource",
    "Violation",
    "__version__",
    "open_source",
]
