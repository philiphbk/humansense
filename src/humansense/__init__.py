"""HumanSense: non-verbal human cue sensing for robots in hospitals."""

__version__ = "2.0.0"

from .pipeline import ALL_MODULES, Pipeline, PipelineConfig
from .result import Cue, Face, FrameResult, Person, Risk, Violation
from .sources import CameraSource, Frame, ImageSequenceSource, VideoFileSource, open_source

__all__ = [
    "ALL_MODULES", "Pipeline", "PipelineConfig", "FrameResult", "Person", "Cue", "Face", "Risk",
    "Violation", "Frame", "CameraSource", "VideoFileSource", "ImageSequenceSource", "open_source",
    "__version__",
]
