"""
Model-file management.

HumanSense does not ship model weights. They are fetched once into a cache directory
(``$HUMANSENSE_MODELS_DIR``, else ``./assets/models`` if it exists, else
``~/.cache/humansense/models``) with ``humansense download-models``.
"""

from __future__ import annotations

import os
import ssl
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Optional

_GCS = "https://storage.googleapis.com/mediapipe-models"

# name -> (url, minimum plausible size in bytes)
MODEL_FILES: Dict[str, tuple] = {
    "pose_landmarker_lite.task": (
        f"{_GCS}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        1_000_000),
    "blaze_face_short_range.tflite": (
        f"{_GCS}/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
        100_000),
    "efficientdet_lite0.tflite": (
        f"{_GCS}/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite",
        1_000_000),
    "efficientdet_lite2.tflite": (
        f"{_GCS}/object_detector/efficientdet_lite2/float16/latest/efficientdet_lite2.tflite",
        1_000_000),
    # HSEmotion EfficientNet-B0 (trained on AffectNet: research use, see docs/LICENSING.md).
    # The hsemotion-onnx package looks for it in ~/.hsemotion, and its own downloader is broken
    # (it never imports urllib.request) so we fetch it ourselves.
    "enet_b0_8_best_vgaf.onnx": (
        "https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/"
        "affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx", 5_000_000),
}

# Models that live somewhere other than models_dir() because a third-party package expects them there
_FIXED_DIR = {"enet_b0_8_best_vgaf.onnx": Path.home() / ".hsemotion"}

# Fetched by the ``ultralytics`` package itself on first use
EXTERNAL_MODELS = {"yolov8n.pt": "downloaded by ultralytics (AGPL-3.0 weights)"}


def models_dir() -> Path:
    env = os.environ.get("HUMANSENSE_MODELS_DIR")
    if env:
        return Path(env).expanduser()
    local = Path.cwd() / "assets" / "models"
    if local.is_dir():
        return local
    return Path.home() / ".cache" / "humansense" / "models"


def model_path(name: str) -> str:
    """Absolute path where model ``name`` is expected (it may not exist yet)."""
    return str(_FIXED_DIR.get(name, models_dir()) / name)


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _ok(path: Path, min_size: int) -> bool:
    return path.is_file() and path.stat().st_size >= min_size


def ensure_model(name: str, *, quiet: bool = False) -> str:
    """Return the path of ``name``, downloading it first if it is missing or truncated."""
    if name not in MODEL_FILES:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODEL_FILES)}")
    url, min_size = MODEL_FILES[name]
    dest = _FIXED_DIR.get(name, models_dir()) / name
    if _ok(dest, min_size):
        return str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if not quiet:
        print(f"downloading {name} -> {dest}", file=sys.stderr)
    with urllib.request.urlopen(url, context=_ssl_context(), timeout=60) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if not _ok(tmp, min_size):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"download of {name} looks truncated (check your connection)")
    tmp.replace(dest)
    return str(dest)


def download_all(names: Optional[Iterable[str]] = None, *, quiet: bool = False) -> Dict[str, str]:
    """Download the given models (default: everything the default pipeline needs)."""
    names = list(names) if names else [
        "pose_landmarker_lite.task", "blaze_face_short_range.tflite", "efficientdet_lite0.tflite",
        "enet_b0_8_best_vgaf.onnx"]
    return {n: ensure_model(n, quiet=quiet) for n in names}
