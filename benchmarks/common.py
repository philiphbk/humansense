"""Shared helpers for the HumanSense benchmarks."""
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

DATA = Path(os.environ.get("HS_DATA", "data")).expanduser()
HERE = Path(__file__).parent
RESULTS = HERE / "results"
CACHE = HERE / "cache"


def environment() -> dict:
    """Facts needed to interpret timing numbers."""
    import cv2
    import numpy
    import humansense
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "machine": platform.machine(), "numpy": numpy.__version__, "opencv": cv2.__version__,
            "humansense": humansense.__version__}
    try:
        info["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                              text=True).strip()
    except Exception:
        info["cpu"] = platform.processor() or "unknown"
    for mod in ("mediapipe", "ultralytics", "onnxruntime"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    return info


def save(name: str, payload: dict):
    RESULTS.mkdir(exist_ok=True)
    payload = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "environment": environment(), **payload}
    (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {RESULTS / (name + '.json')}")


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p == p and r == r and p + r else float("nan")
    return p, r, f
