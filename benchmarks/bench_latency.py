"""
Per-frame processing latency on an otherwise idle machine.

Runs the repo's own demo video (no licensed dataset needed, so this works anywhere) through the
pipeline for a few representative module configurations and reports the per-frame and per-module
timing distribution. Timing numbers are only meaningful on an otherwise idle machine -- other load
inflates tail latency without saying anything about the pipeline itself.

    python benchmarks/bench_latency.py --backend mediapipe
"""
import argparse
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import HERE, save  # noqa: E402

DEMO_VIDEO = HERE.parent / "assets" / "demo.mp4"

# name -> modules. "crowd_proximity_pose" is the ROS node's default module set.
CONFIGS = {
    "crowd_only": ("crowd",),
    "crowd_proximity_pose": ("crowd", "proximity", "pose"),
    "all_modules": ("crowd", "proximity", "pose", "emotion"),
}


def _load_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()
    return frames


def _stats(xs):
    xs = sorted(xs)
    return {"median": statistics.median(xs), "p95": float(np.percentile(xs, 95)),
            "p99": float(np.percentile(xs, 99)), "mean": statistics.fmean(xs), "max": xs[-1]}


def run_config(frames, modules, backend, conf, max_width, n_warmup=10):
    from humansense import Pipeline, PipelineConfig
    pipe = Pipeline(PipelineConfig(modules=modules, backend=backend, conf=conf, max_width=max_width))
    pipe.warmup(n=n_warmup)

    per_frame, per_module = [], {}
    for i, img in enumerate(frames):
        res = pipe.process(img, timestamp=i / 30.0)
        per_frame.append(res.timings_ms["total"])
        for k, v in res.timings_ms.items():
            per_module.setdefault(k, []).append(v)

    return {"n_frames": len(frames), "total_ms": _stats(per_frame),
            "per_module_ms": {k: _stats(v) for k, v in per_module.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mediapipe", choices=["mediapipe", "yolo"])
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--max-width", type=int, default=720)
    ap.add_argument("--repeats", type=int, default=3,
                    help="repeat each config and keep the best (lowest median) run, "
                         "to shed one-off scheduling noise rather than hide it by averaging")
    a = ap.parse_args()

    frames = _load_frames(DEMO_VIDEO)
    out = {"benchmark": "per-frame processing latency (idle machine)", "backend": a.backend,
           "conf": a.conf, "max_width": a.max_width, "source": str(DEMO_VIDEO.relative_to(HERE.parent)),
           "repeats": a.repeats, "configs": {}}
    for name, modules in CONFIGS.items():
        runs = [run_config(frames, modules, a.backend, a.conf, a.max_width) for _ in range(a.repeats)]
        best = min(runs, key=lambda r: r["total_ms"]["median"])
        out["configs"][name] = {"modules": list(modules), **best}
        print(f"{name}: median {best['total_ms']['median']:.1f} ms  p95 {best['total_ms']['p95']:.1f} ms "
              f"({best['n_frames']} frames, best of {a.repeats} runs)")

    save(f"latency_{a.backend}", out)


if __name__ == "__main__":
    main()
