"""Command-line interface:  humansense {run, download-models, app}"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time


def _cmd_download(args):
    from .models import download_all
    paths = download_all(args.names or None)
    for n, p in paths.items():
        print(f"{n}: {p}")
    return 0


def _cmd_app(args):
    import os
    import subprocess
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("The demo UI needs Streamlit:  pip install 'humansense[app]'", file=sys.stderr)
        return 2
    script = os.path.join(os.path.dirname(__file__), "app", "streamlit_app.py")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", script, *args.streamlit_args])


def _cmd_run(args):
    import cv2

    from .pipeline import Pipeline, PipelineConfig
    from .sources import CameraSource, open_source

    cfg = PipelineConfig(
        modules=tuple(args.modules.split(",")), backend=args.backend, conf=args.conf,
        max_width=args.max_width or None, draw=bool(args.display or args.save_video),
        robot_position=tuple(args.robot) if args.robot else None, calibration_path=args.calibration)
    pipe = Pipeline(cfg)
    out = (
        open(args.jsonl, "w")  # noqa: SIM115 - lives for the whole run, closed explicitly below
        if args.jsonl and args.jsonl != "-" else (sys.stdout if args.jsonl else None)
    )
    writer = None
    totals, n = [], 0
    lat = []
    with open_source(args.source, loop=False, realtime=args.realtime, fps=args.fps) as src:
        live = isinstance(src, CameraSource)
        t_start = time.perf_counter()
        for frame in src:
            res = pipe.process(frame.image, timestamp=frame.timestamp)
            n += 1
            totals.append(res.timings_ms["total"])
            if live:
                lat.append((time.time() - frame.timestamp) * 1000.0)
            if out:
                out.write(json.dumps(res.to_dict()) + "\n")
            if res.annotated is not None:
                if args.save_video:
                    if writer is None:
                        h, w = res.annotated.shape[:2]
                        writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 args.fps or 15.0, (w, h))
                    writer.write(res.annotated)
                if args.display:
                    cv2.imshow("HumanSense", res.annotated)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
            if args.max_frames and n >= args.max_frames:
                break
        wall = time.perf_counter() - t_start
        if writer:
            writer.release()
        if out and out is not sys.stdout:
            out.close()
        if n:
            msg = (f"{n} frames in {wall:.1f}s = {n / wall:.1f} fps; "
                   f"per-frame compute median {statistics.median(totals):.0f} ms "
                   f"(detector: {pipe.detector_name})")
            if live:
                msg += (f"; camera delivered {src.delivered}/{src.captured} grabbed frames "
                        f"({src.dropped} skipped to stay real-time); "
                        f"capture->result latency median {statistics.median(lat):.0f} ms")
            print(msg, file=sys.stderr)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="humansense", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download-models", help="fetch MediaPipe model files into the cache")
    d.add_argument("names", nargs="*", help="model file names (default: the standard set)")
    d.set_defaults(fn=_cmd_download)

    r = sub.add_parser("run", help="process a camera, video file or image directory headlessly")
    r.add_argument("--source", default="0", help="camera index/URL, video file or image dir (default 0)")
    r.add_argument("--modules", default="crowd,proximity,pose,emotion")
    r.add_argument("--backend", default="auto", choices=["auto", "yolo", "mediapipe"])
    r.add_argument("--conf", type=float, default=0.4)
    r.add_argument("--max-width", type=int, default=720, help="0 = native resolution")
    r.add_argument("--robot", type=float, nargs=2, metavar=("X", "Y"), help="normalised robot position")
    r.add_argument("--calibration", help="calibration.json (floor homography / no-fall zones)")
    r.add_argument("--fps", type=float, help="camera fps request / image-sequence rate")
    r.add_argument("--realtime", action="store_true", help="pace video files to their frame rate")
    r.add_argument("--max-frames", type=int)
    r.add_argument("--jsonl", help="write one JSON result per line ('-' = stdout)")
    r.add_argument("--save-video", help="write the annotated video here (mp4)")
    r.add_argument("--display", action="store_true", help="show a window (needs non-headless OpenCV)")
    r.set_defaults(fn=_cmd_run)

    a = sub.add_parser("app", help="launch the Streamlit demo UI")
    a.add_argument("streamlit_args", nargs=argparse.REMAINDER)
    a.set_defaults(fn=_cmd_app)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
