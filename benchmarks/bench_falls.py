"""
Fall detection on the UR Fall Detection dataset (URFD, cam0 RGB; Kwolek & Kepski, 2014).
30 fall sequences + 40 activities of daily living (ADL). 16 of the ADLs deliberately include lying
on the floor / a couch, which makes them hard negatives for any posture-based rule.

    python benchmarks/bench_falls.py --backend mediapipe

Per-frame outputs are cached so that metrics can be recomputed without re-running inference.
"""
import argparse
import csv
import json
import re
import statistics
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE, DATA, prf, save, wilson  # noqa: E402

FPS = 30.0


def load_labels(root: Path):
    labels = defaultdict(dict)
    for f in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        for r in csv.reader(open(root / f)):
            labels[r[0]][int(r[1])] = int(r[2])
    return labels


def frames_from_zip(path: Path):
    with zipfile.ZipFile(path) as zf:
        names = sorted((n for n in zf.namelist() if n.endswith(".png")),
                       key=lambda n: int(re.search(r"-(\d+)\.png$", n).group(1)))
        for n in names:
            idx = int(re.search(r"-(\d+)\.png$", n).group(1))
            yield idx, cv2.imdecode(np.frombuffer(zf.read(n), np.uint8), cv2.IMREAD_COLOR)


def run(args):
    from humansense import Pipeline, PipelineConfig
    root = DATA / "urfd"
    labels = load_labels(root)
    pipe = Pipeline(PipelineConfig(modules=("crowd", "proximity", "pose"), backend=args.backend,
                                   max_width=None, conf=args.conf))
    CACHE.mkdir(exist_ok=True)
    cache = CACHE / f"urfd_{args.backend}.jsonl"
    seqs = sorted(labels)
    if args.limit:
        seqs = ([s for s in seqs if s.startswith("fall")][: args.limit]
                + [s for s in seqs if s.startswith("adl")][: args.limit])
    rows, t0 = [], time.time()
    for si, seq in enumerate(seqs):
        pipe.reset()
        for idx, img in frames_from_zip(root / f"{seq}-cam0-rgb.zip"):
            r = pipe.process(img, timestamp=idx / FPS)
            rows.append({"seq": seq, "frame": idx, "label": labels[seq].get(idx, -1),
                         "fall": int(any(c.kind == "fall" for c in r.cues)),
                         "fall_risk": int(any(c.kind == "fall_risk" for c in r.cues)),
                         "critical": int(r.risk.level == "critical"),
                         "score": r.risk.score, "n": r.person_count,
                         # per-person boxes and posture flags, so temporal rules can be studied offline
                         "people": [[q.id, *q.bbox] for q in r.people],
                         "fall_ids": [c.person_id for c in r.cues if c.kind == "fall"],
                         "fall_risk_ids": [c.person_id for c in r.cues if c.kind == "fall_risk"],
                         "ms": r.timings_ms["total"]})
        print(f"[{si + 1}/{len(seqs)}] {seq} done ({time.time() - t0:.0f}s)", flush=True)
    cache.write_text("\n".join(json.dumps(r) for r in rows))
    return rows


# ---------------------------------------------------------------- metrics
ALARMS = {
    "fall_cue": lambda r: r["fall"],                              # the pose module's FALL cue
    "fall_or_risk_cue": lambda r: r["fall"] or r["fall_risk"],    # also the weaker "fall risk" cue
    "risk_critical": lambda r: r["critical"],                     # fused risk level (what a planner sees)
}


def analyze(rows):
    by_seq = defaultdict(list)
    for r in rows:
        by_seq[r["seq"]].append(r)
    for v in by_seq.values():
        v.sort(key=lambda r: r["frame"])
    falls = sorted(s for s in by_seq if s.startswith("fall"))
    adl_lying = sorted(s for s in by_seq if s.startswith("adl") and any(r["label"] == 1 for r in by_seq[s]))
    adl_plain = sorted(s for s in by_seq if s.startswith("adl") and s not in adl_lying)

    out = {"n_sequences": {"fall": len(falls), "adl_lying": len(adl_lying), "adl_plain": len(adl_plain)},
           "n_frames": len(rows), "alarms": {}}
    for name, alarm_fn in ALARMS.items():
        alarm_seq = {s: [r["frame"] for r in v if alarm_fn(r)] for s, v in by_seq.items()}
        # Fall sequence is detected if an alarm occurs from 0.5 s before the fall onset (first label>=0 frame)
        detected, delays = [], []
        for s in falls:
            onset = min(r["frame"] for r in by_seq[s] if r["label"] >= 0)
            lying = min((r["frame"] for r in by_seq[s] if r["label"] == 1), default=onset)
            hits = [f for f in alarm_seq[s] if f >= onset - int(0.5 * FPS)]
            detected.append(bool(hits))
            if hits:
                delays.append((hits[0] - lying) / FPS)
        sens_k = sum(detected)
        fa_plain = sum(bool(alarm_seq[s]) for s in adl_plain)
        fa_lying = sum(bool(alarm_seq[s]) for s in adl_lying)
        # Frame level: positive = dataset says the person is lying on the ground (label 1)
        tp = fp = n_miss = tn = 0
        for r in rows:
            pos, pred = r["label"] == 1, bool(alarm_fn(r))
            tp += pos and pred
            fp += (not pos) and pred
            n_miss += pos and (not pred)
            tn += (not pos) and (not pred)
        p, rc, f1 = prf(tp, fp, n_miss)
        out["alarms"][name] = {
            "sensitivity": {"k": sens_k, "n": len(falls), "value": sens_k / len(falls),
                            "ci95": wilson(sens_k, len(falls))},
            "false_alarm_adl_plain": {"k": fa_plain, "n": len(adl_plain),
                                      "specificity": 1 - fa_plain / len(adl_plain),
                                      "ci95_spec": wilson(len(adl_plain) - fa_plain, len(adl_plain))},
            "false_alarm_adl_lying": {"k": fa_lying, "n": len(adl_lying),
                                      "specificity": 1 - fa_lying / len(adl_lying),
                                      "ci95_spec": wilson(len(adl_lying) - fa_lying, len(adl_lying))},
            "delay_after_lying_onset_s": {"median": statistics.median(delays) if delays else None,
                                          "n": len(delays)},
            "frame_level_vs_lying_label": {"precision": p, "recall": rc, "f1": f1,
                                           "tp": tp, "fp": fp, "fn": n_miss, "tn": tn},
            "missed_falls": [s for s, d in zip(falls, detected) if not d],
            "false_alarm_sequences": [s for s in adl_plain + adl_lying if alarm_seq[s]],
        }
    ms = [r["ms"] for r in rows]
    out["compute_ms_per_frame"] = {"median": statistics.median(ms), "p95": float(np.percentile(ms, 95))}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mediapipe", choices=["mediapipe", "yolo"])
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N fall + N ADL sequences")
    ap.add_argument("--reuse", action="store_true", help="recompute metrics from the cached frames")
    a = ap.parse_args()
    cache = CACHE / f"urfd_{a.backend}.jsonl"
    rows = ([json.loads(l) for l in cache.read_text().splitlines()] if a.reuse and cache.exists()
            else run(a))
    res = analyze(rows)
    save(f"falls_{a.backend}" + ("_debug" if a.limit else ""),
         {"benchmark": "URFD cam0 RGB fall detection", "backend": a.backend, "conf": a.conf, **res})
    print(json.dumps({k: v for k, v in res.items() if k != "alarms"}, indent=1))
    for n, m in res["alarms"].items():
        print(n, "sens", m["sensitivity"]["k"], "/", m["sensitivity"]["n"],
              "| ADL plain FA", m["false_alarm_adl_plain"]["k"], "/", m["false_alarm_adl_plain"]["n"],
              "| ADL lying FA", m["false_alarm_adl_lying"]["k"], "/", m["false_alarm_adl_lying"]["n"])
