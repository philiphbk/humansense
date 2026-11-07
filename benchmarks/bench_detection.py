"""
Person detection (COCO AP) and crowd counting accuracy on COCO val2017 for each detector backend.

    python benchmarks/bench_detection.py --backend mediapipe

* AP/AP50/AP75/APs/APm/APl: standard COCOeval, category 'person', all 5000 val images, detections
  collected at a low confidence (0.05) as the protocol requires.
* Counting: HumanSense's default operating point (conf = 0.4). Reference count = non-crowd person
  annotations with area >= 32*32 px (people small enough to be unreadable are not asked of the model);
  the all-annotations variant is reported too.
"""
import argparse
import io
import json
import statistics
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, save  # noqa: E402


def load_coco(root: Path):
    from pycocotools.coco import COCO
    with zipfile.ZipFile(root / "annotations_trainval2017.zip") as zf:
        data = json.loads(zf.read("annotations/instances_val2017.json"))
    # keep only what we need (person class) in memory
    coco = COCO()
    coco.dataset = data
    coco.createIndex()
    return coco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["mediapipe", "yolo"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--count-conf", type=float, default=0.4)
    a = ap.parse_args()

    from pycocotools.cocoeval import COCOeval
    from humansense import detectors

    root = DATA / "coco"
    coco = load_coco(root)
    det = detectors.set_backend(a.backend)
    img_ids = sorted(coco.getImgIds())
    if a.limit:
        img_ids = img_ids[: a.limit]

    dets, counts, times = [], [], []
    zf = zipfile.ZipFile(root / "val2017.zip")
    t0 = time.time()
    for k, iid in enumerate(img_ids):
        info = coco.imgs[iid]
        img = cv2.imdecode(np.frombuffer(zf.read(f"val2017/{info['file_name']}"), np.uint8), cv2.IMREAD_COLOR)
        t = time.perf_counter()
        out = det.detect(img, conf=0.05)
        times.append((time.perf_counter() - t) * 1000)
        for d in out:
            x1, y1, x2, y2 = d["bbox"]
            dets.append({"image_id": iid, "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "score": d["confidence"]})
        anns = coco.loadAnns(coco.getAnnIds(imgIds=iid, catIds=[1], iscrowd=False))
        counts.append({"gt_all": len(anns), "gt_large": sum(an["area"] >= 32 * 32 for an in anns),
                       "pred": sum(d["confidence"] >= a.count_conf for d in out)})
        if (k + 1) % 500 == 0:
            print(f"{k + 1}/{len(img_ids)} images ({time.time() - t0:.0f}s)", flush=True)

    res = {"benchmark": "COCO val2017 person detection + counting", "backend": a.backend,
           "images": len(img_ids), "detector_ms_per_image": {"median": statistics.median(times),
                                                              "p95": float(np.percentile(times, 95))}}
    if dets:
        # silence pycocotools' prints
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            E = COCOeval(coco, coco.loadRes(dets), "bbox")
            E.params.catIds, E.params.imgIds = [1], img_ids
            E.evaluate(); E.accumulate(); E.summarize()
        names = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
        res["coco"] = {n: float(E.stats[i]) for i, n in enumerate(names)}

    def err(ref):
        e = np.array([c["pred"] - c[ref] for c in counts], float)
        g = np.array([c[ref] for c in counts], float)
        bins = {"0": g == 0, "1-5": (g >= 1) & (g <= 5), "6-15": (g >= 6) & (g <= 15), "16+": g >= 16}
        return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
                "bias": float(e.mean()),
                "MAE_by_gt_count": {k: {"n": int(m.sum()), "MAE": float(np.abs(e[m]).mean()) if m.any() else None,
                                        "bias": float(e[m].mean()) if m.any() else None}
                                    for k, m in bins.items()}}
    res["counting_conf"] = a.count_conf
    res["counting_vs_large_persons"] = err("gt_large")
    res["counting_vs_all_persons"] = err("gt_all")
    save(f"detection_{a.backend}" + ("_debug" if a.limit else ""), res)
    print(json.dumps({k: res[k] for k in ("coco", "detector_ms_per_image")}, indent=1))
    print("count MAE (large persons):", res["counting_vs_large_persons"]["MAE"])


if __name__ == "__main__":
    main()
