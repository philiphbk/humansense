"""
Pose benchmark on COCO val2017 person keypoints.

(1) Keypoint accuracy: MediaPipe PoseLandmarker (lite) on ground-truth person crops (the same padded
    crop HumanSense feeds it), mapped to the 17 COCO joints; COCO OKS-AP, so detection errors are
    excluded and only the pose estimator is measured.
(2) Cue fidelity: HumanSense's rule-based cues (raised hand, seated, arms crossed, protective,
    fall) computed from the predicted landmarks vs. from the human-annotated keypoints. This
    measures how much pose-estimation error propagates into the cues; it does NOT validate the
    rules themselves (the reference applies the same rule to annotated joints).

    python benchmarks/bench_pose.py
"""
import contextlib
import io
import json
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, prf, save  # noqa: E402

# COCO joint order -> MediaPipe landmark index
COCO_TO_MP = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
MIN_KPS, MIN_BOX_H = 5, 64


def crop_box(bbox, w, h):
    x, y, bw, bh = bbox
    x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)
    pw, ph = int((x2 - x1) * 0.15), int((y2 - y1) * 0.10)           # same padding as analyze_pose
    return max(x1 - pw, 0), max(y1 - ph, 0), min(x2 + pw, w), min(y2 + ph, h), (x1, y1, x2, y2)


def gt_landmarks(kps):
    """COCO 17x3 -> MediaPipe-style 33 landmarks (x, y, visibility, z). v=2 -> visible, else invisible."""
    pts = [(0.0, 0.0, 0.0, 0.0)] * 33
    for j, mp_i in enumerate(COCO_TO_MP):
        x, y, v = kps[3 * j: 3 * j + 3]
        pts[mp_i] = (float(x), float(y), 1.0 if v == 2 else 0.0, 0.0)
    return pts


def kind(text):
    from humansense.result import cue_kind
    return cue_kind(text)


def main():
    import mediapipe as mp
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from humansense.modules import pose_detector as PD

    root = DATA / "coco"
    with zipfile.ZipFile(root / "annotations_trainval2017.zip") as zf:
        data = json.loads(zf.read("annotations/person_keypoints_val2017.json"))
    coco = COCO(); coco.dataset = data; coco.createIndex()
    zf = zipfile.ZipFile(root / "val2017.zip")
    lm = PD.get_landmarker()

    anns = [a for a in coco.loadAnns(coco.getAnnIds(catIds=[1], iscrowd=False))
            if a["num_keypoints"] >= MIN_KPS and a["bbox"][3] >= MIN_BOX_H]
    by_img = defaultdict(list)
    for a in anns:
        by_img[a["image_id"]].append(a)

    preds, cue_rows, detected, t0 = [], [], 0, time.time()
    for k, (iid, group) in enumerate(sorted(by_img.items())):
        info = coco.imgs[iid]
        img = cv2.imdecode(np.frombuffer(zf.read(f"val2017/{info['file_name']}"), np.uint8), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        for a in group:
            cx1, cy1, cx2, cy2, box = crop_box(a["bbox"], W, H)
            if cx2 - cx1 < 20 or cy2 - cy1 < 20:
                continue
            crop = np.ascontiguousarray(cv2.cvtColor(img[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB))
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
            gt_cues = {kind(c[0]) for c in PD._person_cues(gt_landmarks(a["keypoints"]), box)}
            if not res.pose_landmarks:
                cue_rows.append({"h": a["bbox"][3], "gt": sorted(gt_cues), "pred": [], "found": 0})
                continue
            detected += 1
            cw, ch = cx2 - cx1, cy2 - cy1
            pts = [(cx1 + p.x * cw, cy1 + p.y * ch, getattr(p, "visibility", 1.0), p.z * cw)
                   for p in res.pose_landmarks[0]]
            kp, vis = [], []
            for mp_i in COCO_TO_MP:
                x, y, v, _ = pts[mp_i]
                kp += [float(x), float(y), 1]
                vis.append(v)
            preds.append({"image_id": iid, "category_id": 1, "keypoints": kp, "score": float(np.mean(vis))})
            pred_cues = {kind(c[0]) for c in PD._person_cues(pts, box)}
            cue_rows.append({"h": a["bbox"][3], "gt": sorted(gt_cues), "pred": sorted(pred_cues), "found": 1})
        if (k + 1) % 200 == 0:
            print(f"{k + 1}/{len(by_img)} images ({time.time() - t0:.0f}s)", flush=True)

    out = {"benchmark": "COCO val2017 keypoints (GT boxes, MediaPipe lite)",
           "persons_evaluated": len(cue_rows), "pose_found_rate": detected / max(len(cue_rows), 1),
           "min_keypoints": MIN_KPS, "min_box_height_px": MIN_BOX_H}
    # OKS AP restricted to the evaluated persons (others would count as unmatched GT)
    keep_ids = {a["id"] for a in anns}
    sub = COCO(); sub.dataset = {**data, "annotations": [a for a in data["annotations"] if a["id"] in keep_ids]}
    sub.createIndex()
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(sub, sub.loadRes(preds), "keypoints")
        E.params.imgIds = sorted(by_img)
        E.evaluate(); E.accumulate(); E.summarize()
    out["oks"] = {n: float(E.stats[i]) for i, n in enumerate(
        ["AP", "AP50", "AP75", "AP_medium", "AP_large", "AR", "AR50", "AR75", "AR_medium", "AR_large"])}

    # cue fidelity
    kinds = ["raised_hand", "seated", "arms_crossed", "protective_posture", "fall", "fall_risk"]
    fid = {}
    for kd in kinds:
        tp = sum(kd in r["gt"] and kd in r["pred"] for r in cue_rows)
        fp = sum(kd not in r["gt"] and kd in r["pred"] for r in cue_rows)
        fn = sum(kd in r["gt"] and kd not in r["pred"] for r in cue_rows)
        p, rc, f1 = prf(tp, fp, fn)
        fid[kd] = {"reference_positives": tp + fn, "precision": p, "recall": rc, "f1": f1,
                   "tp": tp, "fp": fp, "fn": fn}
    out["cue_fidelity"] = fid
    save("pose", out)
    print(json.dumps({"oks": out["oks"], "found": out["pose_found_rate"]}, indent=1))
    for kd, v in fid.items():
        print(f"{kd:20s} ref+={v['reference_positives']:5d} P={v['precision']:.2f} R={v['recall']:.2f} F1={v['f1']:.2f}")


if __name__ == "__main__":
    main()
