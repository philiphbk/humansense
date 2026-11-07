"""
Facial-emotion module on FER2013 (official private-test split, 3,589 images, 7 classes).

HSEmotion (EfficientNet-B0, trained on AffectNet) is applied directly to the aligned 48x48 faces
(HumanSense's MediaPipe face detector is bypassed: it is not meant for 48-px crops). Because the
model has an extra 'Contempt' class, predictions are the arg-max over the 7 shared classes.

Also reported: (a) the binary 'alarm cue' HumanSense actually uses (Anger/Disgust/Fear = priority>=3),
(b) accuracy with the lower face blacked out, a crude proxy for a surgical mask.

    python benchmarks/bench_emotion.py
"""
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, prf, save  # noqa: E402

FER = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]   # dataset label order
ALARM = {"Anger", "Disgust", "Fear"}                                               # priority >= 3 in HumanSense


def main():
    from humansense.modules.emotion_detector import get_emotion_recognizer
    rec = get_emotion_recognizer()
    classes = {v: k for k, v in rec.idx_to_class.items()}          # name -> model index
    cols = [classes[n] for n in FER]

    df = pd.read_parquet(DATA / "fer2013" / "privateTest.parquet")
    y = df["label"].to_numpy()

    def run(transform=None):
        pred, t0 = [], time.time()
        for i, rowimg in enumerate(df["image"]):
            im = np.array(Image.open(io.BytesIO(rowimg["bytes"])).convert("RGB"))
            if transform:
                im = transform(im)
            _, scores = rec.predict_emotions(im, logits=False)
            pred.append(int(np.argmax(np.asarray(scores)[cols])))
            if (i + 1) % 1000 == 0:
                print(f"{i + 1}/{len(df)} ({time.time() - t0:.0f}s)", flush=True)
        return np.array(pred)

    def metrics(pred):
        cm = np.zeros((7, 7), int)
        for t, p in zip(y, pred):
            cm[t, p] += 1
        per = {}
        f1s = []
        for c, n in enumerate(FER):
            p, r, f = prf(cm[c, c], cm[:, c].sum() - cm[c, c], cm[c].sum() - cm[c, c])
            per[n] = {"support": int(cm[c].sum()), "precision": p, "recall": r, "f1": f}
            f1s.append(f)
        alarm_idx = [FER.index(n) for n in ALARM]
        t_pos, p_pos = np.isin(y, alarm_idx), np.isin(pred, alarm_idx)
        ap, ar, af = prf(int((t_pos & p_pos).sum()), int((~t_pos & p_pos).sum()), int((t_pos & ~p_pos).sum()))
        return {"accuracy": float((pred == y).mean()), "macro_f1": float(np.nanmean(f1s)),
                "per_class": per, "confusion_matrix": cm.tolist(), "labels": FER,
                "alarm_cue": {"definition": "Anger|Disgust|Fear vs the rest", "precision": ap,
                              "recall": ar, "f1": af, "prevalence": float(t_pos.mean())}}

    def lower_face_masked(im):
        im = im.copy()
        im[int(im.shape[0] * 0.55):] = 0
        return im

    out = {"benchmark": "FER2013 private test (3589 imgs), HSEmotion enet_b0_8_best_vgaf",
           "n": int(len(df)), "full_face": metrics(run()),
           "lower_face_occluded": metrics(run(lower_face_masked)),
           "majority_class_accuracy": float(np.bincount(y).max() / len(y))}
    save("emotion", out)
    for k in ("full_face", "lower_face_occluded"):
        m = out[k]
        print(f"{k}: acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} alarmF1={m['alarm_cue']['f1']:.3f}")


if __name__ == "__main__":
    main()
