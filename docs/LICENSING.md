# Licensing of HumanSense and what it depends on

HumanSense's own code is **Apache-2.0** (see [LICENSE](../LICENSE)). It does not ship any model
weights or datasets. What you can do with a *deployment* depends on which components you enable:

| Component | Used for | License | Notes |
|---|---|---|---|
| HumanSense code | everything | Apache-2.0 | |
| MediaPipe (library) | pose, face detection, `mediapipe` person backend | Apache-2.0 | |
| MediaPipe model files (pose landmarker, BlazeFace, EfficientDet-Lite) | same | see Google's model cards | Downloaded by `humansense download-models`; check the model cards for the terms of the exact files you use. |
| **Ultralytics YOLOv8** (`yolo` backend) | person detection | **AGPL-3.0** (code and weights) | Optional extra (`pip install humansense[yolo]`), never imported unless you select it. If you distribute or network-serve a product that includes it, AGPL obligations apply to that product; Ultralytics sells an enterprise license. The default-installed `mediapipe` backend avoids this. |
| HSEmotion `enet_b0_8_best_vgaf` | facial emotion (opt-in) | code Apache-2.0; **weights trained on AffectNet** | AffectNet is licensed for non-commercial research. Treat the emotion module as research-only unless you obtain appropriate rights. |
| ONNX Runtime, NumPy, SciPy, OpenCV | runtime | MIT / BSD / Apache-2.0 | |

## Datasets used by the benchmarks (not redistributed)

| Dataset | License / terms |
|---|---|
| UR Fall Detection (URFD) | CC BY-NC-SA 4.0, non-commercial academic use |
| COCO 2017 | annotations CC BY 4.0; images under their Flickr terms |
| FER2013 | Kaggle challenge data; used for research evaluation only |

The benchmark scripts download nothing themselves; see [benchmarks/README.md](../benchmarks/README.md).

This document is engineering guidance, not legal advice.
