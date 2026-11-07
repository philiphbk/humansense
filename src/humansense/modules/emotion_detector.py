"""
Stage 5 — Facial Emotion Detection Module
Uses MediaPipe Tasks API (FaceDetector) — compatible with mediapipe >= 0.10.30.
Then routes face crops through HSEmotion ONNX for emotion classification.
No TensorFlow required.

Emotions → hospital non-verbal cues:
  Anger / Disgust  → Agitation / Distress
  Fear             → Alarm
  Sadness          → Needs attention
  Surprise         → Startled
  Neutral/Happiness → Normal
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

from ..models import ensure_model

# ── Model path ───────────────────────────────────────────────────────────────
FACE_MODEL_NAME = "blaze_face_short_range.tflite"

# ── Emotion metadata ─────────────────────────────────────────────────────────
EMOTION_MAP = {
    "Anger":    {"label": "Agitation",       "color": (0,   0,  200), "priority": 3},
    "Disgust":  {"label": "Distress",        "color": (0,  80,  220), "priority": 3},
    "Fear":     {"label": "Alarm",           "color": (0,   0,  220), "priority": 4},
    "Happiness":{"label": "Comfortable",     "color": (50, 200,  50), "priority": 1},
    "Neutral":  {"label": "Neutral",         "color": (150,150, 150), "priority": 1},
    "Sadness":  {"label": "Needs attention", "color": (0,  165, 255), "priority": 2},
    "Surprise": {"label": "Startled",        "color": (0,  200, 200), "priority": 2},
    "Contempt": {"label": "Discomfort",      "color": (80, 120, 200), "priority": 2},
}

# Singleton instances
_face_detector      = None
_emotion_recognizer = None


def get_face_detector():
    global _face_detector
    if _face_detector is None:
        options = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model(FACE_MODEL_NAME)),
            running_mode=mp_vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
        _face_detector = mp_vision.FaceDetector.create_from_options(options)
    return _face_detector


def get_emotion_recognizer():
    global _emotion_recognizer
    if _emotion_recognizer is None:
        ensure_model("enet_b0_8_best_vgaf.onnx")   # hsemotion's own downloader is unreliable
        _emotion_recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf')
    return _emotion_recognizer


def detect_emotions(frame: np.ndarray, draw: bool = True):
    """
    Detect faces and classify emotions.
    Returns:
      - annotated_frame
      - face_results: list of {bbox, emotion, label, confidence, priority}
      - highest_priority_cue: the most urgent emotional cue in the frame
    """
    try:
        face_det = get_face_detector()
    except (OSError, RuntimeError):
        annotated = frame.copy() if draw else None
        if draw:
            cv2.putText(annotated, "Face model unavailable — run: humansense download-models",
                        (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1, cv2.LINE_AA)
        return {"annotated_frame": annotated, "face_results": [], "highest_priority_cue": None}

    emotion_rec = get_emotion_recognizer()
    h, w       = frame.shape[:2]

    # Tasks API needs an mp.Image in SRGB format
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_det.detect(mp_img)

    annotated    = frame.copy() if draw else None
    face_results = []

    if not result.detections:
        if draw:
            cv2.putText(annotated, "  No faces detected  ", (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        return {"annotated_frame": annotated, "face_results": [], "highest_priority_cue": None}

    for detection in result.detections:
        # New API: bounding_box is already in pixel coordinates
        bb   = detection.bounding_box
        pad  = 10
        x1   = max(0,  bb.origin_x - pad)
        y1   = max(0,  bb.origin_y - pad)
        x2   = min(w,  bb.origin_x + bb.width  + pad)
        y2   = min(h,  bb.origin_y + bb.height + pad)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0 or face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
            continue

        try:
            emotion, scores = emotion_rec.predict_emotions(face_crop, logits=False)
            confidence = float(np.max(scores))
        except Exception:
            emotion    = "Neutral"
            confidence = 0.5

        meta = EMOTION_MAP.get(emotion, EMOTION_MAP["Neutral"])
        face_results.append({
            "bbox":       (x1, y1, x2, y2),
            "emotion":    emotion,
            "label":      meta["label"],
            "color":      meta["color"],
            "confidence": confidence,
            "priority":   meta["priority"],
        })
        if draw:
            _draw_face_annotation(annotated, x1, y1, x2, y2,
                                  meta["label"], emotion, confidence, meta["color"])

    face_results.sort(key=lambda x: x["priority"], reverse=True)
    highest = face_results[0] if face_results else None
    if draw:
        _draw_emotion_summary(annotated, face_results)

    return {
        "annotated_frame":      annotated,
        "face_results":         face_results,
        "highest_priority_cue": highest,
    }


# ── Drawing helpers ──────────────────────────────────────────────────────────

def _draw_face_annotation(frame, x1, y1, x2, y2, label, raw_emotion, confidence, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    tag = f"{label} ({confidence:.0%})"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), (20, 20, 20), -1)
    cv2.putText(frame, tag, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_emotion_summary(frame, face_results):
    y_off = 105
    for i, face in enumerate(face_results[:3]):
        text  = f"  Face {i+1}: {face['label']} ({face['emotion']})  "
        color = face["color"]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (8, y_off), (8 + tw + 4, y_off + th + 10), (20, 20, 20), -1)
        cv2.putText(frame, text, (10, y_off + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        y_off += th + 14
