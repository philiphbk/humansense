# 🏥 HumanSense

**Non-verbal cue detection for autonomous hospital systems**

A real-time computer vision app that detects human non-verbal communication cues — designed to help autonomous robots navigate crowded hospital environments more safely and intelligently.

---

## What It Does

HumanSense processes video footage and detects four categories of non-verbal cues simultaneously:

| Module | What it detects |
|--------|----------------|
| 👥 **Crowd Detection** | Person count, density level (low/medium/high), density heatmap |
| 📏 **Proximity Alerts** | Personal space violations between people, proximity to the robot |
| 🧍 **Pose Analysis** | Raised hands, fall detection, defensive/protective posture, seated vs. standing |
| 😶 **Facial Emotions** | Distress, alarm, agitation, sadness, comfort — mapped to hospital-relevant cues |

Everything runs locally — no cloud APIs, no data leaving your machine.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download model weights (one-time, needs internet)

```bash
python download_models.py
```

This downloads ~15MB of model files and caches them locally. After this, the app works fully offline.

### 3. Run the app

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

---

## Using the App

1. **Upload a video** — MP4, AVI, MOV, or MKV. Hospital corridor footage works best.
2. **Toggle detection modules** in the sidebar — you can enable/disable each one independently.
3. **Adjust the robot position** — drag the sliders to simulate where the robot is in the scene. The system will alert when people get too close.
4. **Watch the event log** — real-time alerts appear on the right panel as the video processes.
5. **Tune performance** — increase "Process every N frames" if your machine is slow.

---

## How the Detection Works

### Crowd Detection (`modules/crowd_detector.py`)
Uses **YOLOv8n** (the nano model — fast and accurate enough for people). Each person gets a bounding box. Density is calculated as persons per 100k pixels of frame area: low (<1.5), medium (<4.0), high (4+).

The **heatmap** uses Gaussian blurring centered on each person's position, then applies a JET colormap for the familiar blue-green-yellow-red visualization.

### Proximity Detection (`modules/proximity_detector.py`)
No depth camera required. Uses bounding box center distances as a proxy for real-world proximity. The threshold is 18% of frame width — roughly arm's length in a typical framing. Robot proximity uses a 27% bubble (1.5× larger).

Color coding: green = safe, orange = caution, red = warning.

### Pose Analysis (`modules/pose_detector.py`)
Uses **MediaPipe Pose** (33 body landmarks). Key logic:
- **Raised hand**: wrist Y-coordinate above shoulder Y-coordinate by >20px
- **Fall detection**: shoulder-to-hip vertical distance < 12% of frame height (person is horizontal)
- **Protective posture**: shoulder width < 40% of body height (hunching in)
- **Arms crossed**: wrists in opposite-side positions near the torso

### Facial Emotions (`modules/emotion_detector.py`)
Uses **MediaPipe Face Detection** to locate faces, then **HSEmotion ONNX** (EfficientNet-B0 trained on AffectNet) to classify 8 emotions. Emotions are mapped to hospital-relevant cues:
- Anger/Disgust → Agitation/Distress (priority 3 — high alert)
- Fear → Alarm (priority 4 — highest)
- Sadness → Needs attention (priority 2)
- Happiness/Neutral → Normal (priority 1)

---

## Architecture

```
humansense/
├── app.py                    # Streamlit UI + video processing loop
├── download_models.py        # One-time model downloader
├── requirements.txt
├── modules/
│   ├── crowd_detector.py     # YOLOv8 person detection + heatmap
│   ├── proximity_detector.py # Distance-based proximity alerts
│   ├── pose_detector.py      # MediaPipe pose landmark analysis
│   └── emotion_detector.py   # HSEmotion ONNX face emotion classifier
└── utils/
    └── frame_utils.py        # Frame compositing, status bar, event log
```

---

## Performance

On a modern laptop (M1/M2 Mac or recent Intel):
- Crowd + Proximity only: ~10–15 FPS
- All 4 modules: ~5–8 FPS
- Use "Process every N frames = 3" for smoother playback at full quality

For real-time deployment, consider running pose/emotion detection on every 3rd frame and crowd/proximity on every frame.

---

## Version 2 Ideas

Things worth adding in the next version:

- **Webcam/live camera support** — process a live feed, not just uploaded videos
- **ROS integration** — publish detected cues as ROS topics for actual robot consumption
- **Depth camera support** — use Intel RealSense or similar for accurate real-world distance measurement (replaces the pixel-distance heuristic)
- **Custom zone mapping** — define zones in the hospital (ICU, corridor, waiting room) with different alert thresholds
- **Action recognition** — distinguish between "walking fast" vs. "running" vs. "stumbling"
- **Multi-camera support** — stitch feeds from multiple hospital cameras into one unified view
- **Alert dashboard** — log events to a database, send Slack/email notifications for high-priority alerts
- **Model fine-tuning** — collect hospital-specific data and fine-tune the emotion/pose models for better accuracy in clinical settings

---

## Tech Stack

- [Streamlit](https://streamlit.io) — web UI framework
- [OpenCV](https://opencv.org) — video I/O and frame processing
- [MediaPipe](https://mediapipe.dev) — pose and face detection (Google)
- [Ultralytics YOLOv8](https://ultralytics.com) — person detection
- [HSEmotion ONNX](https://github.com/HSE-asavchenko/face-emotion-recognition) — facial emotion recognition
- [ONNX Runtime](https://onnxruntime.ai) — fast CPU inference for emotion model

---

Built with HumanSense v1.0
