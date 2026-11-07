# 🏥 HumanSense

**Non-verbal cue detection for autonomous hospital systems**

A real-time computer vision pipeline that detects human non-verbal communication cues — crowding,
personal-space violations, falls, and distress — designed to help autonomous robots navigate
crowded hospital environments more safely. Ships as a Python library/CLI, a Streamlit demo UI, and
a ROS 2 node that turns any camera topic into structured, planner-friendly messages.

Everything runs locally — no cloud APIs, no data leaving the machine.

---

## What It Does

| Module | What it detects |
|--------|----------------|
| 👥 **Crowd Detection** | Person count and density level (low/medium/high) |
| 📏 **Proximity Alerts** | Personal-space violations between people, and between people and the robot |
| 🧍 **Pose Analysis** | Raised hands, falls, fall risk, protective/defensive posture, seated vs. standing |
| 😶 **Facial Emotions** *(opt-in)* | Distress, alarm, agitation, sadness — mapped to hospital-relevant cues |

A risk-fusion stage combines these into a single 0–100 score and `ok`/`elevated`/`critical` level,
with the reasons that produced it.

---

## Quick Start

### 1. Install

```bash
pip install -e ".[app]"       # add ,yolo for the more accurate (AGPL-3.0) detector backend
```

### 2. Download model weights (one-time, needs internet)

```bash
humansense download-models
```

This caches the MediaPipe/ONNX model files locally (~15 MB). After this, everything runs offline.

### 3. Run it

```bash
# Headless, on a video file, image directory, or a live camera (see below)
humansense run --source demo.mp4 --jsonl -

# Streamlit demo UI (upload a video or use a live webcam)
humansense app
```

---

## Live Camera

`humansense run --source <spec>` accepts:

- a camera index (`0`, `1`, …) or device path (`/dev/video0`)
- a network stream (`rtsp://...`, `http://...`)
- a video file (`clip.mp4`) or a directory of images

```bash
humansense run --source 0 --display --robot 0.5 1.0
```

Live sources are handled by [`CameraSource`](src/humansense/sources.py): a dedicated grab thread
always keeps the *newest* frame, so a slow inference loop never falls seconds behind the camera's
own buffer — frames are dropped instead, and the drop count is reported, never hidden. It
auto-reconnects on device loss. `humansense run` prints delivered/dropped frame counts and
capture-to-result latency for any live source.

---

## ROS 2 Integration

[`ros2/`](ros2/) ships two packages: `humansense_msgs` (the message types) and `humansense_ros`
(the node). Subscribe it to any `sensor_msgs/Image` topic — your camera driver, a bag replay, or
the bundled test publisher:

```text
image  (sensor_msgs/Image, in)              ──▶  humansense_node  ──▶  ~/state        (humansense_msgs/HumanSenseState)
                                                                    ──▶  ~/risk         (humansense_msgs/Risk)
                                                                    ──▶  ~/detections   (vision_msgs/Detection2DArray)
                                                                    ──▶  ~/annotated    (sensor_msgs/Image, opt-in)
```

```bash
# Against a real camera topic
ros2 launch humansense_ros humansense.launch.py image_topic:=/camera/image_raw

# Self-contained demo: replays a video/image-dir as the camera topic
ros2 launch humansense_ros demo.launch.py source:=/path/to/video.mp4
```

Parameters (see [`config/humansense.yaml`](ros2/humansense_ros/config/humansense.yaml)): `backend`,
`modules`, `conf`, `max_width`, `robot_x`/`robot_y`, `calibration_path`, `publish_annotated`. All
published messages copy the input image's header, so `now - header.stamp` is the real sensing
latency. The node uses a depth-1, reliable QoS on the `image` topic: a newer frame still supersedes
an unacked older one locally (the same "always process the newest frame" policy as `CameraSource`),
but delivery of whatever *is* sent is retried instead of being silently lost to a single dropped
packet -- an uncompressed frame is many UDP fragments, and losing any one of them drops the whole
frame under best-effort (see [paper/REPORT.md](paper/REPORT.md) for the measurement that led to
this).

Build and run in Docker (ROS 2 Jazzy):

```bash
docker build -f docker/ros2.Dockerfile -t humansense-ros .      # add --build-arg WITH_YOLO=1 for the yolo backend
docker run --rm -it humansense-ros
scripts/ros2_e2e.sh <video_or_image_dir> result.json            # end-to-end smoke test
```

---

## How the Detection Works

### Crowd Detection ([`modules/crowd_detector.py`](src/humansense/modules/crowd_detector.py))

Person detection is pluggable (see [`detectors.py`](src/humansense/detectors.py)): MediaPipe
EfficientDet-Lite by default (Apache-2.0), or Ultralytics YOLOv8n (`backend=yolo`, more accurate,
AGPL-3.0, optional extra). Density is persons per 100k px² of frame area: low (<1.5), medium (<4.0),
high (4+), smoothed over a short window.

### Proximity Detection ([`modules/proximity_detector.py`](src/humansense/modules/proximity_detector.py))

No depth camera required by default. Distances are normalised by each person's on-screen height so
the threshold follows perspective; an optional floor-homography calibration converts to real
metres. Personal space is ~0.8 body heights (≈1.3 m); the robot's bubble is 1.5× larger. Violations
are debounced (2 frames on, 3 off) to avoid flicker.

### Pose Analysis ([`modules/pose_detector.py`](src/humansense/modules/pose_detector.py))

MediaPipe Pose landmarks drive: raised hand, fall / fall-risk (torso tilt from vertical), protective
posture, arms crossed, seated vs. standing. A horizontal posture only becomes "FALL detected" with
a recent burst of per-person tracked speed (`utils/tracking.py`) confirming an actual collapse
*event*; a posture that has been horizontal for a while with no such burst (e.g. someone already
resting on a bed or couch) stays at the weaker "fall risk" cue instead -- tuned and evaluated on the
URFD dataset, see [paper/REPORT.md](paper/REPORT.md). The same tracked speed also distinguishes a
fall followed by no movement from a quick recovery (see `modules/risk.py`).

### Facial Emotions ([`modules/emotion_detector.py`](src/humansense/modules/emotion_detector.py)) — opt-in

MediaPipe face detection + HSEmotion ONNX (EfficientNet-B0 trained on AffectNet) classify 8
emotions, mapped to hospital-relevant cues (anger/disgust → agitation, fear → alarm, sadness →
needs attention). Excluded from the default module set — see [docs/ETHICS.md](docs/ETHICS.md) for
why.

### Risk Fusion ([`modules/risk.py`](src/humansense/modules/risk.py))

Combines the above into one 0–100 score: a single weak cue stays low, independent cues that agree
(a fall, with no movement, near the robot) add up.

---

## Architecture

```text
src/humansense/
├── cli.py              # `humansense` CLI: run, download-models, app
├── pipeline.py          # Pipeline / PipelineConfig — one BGR frame in, one FrameResult out
├── sources.py           # FrameSource: live camera, video file, image sequence
├── detectors.py         # pluggable person-detector backends (yolo / mediapipe)
├── result.py            # typed, JSON-serialisable FrameResult / Person / Cue / Risk
├── models.py             # model download/cache management
├── modules/
│   ├── crowd_detector.py
│   ├── proximity_detector.py
│   ├── pose_detector.py
│   ├── emotion_detector.py
│   └── risk.py
├── utils/
│   ├── tracking.py       # PersonTracker, Debouncer
│   ├── calibration.py    # floor homography / no-fall zones
│   ├── privacy.py        # e.g. blur_heads for recorded output
│   └── event_store.py    # local SQLite event log
└── app/streamlit_app.py  # demo UI (upload a video or use a live webcam)

ros2/
├── humansense_msgs/      # Cue, Person, Risk, Violation, HumanSenseState
└── humansense_ros/       # humansense_node, video_publisher, launch files, config
```

---

## Performance

Reproducible numbers are in [`benchmarks/`](benchmarks/README.md), `benchmarks/results/*.json`, and
[paper/REPORT.md](paper/REPORT.md). From `bench_latency.py` (mediapipe backend, 720x480,
`assets/demo.mp4`, containerized Linux/x86_64 -- see the report for the exact environment and the
caveats that come with it): person detection alone runs at a 25.3 ms/frame median; the ROS node's
default module set (crowd + proximity + pose) at 35.3 ms median, 37.3 ms p95; adding emotion costs
another ~2 ms median. Pose is the next-most-expensive stage after detection (~10 ms median) --
the ROS node and CLI both report per-frame timings (`timings_ms`) per module so you can decide what
to run on every frame versus every Nth on your own hardware.

---

## Responsible Use & Licensing

Read [docs/ETHICS.md](docs/ETHICS.md) before deploying this on real people: it is a research
prototype, not a certified safety function, and emotion recognition in particular carries legal
and bias caveats. [docs/LICENSING.md](docs/LICENSING.md) covers what each optional component
(YOLOv8, HSEmotion weights, benchmark datasets) adds to your obligations — HumanSense's own code is
Apache-2.0.

---

## Roadmap

Things worth adding next:

- **Depth camera support** — Intel RealSense or similar, for proximity distances that don't rely on
  a floor-homography calibration
- **Custom zone mapping** — per-zone alert thresholds (ICU, corridor, waiting room)
- **Action recognition** — distinguish "walking fast" vs. "running" vs. "stumbling"
- **Multi-camera fusion** — one `Pipeline` per camera today ([pipeline.py](src/humansense/pipeline.py)
  limitation); no cross-camera identity/view stitching yet
- **Alert dashboard** — the local SQLite event log ([utils/event_store.py](src/humansense/utils/event_store.py))
  has no UI yet; Slack/email notification hooks
- **Model fine-tuning** — collect hospital-specific data and fine-tune pose/emotion models for
  clinical settings

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Tech Stack

- [OpenCV](https://opencv.org) — video I/O and frame processing
- [MediaPipe](https://mediapipe.dev) — pose, face detection, and the default person-detector backend
- [Ultralytics YOLOv8](https://ultralytics.com) — optional, more accurate person-detector backend
- [HSEmotion ONNX](https://github.com/HSE-asavchenko/face-emotion-recognition) — facial emotion recognition
- [ONNX Runtime](https://onnxruntime.ai) — fast CPU inference for the emotion model
- [Streamlit](https://streamlit.io) — demo UI
- [ROS 2](https://www.ros.org) (Jazzy) — robot integration

---

Built with HumanSense v2.0
