"""
HumanSense v2.0 — Non-verbal Cue Detection for Autonomous Hospital Systems
────────────────────────────────────────────────────────────────────────────
Real-time computer vision that detects four categories of human non-verbal cues:
  • Crowd density & person tracking  (YOLOv8)
  • Proximity / personal space alerts
  • Pose-based cues — raised hand, fall, defensive posture  (MediaPipe Tasks)
  • Facial emotion recognition  (HSEmotion ONNX)

Phase 4 additions:
  • Live webcam mode
  • Auto-playing demo video
  • FPS counter & adaptive frame skip
  • Toast alerts for high-priority events
  • Graceful error handling throughout
"""

import os
import tempfile
import time
from collections import deque

import cv2
import numpy as np
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HumanSense",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0e1117; }
  .block-container { padding-top: 1.2rem; padding-bottom: 0.5rem; }

  /* ── Header ── */
  .hs-header {
    background: linear-gradient(135deg,#1a1f2e 0%,#0d1117 100%);
    border:1px solid #2d3748; border-radius:12px;
    padding:1rem 1.4rem; margin-bottom:0.8rem;
    display:flex; align-items:center; gap:1rem;
  }
  .hs-title   { font-size:1.7rem; font-weight:700; color:#e2e8f0; margin:0; }
  .hs-subtitle{ font-size:0.82rem; color:#718096; margin:0; }
  .hs-version { background:#2d3748; border-radius:20px; padding:0.2rem 0.7rem;
                font-size:0.7rem; color:#68d391; border:1px solid #38a169; }
  .hs-live    { background:#2d1a1a; border-radius:20px; padding:0.2rem 0.7rem;
                font-size:0.7rem; color:#fc8181; border:1px solid #c53030;
                animation:pulse 1.8s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.55} }

  /* ── Metric cards ── */
  .metric-row { display:flex; gap:6px; margin-bottom:6px; }
  .metric-card {
    background:#1a202c; border:1px solid #2d3748; border-radius:8px;
    padding:0.6rem 0.5rem; text-align:center; flex:1; min-width:0;
  }
  .metric-val { font-size:1.35rem; font-weight:700; line-height:1.1; }
  .metric-lbl { font-size:0.62rem; color:#718096; text-transform:uppercase;
                letter-spacing:.06em; margin-top:2px; }
  .c-ok  { color:#68d391; }
  .c-cau { color:#f6ad55; }
  .c-wrn { color:#fc8181; }
  .c-inf { color:#63b3ed; }
  .c-dim { color:#a0aec0; }

  /* ── Event log ── */
  .ev-log {
    background:#1a202c; border:1px solid #2d3748; border-radius:8px;
    padding:0.4rem; font-size:0.72rem; font-family:monospace;
    max-height:190px; overflow-y:auto;
  }
  .ev-row { padding:2px 4px; border-bottom:1px solid #2d374830; color:#a0aec0; }
  .ev-row:last-child{ border-bottom:none; }
  .ev-t { color:#4a5568; margin-right:6px; }
  .ev-w { color:#fc8181; } .ev-c { color:#f6ad55; } .ev-i { color:#63b3ed; }

  div[data-testid="stRadio"] label { font-size:0.85rem !important; }
  div[data-testid="stImage"] img   { border-radius:8px; }
</style>
""", unsafe_allow_html=True)


# ── Load the pipeline (cached across reruns, one per detector backend) ───────
@st.cache_resource(show_spinner="Loading AI models — first run may take ~30 s…")
def load_pipeline(backend: str):
    from humansense import Pipeline, PipelineConfig
    from humansense.utils.frame_utils import get_event_log, log_event
    return Pipeline(PipelineConfig(backend=backend, draw=True)), log_event, get_event_log


# ── Session state defaults ────────────────────────────────────────────────────
for _k, _v in [("running", False), ("fps_history", deque(maxlen=20))]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️  HumanSense")

    st.markdown("##### Input Source")
    source = st.radio(
        "source",
        ["🎬  Demo video", "📁  Upload video", "📷  Live camera"],
        label_visibility="collapsed",
    )

    camera_src = "0"
    if "camera" in source.lower():
        camera_src = st.text_input("Camera index or URL", "0",
                                   help="0 = default webcam, or an rtsp:// / http:// stream URL")

    uploaded_file = None
    if "Upload" in source:
        uploaded_file = st.file_uploader(
            "Choose a video file", type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
        )

    st.divider()
    st.markdown("##### Person detector")
    backend = st.selectbox("Backend", ["auto", "mediapipe", "yolo"],
                           help="mediapipe: Apache-2.0, installed by default. "
                                "yolo: more accurate, needs `pip install humansense[yolo]` (AGPL-3.0).")

    st.divider()
    st.markdown("##### Detection Modules")
    show_crowd     = st.checkbox("👥  Crowd detection",   value=True)
    show_heatmap   = st.checkbox("🌡️  Density heatmap",  value=True)
    show_proximity = st.checkbox("📏  Proximity alerts",  value=True)
    show_pose      = st.checkbox("🧍  Pose analysis",     value=True)
    show_emotions  = st.checkbox("😶  Facial emotions",   value=True)

    st.divider()
    st.markdown("##### Robot Simulation")
    robot_enabled = st.checkbox("Show robot position", value=True)
    robot_pos = None
    if robot_enabled:
        rx = st.slider("X position", 0.0, 1.0, 0.65, 0.01)
        ry = st.slider("Y position", 0.0, 1.0, 0.85, 0.01)
        robot_pos = (rx, ry)

    st.divider()
    st.markdown("##### Privacy")
    blur_faces = st.checkbox("Blur heads in output", value=False,
                             help="Pixelates the top of every detected person")

    st.divider()
    st.markdown("##### Performance")
    conf_thr   = st.slider("Detection confidence", 0.25, 0.85, 0.40, 0.05)
    frame_skip = st.slider("Process every N frames", 1, 6, 2,
                            help="Higher = faster but choppier")
    max_width  = st.select_slider("Max frame width",
                                   options=[480, 640, 720, 960], value=720)

    st.divider()
    st.caption("HumanSense v2.0  ·  YOLOv8 / EfficientDet · MediaPipe · HSEmotion ONNX")


# ── Header ────────────────────────────────────────────────────────────────────
badge = ('<span class="hs-live">● LIVE</span>' if st.session_state.running
         else '<span class="hs-version">v2.0</span>')
st.markdown(f"""
<div class="hs-header">
  <div style="font-size:2rem">🏥</div>
  <div style="flex:1">
    <p class="hs-title">HumanSense</p>
    <p class="hs-subtitle">Non-verbal cue detection for autonomous hospital systems</p>
  </div>
  {badge}
</div>
""", unsafe_allow_html=True)

try:
    PIPE, log_event, get_event_log = load_pipeline(backend)
except Exception as _e:  # noqa: BLE001 - surface any load failure to the UI instead of crashing
    st.error(f"**Model load failed:** {_e}")
    st.info("Run `humansense download-models` (needs internet once), then restart the app. "
            "If you picked the yolo backend: `pip install 'humansense[yolo]'`.")
    st.stop()


# ── Main layout ───────────────────────────────────────────────────────────────
col_vid, col_panel = st.columns([3, 1], gap="medium")

with col_vid:
    vid_ph    = st.empty()
    status_ph = st.empty()

with col_panel:
    st.markdown("#### 📊 Metrics")
    m_people    = st.empty()
    m_density   = st.empty()
    m_proximity = st.empty()
    m_pose      = st.empty()
    m_emotion   = st.empty()
    m_fps       = st.empty()
    st.markdown("---")
    st.markdown("#### 📋 Events")
    log_ph = st.empty()


# ── Render helpers ────────────────────────────────────────────────────────────
def _card(val, lbl, cls="c-dim"):
    return (f'<div class="metric-card">'
            f'<div class="metric-val {cls}">{val}</div>'
            f'<div class="metric-lbl">{lbl}</div></div>')

def _row(*cards):
    return f'<div class="metric-row">{"".join(cards)}</div>'


def update_metrics(results, fps_val):
    crowd  = results.get("crowd",     {})
    prox   = results.get("proximity", {})
    pose   = results.get("pose",      {})
    emo    = results.get("emotions",  {})

    count   = crowd.get("person_count",  0)
    density = crowd.get("density_level", "—")
    alert   = prox.get("alert_level",    "none")
    cues    = pose.get("cues",           [])
    top_emo = emo.get("highest_priority_cue")

    d_cls = {"low":"c-ok","medium":"c-cau","high":"c-wrn"}.get(density, "c-dim")
    p_cls = {"none":"c-ok","caution":"c-cau","warning":"c-wrn"}.get(alert, "c-dim")
    e_cls = ("c-wrn" if top_emo and top_emo["priority"] >= 3
             else "c-cau" if top_emo and top_emo["priority"] == 2
             else "c-ok")

    m_people   .markdown(_row(_card(count,          "People",   "c-inf")),   unsafe_allow_html=True)
    m_density  .markdown(_row(_card(density.upper(),"Density",  d_cls)),     unsafe_allow_html=True)
    m_proximity.markdown(_row(_card(alert.upper(),  "Proximity",p_cls)),     unsafe_allow_html=True)
    m_pose     .markdown(_row(_card(len(cues),      "Pose Cues","c-cau" if cues else "c-ok")),
                         unsafe_allow_html=True)
    emo_lbl = top_emo["label"] if top_emo else "OK"
    m_emotion  .markdown(_row(_card(emo_lbl,        "Emotion",  e_cls)),     unsafe_allow_html=True)
    m_fps      .markdown(_row(_card(f"{fps_val:.0f}","FPS",     "c-dim")),   unsafe_allow_html=True)
    log_ph.markdown(_render_log(get_event_log()), unsafe_allow_html=True)


def _render_log(events):
    if not events:
        return ('<div class="ev-log"><div class="ev-row" style="color:#4a5568">'
                'No events yet…</div></div>')
    rows = "".join(
        f'<div class="ev-row"><span class="ev-t">{e["time"]}</span>'
        f'<span class="{"ev-w" if e["type"]=="warning" else "ev-c" if e["type"]=="caution" else "ev-i"}">'
        f'{e["detail"]}</span></div>'
        for e in events[:14]
    )
    return f'<div class="ev-log">{rows}</div>'


def _show_placeholder_metrics():
    for ph in [m_people, m_density, m_proximity, m_pose, m_emotion, m_fps]:
        ph.markdown(_row(_card("—", "—", "c-dim")), unsafe_allow_html=True)
    log_ph.markdown(_render_log([]), unsafe_allow_html=True)


# ── Toast alerts (non-blocking) ───────────────────────────────────────────────
def _maybe_toast(results):
    now  = time.time()
    last = st.session_state.get("_last_toast", {})

    def _toast(key, msg, icon):
        if now - last.get(key, 0) > 4.0:
            try:
                st.toast(msg, icon=icon)
            except Exception:  # noqa: BLE001, S110 - a toast failing must never break rendering
                pass
            last[key] = now

    crowd = results.get("crowd", {})
    prox  = results.get("proximity", {})
    pose  = results.get("pose", {})

    if crowd.get("density_level") == "high":
        _toast("density",   "High crowd density!",             "🚨")
    if prox.get("alert_level") == "warning":
        _toast("proximity", "Proximity warning — path blocked!","⚠️")
    if results.get("risk", {}).get("level") == "critical":
        _toast("risk",      "Critical risk — " + "; ".join(results["risk"]["reasons"][:2]), "🚨")
    if any("FALL" in c.upper() for c in pose.get("cues", [])):
        _toast("fall",      "Fall detected!",                  "🆘")

    st.session_state["_last_toast"] = last


# ── Core frame processor ──────────────────────────────────────────────────────
def _build_settings():
    mods = [m for m, on in (("crowd", show_crowd), ("proximity", show_proximity),
                            ("pose", show_pose), ("emotion", show_emotions)) if on]
    return {"modules": mods or ["crowd"], "robot_pos": robot_pos, "conf": conf_thr,
            "frame_skip": frame_skip, "max_width": max_width, "blur_faces": blur_faces,
            "show_heatmap": show_heatmap}


def _log_result(res):
    raw = res.raw
    cr = raw.get("crowd", {})
    if cr.get("density_level") == "high":
        log_event("warning", f"High density — {cr['person_count']} people")
    elif cr.get("density_level") == "medium":
        log_event("caution", f"Medium density — {cr['person_count']} people")
    pr = raw.get("proximity")
    if pr and pr["alert_level"] != "none":
        log_event("warning" if pr["alert_level"] == "warning" else "caution",
                  f"Proximity {pr['alert_level'].upper()} — {pr['violation_count']} violation(s)")
    for cue in raw.get("pose", {}).get("cues", []):
        log_event("warning" if "FALL" in cue.upper() else "caution", f"Pose: {cue}")
    top = raw.get("emotions", {}).get("highest_priority_cue")
    if top and top["priority"] >= 3:
        log_event("warning", f"Emotion alert — {top['label']}")
    elif top and top["priority"] == 2:
        log_event("caution", f"Emotion — {top['label']}")
    if res.risk.level != "ok":
        log_event("warning" if res.risk.level == "critical" else "caution",
                  f"Risk {res.risk.score} — " + "; ".join(res.risk.reasons[:3]))
    for err in res.errors:
        if not st.session_state.get("_err_shown"):
            st.session_state["_err_shown"] = True
            status_ph.warning(f"Module problem: {err}")


def process_frame(frame, cfg, timestamp=None):
    from dataclasses import replace

    from humansense.utils.privacy import blur_heads
    PIPE.cfg = replace(PIPE.cfg, modules=tuple(cfg["modules"]), conf=cfg["conf"],
                       robot_position=cfg["robot_pos"], max_width=cfg["max_width"],
                       show_heatmap=cfg["show_heatmap"], draw=True)
    res = PIPE.process(frame, timestamp=timestamp)
    _log_result(res)
    final = res.annotated
    if cfg["blur_faces"]:
        final = blur_heads(final, res.raw.get("detections", []))
    return final, res.raw


# ── Video loop (Upload & Demo) ────────────────────────────────────────────────
def run_video(path: str, loop: bool = False, label: str = ""):
    cfg   = _build_settings()
    cap   = cv2.VideoCapture(path)
    if not cap.isOpened():
        status_ph.error(f"Cannot open: {path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    sc, pc = st.columns([1, 4])
    stop   = sc.button("⏹  Stop", key=f"stop_{id(path)}")
    prog   = pc.progress(0, text=label or "Processing…")

    st.session_state.running = True
    PIPE.reset()
    fps_hist = deque(maxlen=20)
    f_idx    = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            f_idx += 1
            if f_idx % cfg["frame_skip"] != 0:
                continue

            t0    = time.perf_counter()
            final, results = process_frame(frame, cfg)
            fps_val = 1.0 / max(time.perf_counter() - t0, 1e-6)
            fps_hist.append(fps_val)
            avg_fps = float(np.mean(fps_hist))

            vid_ph.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB), width="stretch")
            update_metrics(results, avg_fps)
            _maybe_toast(results)

            if total > 0:
                prog.progress(min(f_idx / total, 1.0),
                              text=f"{label}  ·  {f_idx}/{total}  ·  {avg_fps:.1f} fps")
            if stop:
                break
    finally:
        cap.release()
        st.session_state.running = False
        prog.empty()
        status_ph.success(f"✅ Done — {f_idx} frames processed")


# ── Live camera loop ──────────────────────────────────────────────────────────
def run_webcam(device="0"):
    from humansense.sources import CameraSource
    cfg = _build_settings()
    try:
        cam = CameraSource(device, width=1280, height=720)
    except OSError as e:
        status_ph.error(str(e))
        _show_placeholder_metrics()
        return

    stop     = status_ph.button("⏹  Stop camera", key="stop_cam")
    fps_hist = deque(maxlen=20)
    st.session_state.running = True
    PIPE.reset()

    try:
        while True:
            frame = cam.read(timeout=3.0)   # always the newest frame — no stale-buffer lag
            if frame is None:
                status_ph.warning("No frames from the camera — retrying…")
                if cam._lost:
                    break
                continue

            t0 = time.perf_counter()
            final, results = process_frame(frame.image, cfg, timestamp=frame.timestamp)
            fps_hist.append(1.0 / max(time.perf_counter() - t0, 1e-6))

            vid_ph.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB), width="stretch")
            update_metrics(results, float(np.mean(fps_hist)))
            _maybe_toast(results)
            if stop:
                break
    finally:
        cam.close()
        st.session_state.running = False
        status_ph.info(f"Camera stopped — {cam.delivered} frames processed, "
                       f"{cam.dropped} skipped to stay real-time.")


# ── Idle / welcome screens ────────────────────────────────────────────────────
def show_idle(mode: str):
    _show_placeholder_metrics()

    DEMO_PATH = os.path.join(os.getcwd(), "assets", "demo.mp4")

    if "Demo" in mode:
        if os.path.exists(DEMO_PATH):
            cap = cv2.VideoCapture(DEMO_PATH)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
            ret, frame = cap.read()
            cap.release()
            if ret:
                vid_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                             width="stretch",
                             caption="Demo preview — press ▶ Run Demo to start")
            status_ph.info("🎬 Synthetic hospital corridor — press **▶ Run Demo** below")
        else:
            status_ph.warning("Demo video missing. Run `python scripts/generate_demo.py` first.")

    elif "Upload" in mode:
        vid_ph.markdown("""
        <div style="background:#1a202c;border:1px dashed #2d3748;border-radius:8px;
                    padding:3.5rem;text-align:center;color:#4a5568;">
          <div style="font-size:2.5rem;margin-bottom:.6rem">📹</div>
          <div style="font-size:.95rem">Upload a video in the sidebar to begin</div>
          <div style="font-size:.75rem;margin-top:.4rem;color:#2d3748">MP4 · AVI · MOV · MKV</div>
        </div>""", unsafe_allow_html=True)
        status_ph.info("📁 Use the sidebar to upload a video file.")

    else:  # camera
        vid_ph.markdown("""
        <div style="background:#1a202c;border:1px dashed #2d3748;border-radius:8px;
                    padding:3.5rem;text-align:center;color:#4a5568;">
          <div style="font-size:2.5rem;margin-bottom:.6rem">📷</div>
          <div style="font-size:.95rem">Live camera feed will appear here</div>
          <div style="font-size:.75rem;margin-top:.4rem;color:#2d3748">
            Grant camera permission if prompted</div>
        </div>""", unsafe_allow_html=True)
        status_ph.info("📷 Press **▶ Start Camera** to begin live detection.")


# ── Route by source ───────────────────────────────────────────────────────────
DEMO_PATH = os.path.join(os.getcwd(), "assets", "demo.mp4")

if "Demo" in source:
    show_idle(source)
    if os.path.exists(DEMO_PATH):
        if st.button("▶  Run Demo", type="primary", use_container_width=False):
            run_video(DEMO_PATH, loop=False, label="🎬 Hospital corridor demo")
    else:
        st.warning("Run `python scripts/generate_demo.py` to create the demo video first.")

elif "Upload" in source:
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        run_video(tmp_path, loop=False, label=f"📁 {uploaded_file.name}")
    else:
        show_idle(source)

else:  # Live camera
    show_idle(source)
    if st.button("▶  Start Camera", type="primary", use_container_width=False):
        run_webcam(camera_src)
