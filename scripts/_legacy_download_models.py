"""
Run this script once before starting HumanSense to download all model weights.
Requires an internet connection (~30 MB total). After this, the app works offline.

Handles macOS SSL certificate issues automatically.

Usage:
    python download_models.py
"""

import sys
import os
import ssl
import urllib.request
import subprocess
import shutil

print("HumanSense — Model Downloader")
print("=" * 40)

# ── Directories ───────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "assets", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# HSEmotion caches models here by default
HSEMOTION_CACHE = os.path.join(os.path.expanduser("~"), ".hsemotion")
ONNX_MODEL_NAME = "enet_b0_8_best_vgaf.onnx"
ONNX_DEST       = os.path.join(HSEMOTION_CACHE, ONNX_MODEL_NAME)


# ── SSL context builder ───────────────────────────────────────────────────────
def _make_ssl_context():
    """
    Try to build a verified SSL context in order of preference:
      1. certifi bundle (most reliable on macOS)
      2. default system context
    Returns (context, label) or (None, label) if unverified fallback needed.
    """
    # Try certifi first
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        return ctx, "certifi"
    except ImportError:
        pass

    # Try default context
    try:
        ctx = ssl.create_default_context()
        return ctx, "system"
    except Exception:
        pass

    return None, "unverified"


def _ssl_opener(ctx):
    """Build a urllib opener using the given SSL context (or unverified)."""
    if ctx is None:
        ctx = ssl._create_unverified_context()
    handler = urllib.request.HTTPSHandler(context=ctx)
    return urllib.request.build_opener(handler)


# ── Robust file downloader ────────────────────────────────────────────────────
MIN_SIZES = {
    "pose_landmarker_lite.task":      1_000_000,   # ~8 MB
    "blaze_face_short_range.tflite":    100_000,   # ~1 MB
    ONNX_MODEL_NAME:                  5_000_000,   # ~13 MB
}

def _file_ok(path: str) -> bool:
    """Return True if file exists and meets the minimum expected size."""
    if not os.path.exists(path):
        return False
    name    = os.path.basename(path)
    min_sz  = MIN_SIZES.get(name, 1000)
    actual  = os.path.getsize(path)
    return actual >= min_sz


def download(url: str, dest: str, label: str, min_mb: float = 0.0) -> bool:
    """
    Download url → dest with SSL fallback chain and completeness check.
    Deletes any existing file that is too small (partial/corrupt).
    """
    # Remove corrupt / partial files
    if os.path.exists(dest) and not _file_ok(dest):
        print(f"      ⚠️  Removing corrupt cached file ({os.path.getsize(dest)} bytes)…")
        os.remove(dest)

    if _file_ok(dest):
        size_mb = os.path.getsize(dest) / 1_000_000
        print(f"      ✅ {label} already present ({size_mb:.1f} MB), skipping")
        return True

    print(f"      Downloading {label}…")

    ssl_ctx, ssl_label = _make_ssl_context()

    def _progress(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size / total_size * 100, 100)
            print(f"\r      {pct:5.1f}%  [{ssl_label}]", end="", flush=True)

    # ── Attempt 1: urllib with SSL context ────────────────────────────────────
    try:
        opener = _ssl_opener(ssl_ctx)
        tmp    = dest + ".part"
        with opener.open(url) as resp, open(tmp, "wb") as f:
            total   = int(resp.headers.get("Content-Length", 0))
            written = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                written += len(chunk)
                if total > 0:
                    pct = min(written / total * 100, 100)
                    print(f"\r      {pct:5.1f}%  [{ssl_label}]", end="", flush=True)
        print()
        os.rename(tmp, dest)

    except Exception as e1:
        if os.path.exists(dest + ".part"):
            os.remove(dest + ".part")

        # ── Attempt 2: unverified SSL (warn user) ─────────────────────────────
        print(f"\n      ⚠️  Verified SSL failed ({e1})")
        print(f"      Retrying with unverified SSL (safe for model downloads)…")
        try:
            unverified_ctx = ssl._create_unverified_context()
            opener2 = _ssl_opener(unverified_ctx)
            tmp = dest + ".part"
            with opener2.open(url) as resp, open(tmp, "wb") as f:
                total   = int(resp.headers.get("Content-Length", 0))
                written = 0
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    written += len(chunk)
                    if total > 0:
                        pct = min(written / total * 100, 100)
                        print(f"\r      {pct:5.1f}%  [unverified]", end="", flush=True)
            print()
            os.rename(tmp, dest)

        except Exception as e2:
            if os.path.exists(dest + ".part"):
                os.remove(dest + ".part")

            # ── Attempt 3: curl (uses macOS system keychain) ──────────────────
            if shutil.which("curl"):
                print(f"\n      ⚠️  urllib failed ({e2})")
                print(f"      Retrying with curl (uses system keychain)…")
                result = subprocess.run(
                    ["curl", "-L", "--progress-bar", "-o", dest, url],
                    capture_output=False
                )
                if result.returncode != 0 or not _file_ok(dest):
                    print(f"      ❌ curl also failed for {label}")
                    if os.path.exists(dest):
                        os.remove(dest)
                    return False
                print(f"      ✅ Downloaded via curl")
            else:
                print(f"      ❌ All download methods failed for {label}")
                print(f"         Tip: run  pip install certifi  and retry")
                return False

    if not _file_ok(dest):
        size = os.path.getsize(dest) if os.path.exists(dest) else 0
        print(f"      ❌ Downloaded file is too small ({size} bytes) — likely corrupt")
        if os.path.exists(dest):
            os.remove(dest)
        return False

    size_mb = os.path.getsize(dest) / 1_000_000
    print(f"      ✅ {label} ready ({size_mb:.1f} MB)")
    return True


# ── [0] Install certifi if missing (silently) ─────────────────────────────────
try:
    import certifi  # noqa: F401
except ImportError:
    print("\n[0] certifi not found — installing for SSL support…")
    subprocess.run([sys.executable, "-m", "pip", "install", "certifi", "-q"],
                   check=False)
    try:
        import certifi  # noqa: F401
        print("      ✅ certifi installed")
    except ImportError:
        print("      ⚠️  certifi install failed — will fall back to unverified SSL")


# ── [1/4] YOLOv8n ─────────────────────────────────────────────────────────────
print("\n[1/4] YOLOv8n — person detection (~6 MB)")
try:
    # Patch SSL for ultralytics downloader on macOS
    _ssl_ctx, _ = _make_ssl_context()
    if _ssl_ctx:
        ssl._create_default_https_context = lambda: _ssl_ctx  # noqa: SLF001
    from ultralytics import YOLO
    YOLO("yolov8n.pt")
    print("      ✅ YOLOv8n ready")
except Exception as e:
    # Fallback: unverified context
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        from ultralytics import YOLO
        YOLO("yolov8n.pt")
        print("      ✅ YOLOv8n ready (unverified SSL)")
    except Exception as e2:
        print(f"      ❌ Failed: {e2}")
        sys.exit(1)


# ── [2/4] MediaPipe Pose Landmarker ───────────────────────────────────────────
print("\n[2/4] MediaPipe Pose Landmarker (~8 MB)")
POSE_URL  = ("https://storage.googleapis.com/mediapipe-models/"
             "pose_landmarker/pose_landmarker_lite/float16/latest/"
             "pose_landmarker_lite.task")
POSE_DEST = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")

if not download(POSE_URL, POSE_DEST, "pose_landmarker_lite.task"):
    sys.exit(1)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_DEST),
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    lm = mp_vision.PoseLandmarker.create_from_options(opts)
    lm.close()
    print("      ✅ Pose Landmarker verified")
except Exception as e:
    print(f"      ❌ Verification failed: {e}")
    sys.exit(1)


# ── [3/4] MediaPipe Face Detector ─────────────────────────────────────────────
print("\n[3/4] MediaPipe Face Detector (~1 MB)")
FACE_URL  = ("https://storage.googleapis.com/mediapipe-models/"
             "face_detector/blaze_face_short_range/float16/latest/"
             "blaze_face_short_range.tflite")
FACE_DEST = os.path.join(MODELS_DIR, "blaze_face_short_range.tflite")

if not download(FACE_URL, FACE_DEST, "blaze_face_short_range.tflite"):
    sys.exit(1)

try:
    opts = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_DEST),
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    fd = mp_vision.FaceDetector.create_from_options(opts)
    fd.close()
    print("      ✅ Face Detector verified")
except Exception as e:
    print(f"      ❌ Verification failed: {e}")
    sys.exit(1)


# ── [4/4] HSEmotion ONNX ──────────────────────────────────────────────────────
print("\n[4/4] HSEmotion ONNX — emotion classifier (~13 MB)")
os.makedirs(HSEMOTION_CACHE, exist_ok=True)

# Remove corrupt cached file before attempting download
if os.path.exists(ONNX_DEST) and not _file_ok(ONNX_DEST):
    print(f"      ⚠️  Removing corrupt ONNX cache ({os.path.getsize(ONNX_DEST)} bytes)…")
    os.remove(ONNX_DEST)

if not _file_ok(ONNX_DEST):
    # Download manually — bypass the library's own downloader which ignores SSL context
    ONNX_URL = ("https://github.com/HSE-asavchenko/face-emotion-recognition"
                "/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx")
    if not download(ONNX_URL, ONNX_DEST, ONNX_MODEL_NAME):
        sys.exit(1)
else:
    size_mb = os.path.getsize(ONNX_DEST) / 1_000_000
    print(f"      ✅ {ONNX_MODEL_NAME} already present ({size_mb:.1f} MB), skipping")

# Validate ONNX file can actually be loaded
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_DEST, providers=["CPUExecutionProvider"])
    _ = sess.get_inputs()
    print("      ✅ ONNX model loads correctly")
    del sess
except Exception as e:
    print(f"      ❌ ONNX validation failed: {e}")
    print(f"         Removing corrupt file and re-run this script to retry.")
    if os.path.exists(ONNX_DEST):
        os.remove(ONNX_DEST)
    sys.exit(1)

# Final smoke-test through the HSEmotion wrapper
try:
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    rec = HSEmotionRecognizer(model_name="enet_b0_8_best_vgaf")
    print("      ✅ HSEmotion wrapper ready")
except Exception as e:
    print(f"      ❌ HSEmotion wrapper failed: {e}")
    sys.exit(1)


# ── Done ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 40)
print("✅ All models downloaded and verified!")
print()
print("   Start the app with:")
print("   streamlit run app.py")
