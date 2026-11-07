"""
Frame sources: live cameras, video files and image sequences behind one iterator interface.

The live ``CameraSource`` runs a dedicated grab thread that always keeps only the *newest*
frame. When inference is slower than the camera, a plain ``cv2.VideoCapture.read()`` loop
returns increasingly stale frames from the driver's buffer (seconds of lag, which is
unacceptable for a robot reacting to a fall). Here the consumer always gets the latest
frame and the number of frames skipped is counted, never hidden.
"""

from __future__ import annotations

import glob
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Union

import cv2
import numpy as np

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass
class Frame:
    image: np.ndarray        # BGR
    timestamp: float         # seconds (capture wall-clock for cameras, media time for files)
    index: int               # 0-based index of the frame within the source


class FrameSource:
    """Iterable of ``Frame``. Use as a context manager so resources are always released."""

    def __iter__(self) -> Iterator[Frame]:  # pragma: no cover
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class VideoFileSource(FrameSource):
    """A video file. ``realtime=True`` paces playback to the file's own frame rate."""

    def __init__(self, path: str, loop: bool = False, realtime: bool = False):
        self.path, self.loop, self.realtime = path, loop, realtime
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"cannot open video: {path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    def __iter__(self):
        idx, t_start = 0, time.perf_counter()
        while True:
            ok, img = self._cap.read()
            if not ok:
                if self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                return
            ts = idx / self.fps
            if self.realtime:
                delay = ts - (time.perf_counter() - t_start)
                if delay > 0:
                    time.sleep(delay)
            yield Frame(img, ts, idx)
            idx += 1

    def close(self):
        self._cap.release()


class ImageSequenceSource(FrameSource):
    """A directory of images (sorted by name), e.g. dataset frames, replayed at ``fps``."""

    def __init__(self, directory: str, fps: float = 30.0):
        self.files = sorted(f for f in glob.glob(os.path.join(directory, "*"))
                            if f.lower().endswith(IMAGE_EXTS))
        if not self.files:
            raise IOError(f"no images in {directory}")
        self.fps, self.total = fps, len(self.files)

    def __iter__(self):
        for i, f in enumerate(self.files):
            img = cv2.imread(f)
            if img is not None:
                yield Frame(img, i / self.fps, i)


class CameraSource(FrameSource):
    """
    Live camera (device index, V4L2 path, RTSP/HTTP URL or GStreamer pipeline string).

    ``read()`` blocks for the next frame that has not been delivered yet and returns None on
    timeout or once the camera has been lost for good. Iterating yields frames until ``close()``.
    """

    def __init__(self, device: Union[int, str] = 0, width: Optional[int] = None,
                 height: Optional[int] = None, fps: Optional[float] = None,
                 reconnect: bool = True, reconnect_delay: float = 1.0,
                 capture_factory: Optional[Callable[[], "cv2.VideoCapture"]] = None):
        self.device, self.reconnect, self.reconnect_delay = device, reconnect, reconnect_delay
        self._factory = capture_factory or (lambda: self._open_device(device, width, height, fps))
        self._cond = threading.Condition()
        self._latest: Optional[Frame] = None
        self._stop = False
        self._lost = False
        self.captured = 0       # frames grabbed from the device
        self.delivered = 0      # frames handed to the consumer
        self.reconnects = 0
        self._next_index = 0    # first frame index the consumer has not yet received
        self._cap = self._factory()
        if not self._cap.isOpened():
            raise IOError(
                f"cannot open camera {device!r}. On macOS grant the terminal/IDE camera access "
                "(System Settings -> Privacy & Security -> Camera); on Linux check /dev/video* "
                "permissions (video group) and that no other process holds the device.")
        self._thread = threading.Thread(target=self._grab_loop, name="camera-grab", daemon=True)
        self._thread.start()

    @staticmethod
    def _open_device(device, width, height, fps):
        if isinstance(device, str) and device.isdigit():
            device = int(device)
        cap = cv2.VideoCapture(device)
        if cap.isOpened():
            if width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps:
                cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # honoured by some backends only; the thread is the real fix
        return cap

    # -- grab thread
    def _grab_loop(self):
        index = 0
        while not self._stop:
            ok, img = self._cap.read()
            if not ok:
                if not self._reopen():
                    break
                continue
            with self._cond:
                self._latest = Frame(img, time.time(), index)
                index += 1
                self.captured += 1
                self._cond.notify_all()
        with self._cond:
            self._lost = True
            self._cond.notify_all()

    def _reopen(self) -> bool:
        if not self.reconnect or self._stop:
            return False
        self._cap.release()
        while not self._stop:
            time.sleep(self.reconnect_delay)
            cap = self._factory()
            if cap.isOpened():
                self._cap, self.reconnects = cap, self.reconnects + 1
                return True
            cap.release()
        return False

    # -- consumer API
    def read(self, timeout: float = 2.0) -> Optional[Frame]:
        """Newest undelivered frame; None on timeout or if the camera is gone."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                if self._latest is not None and self._latest.index >= self._next_index:
                    frame = self._latest
                    self._next_index = frame.index + 1
                    self.delivered += 1
                    return frame
                if self._lost or self._stop:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)

    @property
    def dropped(self) -> int:
        """Frames grabbed but overwritten before the consumer asked for them."""
        return max(self.captured - self.delivered, 0)

    def __iter__(self):
        while not self._stop:
            f = self.read()
            if f is None:
                if self._lost:
                    return
                continue
            yield f

    def close(self):
        self._stop = True
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=2.0)
        self._cap.release()


def open_source(spec: Union[str, int], **kwargs) -> FrameSource:
    """
    Open a source from a string: ``0`` / ``/dev/video0`` / ``rtsp://...`` -> camera,
    a directory -> image sequence, anything else -> video file.
    """
    s = str(spec)
    if s.isdigit() or s.startswith(("/dev/video", "rtsp://", "http://", "https://")):
        cam_keys = ("width", "height", "fps", "reconnect")
        return CameraSource(int(s) if s.isdigit() else s,
                            **{k: v for k, v in kwargs.items() if k in cam_keys})
    if os.path.isdir(s):
        return ImageSequenceSource(s, fps=kwargs.get("fps") or 30.0)
    return VideoFileSource(s, loop=kwargs.get("loop", False), realtime=kwargs.get("realtime", False))
