import time

import numpy as np
import pytest

from humansense.sources import CameraSource, ImageSequenceSource, open_source


class FakeCapture:
    """A camera producing numbered frames at `fps`; optionally dies after `die_after` frames."""

    def __init__(self, fps=200.0, die_after=None, opened=True):
        self.period, self.die_after, self.n, self._open = 1.0 / fps, die_after, 0, opened

    def isOpened(self):
        return self._open

    def read(self):
        if self.die_after is not None and self.n >= self.die_after:
            time.sleep(0.01)
            return False, None
        time.sleep(self.period)
        self.n += 1
        return True, np.full((4, 4, 3), self.n % 255, np.uint8)

    def release(self):
        self._open = False

    def set(self, *a):
        return True


def test_slow_consumer_gets_fresh_frames_and_drops_are_counted():
    with CameraSource(capture_factory=lambda: FakeCapture(fps=200)) as cam:
        first = cam.read()
        time.sleep(0.25)                      # consumer stalls; camera keeps producing (~50 frames)
        second = cam.read()
        assert second.index - first.index > 20, "must jump to the newest frame, not the next buffered one"
        assert cam.dropped >= 20
        assert cam.delivered == 2


def test_never_delivers_the_same_frame_twice():
    with CameraSource(capture_factory=lambda: FakeCapture(fps=50)) as cam:
        idx = [cam.read().index for _ in range(10)]
    assert idx == sorted(set(idx))


def test_read_times_out_when_camera_is_silent():
    class Silent(FakeCapture):
        def read(self):
            time.sleep(0.05)
            return False, None
    with CameraSource(capture_factory=lambda: Silent(), reconnect=False) as cam:
        assert cam.read(timeout=0.2) is None


def test_camera_loss_without_reconnect_ends_iteration():
    with CameraSource(capture_factory=lambda: FakeCapture(die_after=5), reconnect=False) as cam:
        assert len(list(cam)) >= 1       # iteration terminates instead of hanging


def test_reconnects_after_device_loss():
    made = []

    def factory():
        cap = FakeCapture(die_after=3) if not made else FakeCapture()
        made.append(cap)
        return cap
    with CameraSource(capture_factory=factory, reconnect=True, reconnect_delay=0.05) as cam:
        deadline = time.time() + 3
        while cam.reconnects == 0 and time.time() < deadline:
            time.sleep(0.02)
        assert cam.reconnects == 1
        assert cam.read(timeout=1.0) is not None   # frames flow again


def test_unopenable_camera_raises_actionable_error():
    with pytest.raises(IOError, match="cannot open camera"):
        CameraSource(capture_factory=lambda: FakeCapture(opened=False))


def test_image_sequence_and_open_source(tmp_path):
    import cv2
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i:04d}.png"), np.full((8, 8, 3), i * 40, np.uint8))
    src = open_source(str(tmp_path), fps=10)
    assert isinstance(src, ImageSequenceSource)
    frames = list(src)
    assert [f.index for f in frames] == [0, 1, 2] and frames[2].timestamp == pytest.approx(0.2)
