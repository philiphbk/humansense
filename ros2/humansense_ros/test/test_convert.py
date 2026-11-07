"""Unit tests for the image conversion (run inside a ROS 2 environment: needs sensor_msgs)."""
import numpy as np
import pytest

sensor_msgs = pytest.importorskip("sensor_msgs.msg")
from humansense_ros.convert import bgr_to_image, image_to_bgr  # noqa: E402


def _msg(enc, arr, step=None):
    m = sensor_msgs.Image()
    m.height, m.width = arr.shape[:2]
    m.encoding = enc
    ch = 1 if arr.ndim == 2 else arr.shape[2]
    m.step = step or m.width * ch
    m.data = arr.tobytes()
    return m


def test_roundtrip_bgr8():
    img = np.random.randint(0, 255, (6, 8, 3), np.uint8)
    out = image_to_bgr(bgr_to_image(img, sensor_msgs.Image().header, sensor_msgs.Image))
    assert np.array_equal(out, img)


def test_rgb8_is_swapped_to_bgr():
    rgb = np.zeros((2, 2, 3), np.uint8)
    rgb[..., 0] = 200                                   # pure red in RGB
    assert image_to_bgr(_msg("rgb8", rgb))[0, 0].tolist() == [0, 0, 200]


def test_mono8_and_bgra8():
    g = np.full((3, 4), 77, np.uint8)
    assert image_to_bgr(_msg("mono8", g)).shape == (3, 4, 3)
    bgra = np.zeros((3, 4, 4), np.uint8)
    bgra[..., 3] = 255
    assert image_to_bgr(_msg("bgra8", bgra)).shape == (3, 4, 3)


def test_row_padding_is_dropped():
    img = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    padded = np.zeros((2, 16), np.uint8)
    padded[:, :12] = img.reshape(2, 12)
    m = _msg("bgr8", img)
    m.step, m.data = 16, padded.tobytes()
    assert np.array_equal(image_to_bgr(m), img)


def test_unsupported_encoding_is_explained():
    m = _msg("bgr8", np.zeros((2, 2, 3), np.uint8))
    m.encoding = "yuv422_yuy2"
    with pytest.raises(ValueError, match="unsupported image encoding"):
        image_to_bgr(m)
