"""sensor_msgs/Image <-> numpy, without cv_bridge (fewer dependencies, no numpy-ABI surprises)."""

import numpy as np

# encoding -> (channels, channel order to reach BGR)
_ENCODINGS = {
    "bgr8": (3, None),
    "rgb8": (3, [2, 1, 0]),
    "mono8": (1, "gray"),
    "8UC1": (1, "gray"),
    "bgra8": (4, [0, 1, 2]),
    "rgba8": (4, [2, 1, 0]),
}


def image_to_bgr(msg) -> np.ndarray:
    """Convert a sensor_msgs/Image (8-bit encodings) into an HxWx3 BGR uint8 array."""
    try:
        channels, order = _ENCODINGS[msg.encoding]
    except KeyError:
        raise ValueError(
            f"unsupported image encoding {msg.encoding!r}; supported: {sorted(_ENCODINGS)}. "
            "Convert upstream (e.g. image_proc / a camera driver pixel_format parameter).")
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    rows = buf.reshape(msg.height, msg.step)[:, : msg.width * channels]   # drop row padding
    img = rows.reshape(msg.height, msg.width, channels)
    if order is None:
        return np.ascontiguousarray(img)
    if order == "gray":
        return np.ascontiguousarray(np.repeat(img, 3, axis=2))
    return np.ascontiguousarray(img[:, :, order])


def bgr_to_image(frame: np.ndarray, header, image_cls):
    """Pack an HxWx3 BGR uint8 array into a sensor_msgs/Image (class passed in to avoid a hard import)."""
    msg = image_cls()
    msg.header = header
    msg.height, msg.width = frame.shape[:2]
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(frame).tobytes()
    return msg
