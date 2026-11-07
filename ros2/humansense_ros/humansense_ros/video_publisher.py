"""Publish a video file / image directory / camera as sensor_msgs/Image (demo + testing helper)."""

import time

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from humansense.sources import open_source

from .convert import bgr_to_image


class VideoPublisher(Node):
    def __init__(self):
        super().__init__("video_publisher")
        self.declare_parameter("source", "")
        self.declare_parameter("fps", 15.0, ParameterDescriptor(dynamic_typing=True))   # accept 30 or 30.0
        self.declare_parameter("loop", False)
        self.declare_parameter("frame_id", "camera")
        self.declare_parameter("max_frames", 0)
        self.declare_parameter("start_delay", 2.0)
        src = self.get_parameter("source").value
        if not src:
            raise SystemExit("video_publisher: set the 'source' parameter")
        self._fps = float(self.get_parameter("fps").value)
        self._frame_id = self.get_parameter("frame_id").value
        self._max = int(self.get_parameter("max_frames").value)
        self._src = open_source(src, loop=bool(self.get_parameter("loop").value), fps=self._fps)
        self._it = iter(self._src)
        self._n = 0
        self._pub = self.create_publisher(Image, "image", qos_profile_sensor_data)
        self._t0 = time.monotonic() + float(self.get_parameter("start_delay").value)
        self.create_timer(1.0 / self._fps, self._tick)

    def _tick(self):
        if time.monotonic() < self._t0:
            return          # give subscribers (and the model load) time to come up
        try:
            frame = next(self._it)
        except StopIteration:
            self.get_logger().info(f"source finished after {self._n} frames")
            raise SystemExit(0)
        msg = bgr_to_image(frame.image, self._header(), Image)
        self._pub.publish(msg)
        self._n += 1
        if self._max and self._n >= self._max:
            raise SystemExit(0)

    def _header(self):
        from std_msgs.msg import Header
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        h.frame_id = self._frame_id
        return h


def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
