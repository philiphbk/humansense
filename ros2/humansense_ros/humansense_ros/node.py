"""
HumanSense ROS 2 node.

Subscribes   image                  sensor_msgs/Image        (remap to your camera topic)
Publishes    ~/state                humansense_msgs/HumanSenseState   people, cues, violations, risk
             ~/risk                 humansense_msgs/Risk              just the fused risk (planner-friendly)
             ~/detections           vision_msgs/Detection2DArray      person boxes for rviz / other nodes
             ~/annotated            sensor_msgs/Image                 overlay video (only if enabled)

All outputs copy the input image's header, so ``now - header.stamp`` is the real sensing latency.
"""

import math

import rclpy
from builtin_interfaces.msg import Time  # noqa: F401  (documented dependency)
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from humansense_msgs.msg import Cue, HumanSenseState, Person, Risk, Violation

from .convert import bgr_to_image, image_to_bgr

_LEVELS = {"ok": Risk.LEVEL_OK, "elevated": Risk.LEVEL_ELEVATED, "critical": Risk.LEVEL_CRITICAL}


def build_state(res, header) -> HumanSenseState:
    """FrameResult -> HumanSenseState (pure function; unit-testable)."""
    s = HumanSenseState()
    s.header = header
    s.frame_id = res.frame_id
    s.image_width, s.image_height = res.frame_size
    for p in res.people:
        m = Person()
        m.id = p.id
        m.x1, m.y1, m.x2, m.y2 = p.bbox
        m.confidence, m.speed, m.still_frames = p.confidence, p.speed, p.still_frames
        s.people.append(m)
    s.person_count = res.person_count
    s.density_level = res.density_level
    s.proximity_level = res.proximity_level
    for v in res.violations:
        m = Violation()
        m.type, m.severity = v.type, v.severity
        m.ids = [int(i) for i in v.ids if i is not None]
        m.distance_px = float(v.distance_px)
        m.distance_m = float("nan") if v.distance_m is None else float(v.distance_m)
        s.violations.append(m)
    for c in res.cues:
        m = Cue()
        m.person_id, m.kind, m.label, m.severity = c.person_id, c.kind, c.label, c.severity
        s.cues.append(m)
    s.risk = build_risk(res)
    s.processing_ms = float(res.timings_ms.get("total", 0.0))
    return s


def build_risk(res) -> Risk:
    r = Risk()
    r.level = _LEVELS[res.risk.level]
    r.score = int(min(max(res.risk.score, 0), 100))
    r.reasons = list(res.risk.reasons)
    return r


def build_detections(res, header) -> Detection2DArray:
    arr = Detection2DArray()
    arr.header = header
    for p in res.people:
        d = Detection2D()
        d.header = header
        d.id = str(p.id)
        x1, y1, x2, y2 = p.bbox
        d.bbox.center.position.x = (x1 + x2) / 2.0
        d.bbox.center.position.y = (y1 + y2) / 2.0
        d.bbox.size_x, d.bbox.size_y = float(x2 - x1), float(y2 - y1)
        h = ObjectHypothesisWithPose()
        h.hypothesis.class_id, h.hypothesis.score = "person", float(p.confidence)
        d.results.append(h)
        arr.detections.append(d)
    return arr


class HumanSenseNode(Node):
    def __init__(self):
        super().__init__("humansense")
        P = self.declare_parameter
        P("backend", "auto")                  # auto | mediapipe | yolo
        P("modules", ["crowd", "proximity", "pose"])   # "emotion" is opt-in: least reliable, most privacy-sensitive
        P("conf", 0.4)
        P("max_width", 720)
        P("robot_x", float("nan"))            # robot position in the image, normalised 0..1
        P("robot_y", float("nan"))            # (NaN = disabled)
        P("calibration_path", "")
        P("publish_annotated", False)
        P("models_dir", "")

        if self.get_parameter("models_dir").value:
            import os
            os.environ["HUMANSENSE_MODELS_DIR"] = self.get_parameter("models_dir").value

        from humansense import Pipeline, PipelineConfig
        rx, ry = self.get_parameter("robot_x").value, self.get_parameter("robot_y").value
        self._annotate = bool(self.get_parameter("publish_annotated").value)
        cfg = PipelineConfig(
            modules=tuple(self.get_parameter("modules").value),
            backend=self.get_parameter("backend").value,
            conf=float(self.get_parameter("conf").value),
            max_width=int(self.get_parameter("max_width").value) or None,
            robot_position=None if math.isnan(rx) or math.isnan(ry) else (rx, ry),
            calibration_path=self.get_parameter("calibration_path").value or None,
            draw=self._annotate)
        self._pipe = Pipeline(cfg)
        self._pipe.warmup()          # load every model now, not on the first camera frame
        self._warned_encoding = False

        # Depth 1 + reliable: if inference is slower than the camera, a newer sample supersedes the
        # unacked old one locally (same "always process the newest frame" policy as CameraSource),
        # but delivery of whatever IS sent is retried instead of silently lost to one dropped packet.
        # Best-effort was tried first and measurably lost full frames to packet loss between
        # separate processes/containers well below the inference-bound rate it was meant to shed
        # (see scripts/ros2_e2e.sh and the ROS results in the README) -- unlike small messages,
        # an uncompressed image is many UDP fragments, and losing any one of them drops the whole
        # frame under best-effort.
        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                         reliability=QoSReliabilityPolicy.RELIABLE)
        self._pub_state = self.create_publisher(HumanSenseState, "~/state", 10)
        self._pub_risk = self.create_publisher(Risk, "~/risk", 10)
        self._pub_det = self.create_publisher(Detection2DArray, "~/detections", 10)
        self._pub_img = self.create_publisher(Image, "~/annotated", 1) if self._annotate else None
        self.create_subscription(Image, "image", self._on_image, qos)
        self.get_logger().info(
            f"ready: detector={self._pipe.detector_name} modules={list(cfg.modules)} "
            f"annotated={self._annotate}")

    def _on_image(self, msg: Image):
        try:
            frame = image_to_bgr(msg)
        except ValueError as e:
            if not self._warned_encoding:
                self.get_logger().error(str(e))
                self._warned_encoding = True
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        res = self._pipe.process(frame, timestamp=stamp)
        for err in res.errors:
            self.get_logger().warn(err, throttle_duration_sec=5.0)
        self._pub_state.publish(build_state(res, msg.header))
        self._pub_risk.publish(build_risk(res))
        self._pub_det.publish(build_detections(res, msg.header))
        if self._pub_img is not None and res.annotated is not None:
            self._pub_img.publish(bgr_to_image(res.annotated, msg.header, Image))


def main(args=None):
    rclpy.init(args=args)
    node = HumanSenseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
