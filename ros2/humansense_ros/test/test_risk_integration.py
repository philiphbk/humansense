"""
Live ROS 2 integration test for the humansense_node risk/violation/detections/annotated-image
path -- the one thing the video-driven end-to-end smoke test (e2e_check.py against
assets/demo.mp4, see scripts/ros2_e2e.sh and ci.yml) never exercises, because that clip never
gets a person close enough to another person or to the configured robot position to trigger a
violation (its committed result, benchmarks/results/ros2_e2e.json, shows max_risk_level: 0 and
max_people: 0).

Rather than depend on a real (and licensed, see benchmarks/README.md) fall/crowd dataset, this
stubs the person-detector backend the same way tests/test_pipeline.py does for the plain-Python
pipeline tests: two fixed, close-together boxes feed the *real* crowd/proximity/risk modules, so a
real person_to_person AND person_to_robot violation, a real fused ELEVATED risk score, and real
annotated-image rendering all happen for real -- and get published as real ROS messages over real
topics (/humansense/state, /humansense/detections, /humansense/annotated), which this test
subscribes to and checks. The pose/cue side of the same message-building code is covered
separately (and without needing a live node) by test_node_messages.py.
"""
import time

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("vision_msgs.msg")
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402
from std_msgs.msg import Header  # noqa: E402
from vision_msgs.msg import Detection2DArray  # noqa: E402

from humansense import detectors  # noqa: E402
from humansense_msgs.msg import HumanSenseState, Risk as RiskMsg  # noqa: E402
from humansense_ros.convert import bgr_to_image  # noqa: E402

FRAME_W, FRAME_H = 640, 480
# Two boxes, same height, feet 10px apart -- well inside the ~160px personal-space threshold.
BOXES = [(100, 100, 160, 300), (110, 100, 170, 300)]
# Normalised robot position == person #2's foot point, so the robot's bubble is violated too.
ROBOT_X, ROBOT_Y = (BOXES[1][0] + BOXES[1][2]) / 2 / FRAME_W, BOXES[1][3] / FRAME_H
# on_frames=2 to confirm a violation (see utils/tracking.Debouncer), plus DWELL_FRAMES_NEAR_ROBOT
# (modules/risk.py) more so the person-to-robot "dwelling" bonus pushes score from 20 to 35 --
# past the ELEVATED threshold (25). 14 gives margin over the 9 frames that actually requires.
N_FRAMES = 14
_QOS = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.RELIABLE)


class _StubDetector(detectors.PersonDetector):
    name = "stub"

    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, frame, conf=0.4):
        return [{"bbox": b, "confidence": 0.9} for b in self.boxes]


class _ImagePublisher(Node):
    def __init__(self):
        super().__init__("test_image_publisher")
        self.pub = self.create_publisher(Image, "image", _QOS)
        self._frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)

    def send(self):
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        h.frame_id = "camera"
        self.pub.publish(bgr_to_image(self._frame, h, Image))


class _Collector(Node):
    def __init__(self):
        super().__init__("test_collector")
        self.states, self.detections, self.annotated = [], [], []
        self.create_subscription(HumanSenseState, "/humansense/state", self.states.append, 20)
        self.create_subscription(Detection2DArray, "/humansense/detections",
                                  self.detections.append, 20)
        self.create_subscription(Image, "/humansense/annotated", self.annotated.append, 5)


@pytest.fixture
def harness(monkeypatch):
    # Must be patched before HumanSenseNode() builds its Pipeline (Pipeline.__init__ calls
    # detectors.set_backend), same ordering as the `stub` fixture in tests/test_pipeline.py.
    det = _StubDetector(BOXES)
    monkeypatch.setattr(detectors, "set_backend", lambda *a, **k: det)
    monkeypatch.setattr(detectors, "_current", det)
    rclpy.init(args=[
        "--ros-args",
        "-p", "modules:=['crowd','proximity']",
        "-p", "conf:=0.0",
        "-p", "publish_annotated:=true",
        "-p", f"robot_x:={ROBOT_X}",
        "-p", f"robot_y:={ROBOT_Y}",
    ])
    from humansense_ros.node import HumanSenseNode
    node = HumanSenseNode()
    pub, collector = _ImagePublisher(), _Collector()
    try:
        yield pub, node, collector
    finally:
        for n in (pub, node, collector):
            n.destroy_node()
        rclpy.shutdown()


def _pump(nodes, pub, n_frames):
    """Publish n_frames images, spinning every node after each so callbacks actually run."""
    for _ in range(n_frames):
        pub.send()
        for _ in range(20):
            for n in nodes:
                rclpy.spin_once(n, timeout_sec=0.05)


def test_violations_risk_detections_and_annotated_over_real_topics(harness):
    pub, node, collector = harness
    _pump((pub, node, collector), pub, N_FRAMES)

    assert collector.states, "no HumanSenseState messages received on /humansense/state"
    assert collector.detections, "no Detection2DArray messages received on /humansense/detections"
    assert collector.annotated, "no annotated Image messages received on /humansense/annotated"

    last_state = collector.states[-1]
    assert last_state.person_count == 2

    last_det = collector.detections[-1]
    assert len(last_det.detections) == 2
    assert {d.id for d in last_det.detections} == {"1", "2"}
    assert all(d.results[0].hypothesis.class_id == "person" for d in last_det.detections)

    seen_types = {v.type for s in collector.states for v in s.violations}
    assert "person_to_person" in seen_types, "personal-space violation never fired"
    assert "person_to_robot" in seen_types, "robot-proximity violation never fired"

    max_level = max(s.risk.level for s in collector.states)
    assert max_level >= RiskMsg.LEVEL_ELEVATED, (
        f"risk never reached ELEVATED (max seen: {max_level}); "
        f"last reasons: {list(last_state.risk.reasons)}")

    last_img = collector.annotated[-1]
    assert last_img.height > 0 and last_img.width > 0
