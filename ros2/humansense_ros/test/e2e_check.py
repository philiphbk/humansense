#!/usr/bin/env python3
"""
End-to-end ROS 2 check: listen to HumanSense output and verify/measure it.

    python3 e2e_check.py --min-msgs 50 [--expect-level critical] [--expect-annotated] [--out result.json]

Latency = (receive time - header.stamp). The video publisher stamps each image at publish time, so
this measures publish -> DDS -> inference -> DDS -> subscriber, i.e. the delay a robot would see.

Always subscribes to ~/detections (--detections-topic) and fails if none arrive; ~/annotated
(--annotated-topic) is only subscribed for the summary counts, and only *required* (fails if none
arrive) when --expect-annotated is passed, since the node only publishes it when launched with
publish_annotated:=true. Neither check inspects risk/cue/violation *content* -- a generic demo
clip is unlikely to trigger any -- see test_node_messages.py and test_risk_integration.py for that.
"""
import argparse
import json
import statistics
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from humansense_msgs.msg import HumanSenseState


class Checker(Node):
    def __init__(self, topic, detections_topic, annotated_topic):
        super().__init__("e2e_check")
        self.lat, self.proc, self.levels, self.kinds, self.people = [], [], [], set(), []
        self.first_wall = self.last_wall = None
        self.detections_n = 0
        self.last_detection_count = None
        self.annotated_n = 0
        self.create_subscription(HumanSenseState, topic, self.cb, 50)
        self.create_subscription(Detection2DArray, detections_topic, self.det_cb, 50)
        self.create_subscription(Image, annotated_topic, self.annotated_cb, 5)

    def cb(self, m):
        now = self.get_clock().now().nanoseconds * 1e-9
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.lat.append((now - stamp) * 1000.0)
        self.proc.append(m.processing_ms)
        self.levels.append(m.risk.level)
        self.people.append(m.person_count)
        self.kinds.update(c.kind for c in m.cues)
        self.last_wall = time.monotonic()
        self.first_wall = self.first_wall or self.last_wall

    def det_cb(self, m):
        self.detections_n += 1
        self.last_detection_count = len(m.detections)

    def annotated_cb(self, m):
        self.annotated_n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/humansense/state")
    ap.add_argument("--min-msgs", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--idle-exit", type=float, default=8.0, help="stop after this many silent seconds")
    ap.add_argument("--expect-level", choices=["ok", "elevated", "critical"],
                    help="fail unless the risk level reaches at least this level")
    ap.add_argument("--detections-topic", default="/humansense/detections")
    ap.add_argument("--annotated-topic", default="/humansense/annotated")
    ap.add_argument("--expect-annotated", action="store_true",
                    help="fail unless at least one annotated image is received "
                         "(only meaningful when the node was launched with publish_annotated:=true)")
    ap.add_argument("--out")
    a = ap.parse_args()

    rclpy.init()
    node = Checker(a.topic, a.detections_topic, a.annotated_topic)
    t0 = time.monotonic()
    while rclpy.ok() and time.monotonic() - t0 < a.timeout:
        rclpy.spin_once(node, timeout_sec=0.2)
        if node.last_wall and time.monotonic() - node.last_wall > a.idle_exit:
            break
    n = len(node.lat)
    rank = {"ok": 0, "elevated": 1, "critical": 2}
    summary = {
        "messages": n,
        "latency_ms": {"median": statistics.median(node.lat), "p95": sorted(node.lat)[int(0.95 * (n - 1))],
                       "max": max(node.lat)} if n else None,
        "processing_ms_median": statistics.median(node.proc) if n else None,
        "output_hz": (n - 1) / (node.last_wall - node.first_wall) if n > 1 else None,
        "max_risk_level": max(node.levels) if n else None,
        "max_people": max(node.people) if n else None,
        "cue_kinds": sorted(node.kinds),
        "detections_messages": node.detections_n,
        "last_detection_count": node.last_detection_count,
        "annotated_messages": node.annotated_n,
    }
    print(json.dumps(summary, indent=2))
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=2)
    ok = n >= a.min_msgs and node.detections_n > 0
    if a.expect_level:
        ok = ok and n and summary["max_risk_level"] >= rank[a.expect_level]
    if a.expect_annotated:
        ok = ok and node.annotated_n > 0
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
