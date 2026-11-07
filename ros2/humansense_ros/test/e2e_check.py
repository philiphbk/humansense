#!/usr/bin/env python3
"""
End-to-end ROS 2 check: listen to HumanSense output and verify/measure it.

    python3 e2e_check.py --min-msgs 50 [--expect-level critical] [--out result.json]

Latency = (receive time - header.stamp). The video publisher stamps each image at publish time, so
this measures publish -> DDS -> inference -> DDS -> subscriber, i.e. the delay a robot would see.
"""
import argparse
import json
import statistics
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from humansense_msgs.msg import HumanSenseState


class Checker(Node):
    def __init__(self, topic):
        super().__init__("e2e_check")
        self.lat, self.proc, self.levels, self.kinds, self.people = [], [], [], set(), []
        self.first_wall = self.last_wall = None
        self.create_subscription(HumanSenseState, topic, self.cb, 50)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/humansense/state")
    ap.add_argument("--min-msgs", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--idle-exit", type=float, default=8.0, help="stop after this many silent seconds")
    ap.add_argument("--expect-level", choices=["ok", "elevated", "critical"],
                    help="fail unless the risk level reaches at least this level")
    ap.add_argument("--out")
    a = ap.parse_args()

    rclpy.init()
    node = Checker(a.topic)
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
    }
    print(json.dumps(summary, indent=2))
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=2)
    ok = n >= a.min_msgs
    if a.expect_level:
        ok = ok and n and summary["max_risk_level"] >= rank[a.expect_level]
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
