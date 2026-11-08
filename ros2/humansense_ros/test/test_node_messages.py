"""
Unit tests for the HumanSense -> ROS message conversion (node.build_state / build_risk /
build_detections). These are the pure functions node.py documents as "unit-testable" -- they turn
a `humansense.result.FrameResult` into the messages the node publishes on ~/state, ~/risk and
~/detections, including the risk/violation/cue fields that the end-to-end video smoke test
(e2e_check.py against assets/demo.mp4) never exercises because that clip never triggers them.

Run inside a ROS 2 environment (needs humansense_msgs, vision_msgs, std_msgs, and the `humansense`
Python package built/importable).
"""
import math

import pytest

std_msgs = pytest.importorskip("std_msgs.msg")
pytest.importorskip("vision_msgs.msg")
from humansense.result import Cue, FrameResult, Person, Risk, Violation  # noqa: E402
from humansense_ros.node import build_detections, build_risk, build_state  # noqa: E402
from humansense_msgs.msg import Risk as RiskMsg  # noqa: E402


def _header():
    h = std_msgs.Header()
    h.frame_id = "camera"
    return h


def _critical_result():
    return FrameResult(
        frame_id=7,
        timestamp=1.0,
        frame_size=(640, 480),
        people=[
            Person(id=1, bbox=(10, 20, 30, 120), confidence=0.9, speed=0.1, still_frames=6),
            Person(id=2, bbox=(200, 20, 230, 130), confidence=0.8, speed=0.0, still_frames=0),
        ],
        person_count=2,
        density_level="medium",
        proximity_level="warning",
        violations=[
            Violation(type="person_to_person", ids=(1, 2), severity="warning",
                      distance_px=12.5, distance_m=None),
            Violation(type="person_to_robot", ids=(2,), severity="caution",
                      distance_px=40.0, distance_m=1.1),
        ],
        cues=[
            Cue(person_id=1, kind="fall", label="FALL DETECTED", severity=2),
            Cue(person_id=2, kind="raised_hand", label="Raised hand", severity=1),
        ],
        risk=Risk(score=85, level="critical",
                  reasons=["Fall detected (#1)", "Personal space violated"]),
        timings_ms={"total": 37.2},
    )


def test_build_risk_maps_level_and_clamps_score():
    r = build_risk(_critical_result())
    assert r.level == RiskMsg.LEVEL_CRITICAL
    assert r.score == 85
    assert list(r.reasons) == ["Fall detected (#1)", "Personal space violated"]


def test_build_risk_clamps_out_of_range_score():
    res = _critical_result()
    res.risk = Risk(score=140, level="critical", reasons=[])
    assert build_risk(res).score == 100
    res.risk = Risk(score=-5, level="ok", reasons=[])
    assert build_risk(res).score == 0


def test_build_state_carries_header_and_frame_metadata():
    header = _header()
    s = build_state(_critical_result(), header)
    assert s.header is header
    assert s.frame_id == 7
    assert (s.image_width, s.image_height) == (640, 480)
    assert s.person_count == 2
    assert s.density_level == "medium"
    assert s.proximity_level == "warning"
    assert s.processing_ms == pytest.approx(37.2)


def test_build_state_people_fields():
    s = build_state(_critical_result(), _header())
    assert [p.id for p in s.people] == [1, 2]
    p1 = s.people[0]
    assert (p1.x1, p1.y1, p1.x2, p1.y2) == (10, 20, 30, 120)
    assert p1.confidence == pytest.approx(0.9)
    assert p1.still_frames == 6


def test_build_state_violations_both_kinds_and_nan_distance():
    s = build_state(_critical_result(), _header())
    assert len(s.violations) == 2
    pp, pr = s.violations
    assert pp.type == "person_to_person" and list(pp.ids) == [1, 2]
    assert pp.severity == "warning"
    assert math.isnan(pp.distance_m)                 # no calibration -> NaN, not 0
    assert pr.type == "person_to_robot" and list(pr.ids) == [2]
    assert pr.distance_m == pytest.approx(1.1)


def test_build_state_cues():
    s = build_state(_critical_result(), _header())
    assert len(s.cues) == 2
    fall, hand = s.cues
    assert fall.person_id == 1 and fall.kind == "fall" and fall.severity == 2
    assert hand.kind == "raised_hand" and hand.severity == 1


def test_build_state_embeds_critical_risk():
    s = build_state(_critical_result(), _header())
    assert s.risk.level == RiskMsg.LEVEL_CRITICAL
    assert s.risk.score == 85


def test_build_detections_one_box_per_person():
    arr = build_detections(_critical_result(), _header())
    assert len(arr.detections) == 2
    d1 = arr.detections[0]
    assert d1.id == "1"
    assert (d1.bbox.center.position.x, d1.bbox.center.position.y) == (20.0, 70.0)
    assert (d1.bbox.size_x, d1.bbox.size_y) == (20.0, 100.0)
    assert d1.results[0].hypothesis.class_id == "person"
    assert d1.results[0].hypothesis.score == pytest.approx(0.9)


def test_build_detections_empty_when_no_people():
    res = _critical_result()
    res.people = []
    assert build_detections(res, _header()).detections == []


def test_ok_level_and_no_cues_round_trip():
    res = FrameResult(frame_id=1, timestamp=0.0, frame_size=(100, 100))
    s = build_state(res, _header())
    assert s.risk.level == RiskMsg.LEVEL_OK
    assert s.cues == [] and s.violations == [] and s.people == []
