"""Pipeline plumbing, with a stub person detector so no model files or network are needed."""
import json

import numpy as np
import pytest

from humansense import Pipeline, PipelineConfig, detectors


class StubDetector(detectors.PersonDetector):
    name = "stub"

    def __init__(self, boxes):
        self.boxes = boxes
        self.seen_shapes = []

    def detect(self, frame, conf=0.4):
        self.seen_shapes.append(frame.shape)
        return [{"bbox": b, "confidence": 0.9} for b in self.boxes]


@pytest.fixture
def stub(monkeypatch):
    det = StubDetector([(100, 100, 160, 300), (400, 120, 470, 330)])
    monkeypatch.setattr(detectors, "set_backend", lambda *a, **k: det)
    monkeypatch.setattr(detectors, "_current", det)
    return det


FRAME = np.zeros((480, 640, 3), np.uint8)


def test_result_is_json_serialisable_and_complete(stub):
    pipe = Pipeline(PipelineConfig(modules=("crowd", "proximity"), max_width=None))
    res = pipe.process(FRAME, timestamp=12.5)
    d = json.loads(json.dumps(res.to_dict()))
    assert d["person_count"] == 2 and len(d["people"]) == 2
    assert d["timestamp"] == 12.5 and d["frame_size"] == [640, 480]
    assert {p["id"] for p in d["people"]} == {1, 2}
    assert d["risk"]["level"] in ("ok", "elevated", "critical")
    assert "total" in d["timings_ms"] and not d["errors"]


def test_boxes_are_reported_in_original_pixels_when_downscaled(stub):
    big = np.zeros((960, 1280, 3), np.uint8)
    stub.boxes = [(100, 100, 200, 300)]                     # in the downscaled (640 wide) frame
    res = Pipeline(PipelineConfig(modules=("crowd",), max_width=640)).process(big)
    assert stub.seen_shapes[-1][:2] == (480, 640)
    assert res.people[0].bbox == (200, 200, 400, 600) and res.frame_size == (1280, 960)


def test_draw_false_skips_rendering_and_draw_true_produces_an_image(stub):
    cfg = dict(modules=("crowd", "proximity"), max_width=None)
    assert Pipeline(PipelineConfig(**cfg)).process(FRAME).annotated is None
    out = Pipeline(PipelineConfig(draw=True, **cfg)).process(FRAME).annotated
    assert out is not None and out.shape[1] == 640


def test_a_failing_module_is_reported_not_raised(stub, monkeypatch):
    from humansense.modules import proximity_detector

    def boom(*a, **k):
        raise RuntimeError("kaput")
    monkeypatch.setattr(proximity_detector, "check_proximity", boom)
    res = Pipeline(PipelineConfig(modules=("crowd", "proximity"), max_width=None)).process(FRAME)
    assert res.person_count == 2                            # the other modules still delivered
    assert any("proximity" in e and "kaput" in e for e in res.errors)


def test_unknown_module_name_is_rejected():
    with pytest.raises(ValueError, match="unknown modules"):
        PipelineConfig(modules=("crowd", "telepathy"))


def test_reset_restarts_track_ids(stub):
    pipe = Pipeline(PipelineConfig(modules=("crowd",), max_width=None))
    pipe.process(FRAME)
    pipe.reset()
    assert sorted(p.id for p in pipe.process(FRAME).people) == [1, 2]
