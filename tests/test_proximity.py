import numpy as np
import pytest

from humansense.modules import proximity_detector as P

FRAME = np.zeros((480, 640, 3), np.uint8)


def person(cx, height, pid):
    return {"bbox": (cx - height // 6, 400 - height, cx + height // 6, 400), "id": pid}


@pytest.fixture(autouse=True)
def fresh_state():
    P.reset_state()


def run(dets, frames=2, **kw):
    for _ in range(frames):
        r = P.check_proximity(FRAME, dets, **kw)
    return r


def test_close_people_flag_only_after_persisting():
    dets = [person(200, 300, 1), person(260, 300, 2)]
    assert P.check_proximity(FRAME, dets)["violation_count"] == 0
    assert P.check_proximity(FRAME, dets)["violation_count"] == 1


def test_far_apart_is_clear():
    assert run([person(100, 300, 1), person(500, 300, 2)])["alert_level"] == "none"


def test_different_depths_are_not_close():
    assert run([person(200, 300, 1), person(230, 100, 2)])["violation_count"] == 0


def test_near_the_robot():
    r = run([person(320, 200, 1)], robot_position=(0.5, 400 / 480))
    assert any(v["type"] == "person_to_robot" for v in r["violations"])


def test_metric_mode_uses_real_distance():
    from humansense.utils.calibration import Calibration
    # A 4 m x 4 m floor square covering the whole image
    cal = Calibration({"floor_points_norm": [[0, 0], [1, 0], [1, 1], [0, 1]],
                       "floor_points_m": [[0, 0], [4, 0], [4, 4], [0, 4]]})
    near = run([person(300, 100, 1), person(340, 100, 2)], calibration=cal)      # ~0.25 m apart
    P.reset_state()
    far = run([person(100, 100, 1), person(500, 100, 2)], calibration=cal)       # ~2.5 m apart
    assert near["violation_count"] == 1 and far["violation_count"] == 0
    assert near["violations"][0]["distance_m"] < 1.2
