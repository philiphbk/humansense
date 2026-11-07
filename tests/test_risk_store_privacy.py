import numpy as np

from humansense.modules import risk
from humansense.utils import event_store
from humansense.utils.calibration import Calibration
from humansense.utils.privacy import blur_heads


def setup_function():
    risk.reset_state()


def test_single_weak_cue_stays_low_but_agreeing_cues_escalate():
    assert risk.assess({"pose": {"cue_records": [{"id": 1, "cue": "Seated posture"}]}}, [])["level"] == "ok"
    fall = {"pose": {"cue_records": [{"id": 1, "cue": "FALL detected (horizontal posture)"}]}}
    assert risk.assess(fall, [{"id": 1, "still_frames": 0}])["level"] == "critical"
    still = risk.assess(fall, [{"id": 1, "still_frames": 10}])
    assert still["score"] > 60 and any("No movement" in r for r in still["reasons"])


def test_lingering_at_robot_adds_risk():
    res = {"proximity": {"violations": [{"type": "person_to_robot", "id": 7}], "alert_level": "caution"}}
    scores = [risk.assess(res, [])["score"] for _ in range(risk.DWELL_FRAMES_NEAR_ROBOT)]
    assert scores[-1] > scores[0]


def test_events_are_persisted(tmp_path):
    db = str(tmp_path / "events.db")
    event_store.save_event("warning", "Fall detected", path=db)
    assert event_store.recent_events(path=db)[0][1:] == ("warning", "Fall detected")


def test_blur_changes_only_the_head_region():
    frame = np.random.randint(0, 255, (200, 200, 3), np.uint8)
    out = blur_heads(frame, [{"bbox": (50, 40, 110, 180)}])
    assert not np.array_equal(out[40:60, 50:110], frame[40:60, 50:110])
    assert np.array_equal(out[100:, :], frame[100:, :])


def test_no_fall_zone_membership():
    cal = Calibration({"no_fall_zones": [[[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]]})
    assert cal.in_no_fall_zone((100, 100), 640, 480)
    assert not cal.in_no_fall_zone((600, 450), 640, 480)
