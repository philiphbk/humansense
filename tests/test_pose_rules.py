import pytest

pytest.importorskip("mediapipe")
from humansense.modules import pose_detector as S


def landmarks(**kw):
    pts = [(0, 0, 0.0, 0.0)] * 33
    for name, v in kw.items():
        pts[getattr(S, "_" + name)] = v
    return pts


def upright(scale=1.0, x0=300, y0=100):
    # MediaPipe "left" appears on the image's right for a person facing the camera
    return landmarks(
        L_SHOULDER=(x0 + 30 * scale, y0, 1, 0), R_SHOULDER=(x0 - 30 * scale, y0, 1, 0),
        L_HIP=(x0 + 20 * scale, y0 + 120 * scale, 1, 0), R_HIP=(x0 - 20 * scale, y0 + 120 * scale, 1, 0),
        L_WRIST=(x0 + 40 * scale, y0 + 100 * scale, 1, 0), R_WRIST=(x0 - 40 * scale, y0 + 100 * scale, 1, 0),
        L_KNEE=(x0 + 20 * scale, y0 + 230 * scale, 1, 0), R_KNEE=(x0 - 20 * scale, y0 + 230 * scale, 1, 0),
    )


def names(pts, bbox):
    return [c[0] for c in S._person_cues(pts, bbox)]


@pytest.mark.parametrize("scale,bbox", [(1.0, (250, 80, 350, 420)), (0.2, (290, 95, 310, 160))])
def test_standing_person_is_normal_at_any_distance(scale, bbox):
    assert names(upright(scale), bbox) == []


LYING = landmarks(L_SHOULDER=(200, 300, 1, 0), R_SHOULDER=(200, 340, 1, 0),
                  L_HIP=(320, 300, 1, 0), R_HIP=(320, 340, 1, 0),
                  L_KNEE=(440, 300, 1, 0), R_KNEE=(440, 340, 1, 0))
LYING_BBOX = (180, 280, 480, 360)


def test_lying_person_is_a_fall_and_not_seated():
    S.reset_state()
    # First-ever observation of this pid (no prior tilt to compare against): defaults to
    # confirming the fall rather than the weaker cue.
    got = names(LYING, LYING_BBOX)
    assert any(c.startswith("FALL detected") for c in got)
    assert "Seated posture" not in got


def test_fall_with_a_tracked_person_needs_a_recent_tilt_rotation():
    S.reset_state()
    pid = 1
    S._person_cues(upright(), (250, 80, 350, 420), pid=pid)   # frame 1: upright baseline
    got = [c[0] for c in S._person_cues(LYING, LYING_BBOX, pid=pid)]   # frame 2: collapses
    assert any(c.startswith("FALL detected") for c in got)


def test_already_lying_with_no_rotation_is_only_a_fall_risk():
    S.reset_state()
    pid = 2
    got = []
    for _ in range(S.FALL_COLLAPSE_WINDOW + 2):
        got = [c[0] for c in S._person_cues(LYING, LYING_BBOX, pid=pid)]
    # Been horizontal for a while now with no further rotation in that window: a static rest
    # posture (e.g. a bed/couch), not a fall event.
    assert any(c.startswith("Fall risk") for c in got)
    assert not any(c.startswith("FALL detected") for c in got)


def test_seated_person():
    p = upright()
    p[S._L_KNEE] = (330, 225, 1, 0)
    p[S._R_KNEE] = (270, 225, 1, 0)
    assert "Seated posture" in names(p, (250, 80, 350, 300))


def test_raised_hand():
    p = upright()
    p[S._L_WRIST] = (330, 40, 1, 0)
    assert "Raised hand (left hand)" in names(p, (250, 20, 350, 420))


def test_arms_crossed_only_when_wrists_cross_the_midline():
    p = upright()
    p[S._L_WRIST] = (290, 200, 1, 0)   # left wrist (right-hand side of image) moved left of centre
    p[S._R_WRIST] = (310, 200, 1, 0)
    assert any(c.startswith("Arms crossed") for c in names(p, (250, 80, 350, 420)))
    assert not any(c.startswith("Arms crossed") for c in names(upright(), (250, 80, 350, 420)))
