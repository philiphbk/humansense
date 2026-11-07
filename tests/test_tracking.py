from humansense.utils.tracking import Debouncer, PersonTracker


def det(x, y=100, w=40, h=120):
    return {"bbox": (x, y, x + w, y + h)}


def test_ids_are_stable_while_moving():
    t = PersonTracker()
    ids = []
    for step in range(6):
        d = t.update([det(100 + step * 8), det(400)])
        ids.append((d[0]["id"], d[1]["id"]))
    assert len(set(ids)) == 1


def test_new_person_gets_new_id_and_lost_track_expires():
    t = PersonTracker(max_missed=2)
    first = t.update([det(100)])[0]["id"]
    for _ in range(4):
        t.update([])
    assert t.update([det(100)])[0]["id"] != first


def test_stillness_counter_grows_for_stationary_person_and_resets_on_motion():
    t = PersonTracker()
    for _ in range(5):
        d = t.update([det(100)])[0]
    assert d["still_frames"] >= 4
    assert t.update([det(160)])[0]["still_frames"] == 0


def test_debouncer_needs_persistence_and_releases_slowly():
    d = Debouncer(on_frames=2, off_frames=3)
    assert d.update({"a"}) == set()
    assert d.update({"a"}) == {"a"}
    assert d.update(set()) == {"a"}
    assert d.update(set()) == {"a"}
    assert d.update(set()) == set()
