"""
Risk fusion — combines the individual module outputs into one 0-100 score.
A single weak cue stays low; independent cues that agree (a fall, with nobody moving,
near the robot) add up. Emotion is deliberately weighted low: masks make it unreliable.
"""


# Processed frames a person must stay still after a fall cue to count as "no movement",
# and processed frames of lingering inside the robot's bubble to count as "dwelling".
STILL_FRAMES_AFTER_FALL = 6
DWELL_FRAMES_NEAR_ROBOT = 8

_near_robot_frames: dict[int, int] = {}


def reset_state():
    _near_robot_frames.clear()


def assess(results: dict, detections: list) -> dict:
    """Returns {"score": 0-100, "level": "ok"|"elevated"|"critical", "reasons": [..]}."""
    score, reasons = 0, []
    by_id = {d.get("id"): d for d in detections}

    for rec in results.get("pose", {}).get("cue_records", []):
        cue, pid = rec["cue"], rec["id"]
        if cue.upper().startswith("FALL DETECTED"):
            score += 60
            reasons.append(f"Fall detected (#{pid})")
            if by_id.get(pid, {}).get("still_frames", 0) >= STILL_FRAMES_AFTER_FALL:
                score += 25
                reasons.append(f"No movement after fall (#{pid})")
        elif cue.upper().startswith("FALL RISK"):
            score += 20
            reasons.append(f"Fall risk (#{pid})")
        elif cue.startswith("Raised hand"):
            score += 10
            reasons.append(f"Person asking for attention (#{pid})")

    near = {v["id"] for v in results.get("proximity", {}).get("violations", [])
            if v.get("type") == "person_to_robot"}
    for pid in list(_near_robot_frames):
        if pid not in near:
            del _near_robot_frames[pid]
    for pid in near:
        _near_robot_frames[pid] = _near_robot_frames.get(pid, 0) + 1
        if _near_robot_frames[pid] >= DWELL_FRAMES_NEAR_ROBOT:
            score += 15
            reasons.append(f"Person lingering at robot (#{pid})")

    level = results.get("proximity", {}).get("alert_level")
    if level == "warning":
        score += 20
        reasons.append("Personal space violated")
    elif level == "caution":
        score += 8

    if results.get("crowd", {}).get("density_level") == "high":
        score += 10
        reasons.append("High crowd density")

    top = results.get("emotions", {}).get("highest_priority_cue")
    if top and top["priority"] >= 3:
        score += 8
        reasons.append(f"Facial distress: {top['label']}")
    elif top and top["priority"] == 2:
        score += 3

    score = min(score, 100)
    return {"score": score,
            "level": "critical" if score >= 60 else "elevated" if score >= 25 else "ok",
            "reasons": reasons}
