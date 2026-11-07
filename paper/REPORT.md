# HumanSense v2.0 -- ROS 2 transport, idle-machine latency, and a fall-detection dynamics fix

This is a short, reproducible report, not a formal paper (see `benchmarks/README.md` for how to
reproduce every number here from scratch). It covers three pieces of work done together because
each one's result shaped the next: a ROS 2 end-to-end re-test surfaced a transport bug, fixing it
gave a clean latency baseline, and the fall-detection dynamics fix was tuned and evaluated on top
of that fixed pipeline.

> **Citations below are drafted from general knowledge of these works, not re-verified against
> the primary sources for this report.** Please check titles, venues, years, and especially any
> DOI/page numbers before citing this document or its sources further.
>
> **Provenance.** Sections 2-4 are new measurements from this report's own session.
> `benchmarks/results/emotion.json` and `benchmarks/results/falls_mediapipe.json` predate it (they
> shipped with the repo) and were neither reproduced nor re-verified here; `make_tables.py` still
> renders them (into `paper/tables.md`) since it processes every file in `benchmarks/results/`, but
> they are not otherwise discussed in this report.

## 1. Summary

| Question | Answer |
|---|---|
| Does the ROS 2 node drop frames at some low-enough frame rate? | No frame rate eliminates it under the shipped `BEST_EFFORT` image-topic QoS -- the loss is a transport bug, not a backlog effect. Switched to `RELIABLE`; confirmed zero drops at 10 Hz (and in isolation, at any rate tried). |
| What does the pipeline cost per frame on an idle machine? | Person detection: 25.3 ms median. ROS node's default module set (crowd+proximity+pose): 35.3 ms median, 37.3 ms p95. All four modules: 37.2 ms median. |
| Does a dynamics-aware fall rule beat the static posture rule? | See Section 4 -- filled in once the odd/even URFD runs complete. |

## 2. ROS 2 end-to-end re-test

**Setup.** `docker build -f docker/ros2.Dockerfile`, `ros2 launch humansense_ros demo.launch.py`
replaying `assets/demo.mp4` (720x480, 550 frames) as the camera topic, checked with
`ros2/humansense_ros/test/e2e_check.py`. "Idle machine" here means: no other containers or
workloads running, `--ipc=host`, on a 22-CPU / 16 GB Docker Desktop VM (Windows host, WSL2
backend) -- see Section 5 for what that does and doesn't tell you about a real robot's network.

**What happened at the originally-intended frame rate.** The intended test -- lower the frame rate
until nothing drops -- doesn't have an answer under the shipped QoS. At every rate tried, the loss
was roughly rate-*independent*:

| fps | Published | Delivered | Delivery rate |
|---|---|---|---|
| 5.0 | 550 | 163 | 29.6% |
| 2.0 | 40 | 19 | 47.5% |
| 0.5 | 16 | 4 | 25.0% |

That is inconsistent with the backlog-shedding behavior the node's QoS comment described ("if
inference is slower than the camera..."): inference here is 35 ms/frame median, nowhere near the
limiting factor at any of these rates.

**Root cause.** Isolated with a controlled A/B: two separate processes, one publisher and one
subscriber, no inference in the loop, publishing a 720x480 `bgr8` `Image` (~1 MB, uncompressed) in
a plain loop.

| QoS reliability | Delivered |
|---|---|
| `BEST_EFFORT` | 6 / 30 |
| `RELIABLE` | 30 / 30 |

An uncompressed 1 MB message is fragmented into hundreds of UDP packets; under `BEST_EFFORT`, DDS
does not retransmit, so losing *any one* fragment drops the *whole* message. That failure mode is
dominated by per-message fragment-loss probability, not by publish rate -- which is exactly the
rate-independent loss pattern observed above. (A same-process version of the same A/B showed no
loss under either policy, for both: rclpy's intra-process communication shortcut bypasses the wire
entirely when publisher and subscriber share a process, so it does not exercise the real path and
is not informative here.)

**Fix.** Changed both `humansense_node`'s image subscription and `video_publisher`'s image
publisher from `BEST_EFFORT` to `RELIABLE` (`ros2/humansense_ros/humansense_ros/node.py`,
`video_publisher.py`), keeping history depth at 1 so a newer frame still supersedes an unacked
older one locally -- same "always process the newest frame" policy, just with retry instead of
silent loss for whatever *is* sent.

**Result after the fix**, full demo video at 10 Hz:

| Messages published | Messages delivered | Output rate | Processing (median) | End-to-end latency (median / p95 / max) |
|---|---|---|---|---|
| 550 | 550 | 10.00 Hz | 37.0 ms | 49.7 / 52.9 / 60.9 ms |

Zero drops. Raw output: `benchmarks/results/ros2_e2e.json`.

**Caveat.** This was measured inside one Docker Desktop VM on one Windows host; the specific loss
rate is almost certainly specific to that virtualized network path, not a property of ROS 2 or DDS
in general. The fix (reliable QoS for a shallow-history image topic) is a reasonable default
regardless -- it is a known pattern for exactly this situation -- but if you deploy on a real robot
network and don't see this failure mode, the fix should be a no-op for you, not a workaround for a
problem you don't have.

## 3. Idle-machine latency benchmark

New script: `benchmarks/bench_latency.py` (did not previously exist despite being referenced in
`benchmarks/README.md`). Runs `assets/demo.mp4` (550 frames) through the pipeline for three module
configurations, best-of-3 repeats, mediapipe backend, `conf=0.4`, `max_width=720`.

**Environment**: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64` (containerized; see caveat below),
mediapipe 1.2.2, onnxruntime 1.31.1, opencv 5.0.0. Full environment and per-module p95/p99/mean/max
in `benchmarks/results/latency_mediapipe.json`.

| Config | Modules | Median ms | p95 ms | p99 ms | Max ms |
|---|---|---|---|---|---|
| crowd_only | crowd | 25.3 | 28.7 | 31.7 | 37.0 |
| crowd_proximity_pose (ROS default) | crowd, proximity, pose | 35.3 | 37.3 | 39.0 | 42.0 |
| all_modules | crowd, proximity, pose, emotion | 37.2 | 39.5 | 40.6 | 42.5 |

Per-module median (ms), same runs:

| Config | detect | proximity | pose | emotion | risk |
|---|---|---|---|---|---|
| crowd_only | 25.3 | -- | -- | -- | 0.007 |
| crowd_proximity_pose | 25.1 | 0.012 | 10.1 | -- | 0.007 |
| all_modules | 24.9 | 0.012 | 10.2 | 2.0 | 0.008 |

Detection dominates; pose adds ~10 ms; proximity and risk fusion are negligible (<0.02 ms);
emotion adds ~2 ms when enabled.

**Caveat.** "Idle machine" was enforced by not running other workloads during the benchmark, not by
measuring background load -- the script does not check for contention (Windows has no
`os.getloadavg()`; cross-platform CPU-busy detection was judged out of scope for this script). This
ran inside the same Docker Desktop VM as the ROS test, not bare metal, and containerization/WSL2
overhead is not characterized separately from the pipeline's own cost. Treat these as *this
machine's* numbers, comparable to each other, not as a hardware-independent absolute.

## 4. Fall-detection dynamics fix

**Motivation.** The existing fall rule (`modules/pose_detector.py`) is purely postural: a frame is
"FALL detected" if the torso is tilted more than ~65 degrees from vertical (or the bounding box is
very wide and the torso isn't visible), evaluated independently per frame with no memory. URFD's
ADL set deliberately includes people lying on a bed or couch as a hard negative for exactly this
kind of rule (`benchmarks/bench_falls.py` docstring) -- a person resting there looks posturally
identical to a person who just fell, to a single-frame rule.

**Protocol.** `bench_falls.py --parity odd|even` restricts a run to sequences numbered odd/even
(`fall-01`, `fall-03`, ... = odd). Any threshold choice below was grid-searched against the
odd-numbered sequences only; every "held-out" table is a *single* run of the winning configuration
against the even-numbered sequences, never seen during tuning.

### Attempt 1 (discarded): bbox-centroid speed

A fall is a *collapse event*; resting is a static *state*. The pipeline already tracks each
person's `speed` (body heights moved per processed frame, via the IoU tracker in
`utils/tracking.py`), so the first attempt threaded it into the fall rule: once a person's posture
goes horizontal, require a recent burst of tracked speed (within a window, peak speed over a
threshold) to confirm "FALL detected"; otherwise downgrade to "fall risk".

**Tuning grid (odd sequences only; 15 fall + 20 ADL sequences, `fall_cue` alarm):**

| Window (frames) | Speed threshold | Sensitivity | FA / ADL-lying | FA / ADL-plain |
|---|---|---|---|---|
| -- (baseline, no gating) | -- | 9/15 | 4/8 | 3/12 |
| 10 | 0.06 | 7/15 | 4/8 | 1/12 |
| 10 | 0.10 | 4/15 | 4/8 | 1/12 |
| 15 | 0.08 | 5/15 | 4/8 | 1/12 |
| 6 | 0.12 | 4/15 | 4/8 | 1/12 |

This did not work. **FA/ADL-lying is exactly 4/8 at every threshold tried, including the
baseline** -- the gating had zero effect on the one false-alarm category it was built to fix,
while cutting fall sensitivity from 60% to as low as 27%. The likely reason: bbox-centroid speed
is a translation proxy, and the *act of lying down* on a bed or couch (an ADL activity) involves
centroid movement comparable to a fall's, while an in-place collapse (rotating without much
lateral travel) may not move the centroid much at all. Speed could not tell the two apart because
it isn't measuring the thing that actually differs between them. This attempt was reverted rather
than shipped with a dead `--tuning-didn't-help` knob; keeping it would have been a feature flag no
one should use.

### Attempt 2a (also discarded as first implemented): torso-tilt rotation rate

The thing that should actually differ between a fall and lying down is *how fast the torso
rotates into horizontal*, not how far the person's centroid travels. `pose_detector.py` already
computes a per-frame torso-tilt angle (0 degrees = upright, 90 = horizontal) from the pose
landmarks directly -- no dependency on the external tracker at all. First version: track each
person's tilt angle frame to frame; once horizontal, "FALL detected" requires a recent fast
rotation (within a window, peak rotation rate over a threshold); horizontal with no such recent
rotation downgrades to "fall risk". With no stable per-person id (the single-frame/no-detections
code path, see below), this gating was skipped and the original static rule applied unchanged.

Same grid shape as Attempt 1, same result: **FA/ADL-lying stuck at exactly 4/8 again**, at every
`(window, rate)` tried, while sensitivity still dropped (60% baseline down to 40-53%):

| Window (frames) | Rate threshold (deg/frame) | Sensitivity | FA / ADL-lying | FA / ADL-plain |
|---|---|---|---|---|
| -- (baseline, no gating) | -- | 9/15 | 4/8 | 3/12 |
| 5 | 10 | 8/15 | 4/8 | 2/12 |
| 5 | 20 | 8/15 | 4/8 | 1/12 |
| 5 | 30 | 6/15 | 4/8 | 1/12 |
| 8 | 20 | 8/15 | 4/8 | 1/12 |

**Root cause, found by inspecting which frames actually produced those 4 false alarms**
(`benchmarks/cache/urfd_mediapipe.jsonl`, the pipeline's raw per-frame output): every one of them
has `person_count == 0` -- the person-*detector* found nobody in that frame at all (an unusual,
often-horizontal posture is exactly where a person detector is most likely to miss). When that
happens, `analyze_pose` falls back to analysing the *whole frame* as a single legacy "person" with
no stable per-frame id, and the first implementation of both Attempt 1 and Attempt 2a explicitly
skipped dynamics gating on that path to preserve the original rule for truly one-off,
context-free calls. In production, though, that fallback fires repeatedly -- frame after frame --
for the very same still, undetected scene, so its dynamics are a meaningful signal too. The
gating logic just never got a chance to run on the frames that needed it.

### Attempt 2b (final): gate the no-detection fallback too, fail-safe on cold start

Same tilt-rotation-rate mechanism as 2a, with two changes: (1) the no-detection fallback (id 0)
now goes through the same gating as any other tracked person, since it repeats for the same scene
rather than being a one-off call; (2) a `pid`'s *first-ever* observation (no prior tilt to compare
against, so no rotation rate can be computed) now defaults to *confirming* the fall rather than
downgrading it -- a hospital safety cue should fail toward alerting on insufficient information,
not toward silence. See `src/humansense/modules/pose_detector.py` and the tests in
`tests/test_pose_rules.py`.

**Tuning grid (odd sequences only, `fall_cue` alarm):**

<!-- TUNING_TABLE -->

**Chosen configuration:** `FALL_COLLAPSE_WINDOW = <TBD>`, `FALL_COLLAPSE_RATE = <TBD>` degrees/frame.

**Held-out result (even sequences only; 15 fall + 20 ADL sequences, never used while tuning):**

<!-- HELDOUT_TABLE -->

**Reading these tables.** "Sensitivity" is fall *sequences* detected (at least one alarm from
0.5 s before the fall's labeled onset) out of 15 per half, with a Wilson 95% CI in the full JSON
output -- that CI is wide at this sample size; treat point estimates as indicative, not precise.
"FA / ADL-plain" and "FA / ADL-lying" are false-alarm sequence counts (out of 12 and 8
respectively) against, respectively, ordinary daily activities and the lying-down hard negatives
this fix specifically targets.

**Limitations.** This dataset is small (70 sequences total, 35 per half) and from one lab setup
with one camera angle and a small set of actors -- the odd/even split controls for
tuning-on-the-test-set, not for generalization to a real hospital's cameras, lighting, patients
using mobility aids, or clinical postures (see `docs/ETHICS.md`). The grid searched was also small
(5 points); a wider or continuous search was judged out of scope for this pass.

## 5. Threats to validity (all sections)

- All measurements ran inside one Docker Desktop VM (Windows host, WSL2 backend), not on bare
  metal or a real robot's compute/network stack. The ROS transport bug in Section 2 may be an
  artifact of that virtualization layer specifically.
- `assets/demo.mp4` (used for the ROS and latency tests) is a generic demo clip, not footage from
  a hospital corridor; it says nothing about accuracy, only about transport and compute behavior.
- The fall-detection numbers in Section 4 come from one public dataset recorded in one room; see
  `docs/LICENSING.md` for its terms and `docs/ETHICS.md` for why this project treats fall detection
  as a research-grade cue, never a certified safety function.

## References

*(drafted from general knowledge, not independently re-verified for this report -- check before
citing further, per the note at the top of this document)*

- Kwolek, B., & Kepski, M. (2014). Human fall detection on embedded platform using depth maps and
  wireless accelerometer. *Computer Methods and Programs in Biomedicine*, 117(3), 489-501. --
  the UR Fall Detection (URFD) dataset used in Section 4.
- Lugaresi, C., Tang, J., Nash, H., et al. (2019). MediaPipe: A Framework for Building Perception
  Pipelines. *arXiv:1906.08172*.
- Open Robotics / ROS 2 design docs on Quality of Service policies (`docs.ros.org`, "About Quality
  of Service Settings") -- background on `RELIABLE` vs `BEST_EFFORT` and history depth, relevant
  to Section 2's fix. Not a peer-reviewed source; cite as documentation, not literature.
