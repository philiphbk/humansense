# Contributing

Thanks for helping. The short version:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,app]"        # add ,yolo for the YOLO backend
humansense download-models
pytest                              # fast: no network or camera needed after the model download
ruff check src tests
```

* New cues or rules need a unit test in `tests/` that builds landmarks/detections by hand (see
  `tests/test_pose_rules.py`), plus - if you change behaviour - a before/after number from
  `benchmarks/`. Do not tune thresholds on the benchmark's test data; say how you chose them.
* ROS changes: build and test in the container (`docker build -f docker/ros2.Dockerfile -t humansense-ros .`
  then `scripts/ros2_e2e.sh`).
* Don't commit model weights, datasets or recordings of people.
