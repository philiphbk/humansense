#!/usr/bin/env bash
# End-to-end ROS 2 test in Docker: replay an image directory / video as a camera topic, run the
# HumanSense node, and check + measure its output -- including /humansense/detections (always)
# and, when annotated=true, /humansense/annotated.
#
#   scripts/ros2_e2e.sh <sequence_dir_or_video> <result.json> [expect-level] [backend] [fps] [annotated]
#   ROBOT_X=0.5 ROBOT_Y=0.8 scripts/ros2_e2e.sh ...   # optional, passed through to the node
set -euo pipefail
SRC="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; OUT="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
LEVEL="${3:-}"; BACKEND="${4:-mediapipe}"; FPS="${5:-30}"; ANNOTATED="${6:-false}"
ROBOT_X="${ROBOT_X:-nan}"; ROBOT_Y="${ROBOT_Y:-nan}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
EXPECT=""; [ -n "$LEVEL" ] && EXPECT="--expect-level $LEVEL"
ANNOTATED_CHECK=""; [ "$ANNOTATED" = "true" ] && ANNOTATED_CHECK="--expect-annotated"
docker run --rm --memory=4g \
  -v "$SRC:/data/src:ro" -v "$(dirname "$OUT"):/out" \
  -v "$HERE/ros2/humansense_ros/test:/t:ro" \
  "${IMAGE:-humansense-ros}" bash -c "
    ros2 launch humansense_ros demo.launch.py source:=/data/src fps:=$FPS loop:=false \
        annotated:=$ANNOTATED robot_x:=$ROBOT_X robot_y:=$ROBOT_Y backend:=$BACKEND \
        > /tmp/launch.log 2>&1 &
    python3 /t/e2e_check.py --min-msgs 20 $EXPECT $ANNOTATED_CHECK --out /out/$(basename "$OUT")
    rc=\$?; pkill -f 'ros2 launch' || true; tail -5 /tmp/launch.log; exit \$rc"
