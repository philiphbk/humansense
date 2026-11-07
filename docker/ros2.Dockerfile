# HumanSense + ROS 2 Jazzy.  Build from the repository root:
#   docker build -f docker/ros2.Dockerfile -t humansense-ros .
#   docker build -f docker/ros2.Dockerfile --build-arg WITH_YOLO=1 -t humansense-ros:yolo .
FROM ros:jazzy-ros-base

ARG WITH_YOLO=0
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HUMANSENSE_MODELS_DIR=/opt/humansense/models

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-venv python3-pip python3-colcon-common-extensions build-essential \
      ros-jazzy-vision-msgs ros-jazzy-rosidl-default-generators ros-jazzy-launch-testing-ros \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# A venv that can still see the ROS python packages (rclpy etc.)
RUN python3 -m venv --system-site-packages /opt/hs-venv
ENV PATH=/opt/hs-venv/bin:$PATH

WORKDIR /opt/humansense-src
COPY pyproject.toml README.md ./
COPY src ./src
RUN if [ "$WITH_YOLO" = "1" ]; then pip install ".[yolo]"; else pip install .; fi

# Fetch model files at build time so the container runs offline
RUN humansense download-models \
 && python -c "from hsemotion_onnx.facial_emotions import HSEmotionRecognizer as H; H(model_name='enet_b0_8_best_vgaf')" \
 && if [ "$WITH_YOLO" = "1" ]; then python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"; fi

# MediaPipe loads libEGL/libGLESv2 even for CPU inference (kept as a late layer to reuse the pip cache)
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

# ROS workspace. colcon must be run *through the venv's python* (python -m colcon): the `colcon`
# script has a /usr/bin/python3 shebang, and that interpreter is what ends up in the shebang of the
# generated node executables, which would then not find the `humansense` package.
WORKDIR /ws
COPY ros2 ./src
RUN . /opt/ros/jazzy/setup.sh && python -m colcon build --event-handlers console_cohesion+ \
 && echo '. /opt/ros/jazzy/setup.bash && . /ws/install/setup.bash' >> /etc/bash.bashrc

SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash && exec \"$@\"", "--"]
CMD ["bash"]
