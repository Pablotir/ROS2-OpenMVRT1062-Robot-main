# ROS 2 Humble for Ubuntu 22.04 (JetPack 6 / nano_llm base)
ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

ARG DEBIAN_FRONTEND=noninteractive

# ── Refresh ROS2 GPG key (avoids EXPKEYSIG failures in long-lived images) ─────
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg

# ── Core system deps ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3-pip python3-serial python3-opencv \
    ffmpeg libsm6 libxext6 \
    libatlas-base-dev libopenblas-dev libhdf5-dev \
    build-essential cmake git curl nano tmux \
    libusb-1.0-0-dev libjpeg-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Core ROS2 Humble packages (required — hard failure if missing) ────────────
RUN apt-get update && apt-get install -y \
    ros-humble-xacro \
    ros-humble-robot-state-publisher \
    ros-humble-tf2 \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-tf2-sensor-msgs \
    ros-humble-cv-bridge \
    ros-humble-vision-opencv \
    ros-humble-image-proc \
    ros-humble-nav2-msgs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Optional SLAM / Nav packages (soft failure — build continues if unavailable)
RUN apt-get update; \
    apt-get install -y ros-humble-slam-toolbox          2>/dev/null; \
    apt-get install -y ros-humble-navigation2           2>/dev/null; \
    apt-get install -y ros-humble-nav2-bringup          2>/dev/null; \
    apt-get install -y ros-humble-usb-cam               2>/dev/null; \
    apt-get install -y ros-humble-teleop-twist-keyboard 2>/dev/null; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

# ── Python libraries (pip) — only packages not available via apt ──────────────
# pyserial is installed via apt (python3-serial) to avoid the Jetson pip mirror.
RUN python3 -m pip install --no-cache-dir -i https://pypi.org/simple/ \
    "numpy<2,>=1.26" \
    pillow apriltag requests \
    "opencv-python-headless"

# ── Workspace setup ───────────────────────────────────────────────────────────
ENV ROS_WS=/root/ros2_ws
WORKDIR $ROS_WS
RUN mkdir -p src

# Copy workspace and entrypoint
COPY ./src ./src
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Use bash login shell to allow sourcing
SHELL ["/bin/bash", "-lc"]

# Build the full workspace
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd $ROS_WS && \
    colcon build --symlink-install \
        --cmake-args -DCMAKE_BUILD_TYPE=Release

# Add both ROS2 and workspace to bashrc
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc && \
    echo "source ${ROS_WS}/install/setup.bash" >> /root/.bashrc

ENTRYPOINT ["/entrypoint.sh"]
