# ROS 2 Iron for Ubuntu 22.04 (JetPack 6+)
ARG ROS_DISTRO=iron
FROM ros:${ROS_DISTRO}-ros-base

ARG DEBIAN_FRONTEND=noninteractive

# Core system + guaranteed ROS Iron arm64 packages
RUN apt-get update && apt-get install -y \
    python3-pip python3-opencv ffmpeg libsm6 libxext6 \
    libatlas-base-dev libopenblas-dev libhdf5-dev \
    build-essential cmake git curl nano tmux \
    libusb-1.0-0-dev libjpeg-dev \
    ros-iron-xacro \
    ros-iron-robot-state-publisher \
    ros-iron-tf2-ros \
    ros-iron-tf2-geometry-msgs \
    ros-iron-cv-bridge \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Optional SLAM / Nav packages — semicolons so each is independent, build never fails here
RUN apt-get update; \
    apt-get install -y ros-iron-vision-opencv      2>/dev/null; \
    apt-get install -y ros-iron-slam-toolbox       2>/dev/null; \
    apt-get install -y ros-iron-navigation2        2>/dev/null; \
    apt-get install -y ros-iron-nav2-bringup       2>/dev/null; \
    apt-get install -y ros-iron-teleop-twist-keyboard 2>/dev/null; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

# Python libraries
RUN python3 -m pip install --no-cache-dir \
    "opencv-python-headless" \
    "numpy<2,>=1.26" \
    pyserial pillow apriltag requests

# Workspace setup
ENV ROS_WS=/root/ros2_ws
WORKDIR $ROS_WS
RUN mkdir -p src

# Copy workspace and entrypoint
COPY ./src ./src
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Use bash login shell to allow sourcing
SHELL ["/bin/bash", "-lc"]

# Build the workspace (only our packages)
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd $ROS_WS && \
    colcon build --symlink-install \
        --packages-select robot_control jetson_bot_slam \
        --cmake-args -DCMAKE_BUILD_TYPE=Release

# Add both ROS2 and workspace to bashrc
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc && \
    echo "source ${ROS_WS}/install/setup.bash" >> /root/.bashrc

ENTRYPOINT ["/entrypoint.sh"]

