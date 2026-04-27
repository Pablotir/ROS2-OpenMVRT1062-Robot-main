# ──────────────────────────────────────────────────────────────────────────────
# ros2.Dockerfile — Custom robot image built on top of dustynv/nano_llm
# ──────────────────────────────────────────────────────────────────────────────
# Bakes in all apt packages and the built workspace so the container is
# ready to launch immediately on every `docker compose up` — no setup needed.
#
# Build (on Jetson, run once or after dependency changes):
#   docker compose build
#
# Run:
#   docker compose up -d
#   docker exec -it jetson_bot_vila bash
#   ros2 launch jetson_bot_slam ai_slam_explore.launch.py

FROM dustynv/nano_llm:humble-r36.3.0

ARG DEBIAN_FRONTEND=noninteractive

# ── Fix ROS2 GPG key ──────────────────────────────────────────────────────────
RUN rm -f /usr/share/keyrings/ros-archive-keyring.gpg && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

# ── Fix broken packages from nano_llm base ────────────────────────────────────
RUN dpkg --configure -a 2>/dev/null || true && \
    apt-get install -f -y 2>/dev/null || true

# ── Install ROS2 Humble packages ──────────────────────────────────────────────
# Note: nano_llm ships OpenCV 4.8.1 (custom). Ubuntu apt has 4.5.4.
# We use --force-overwrite for header conflicts.
# cv-bridge / image-proc / vision-opencv are excluded — they need libopencv-dev
# (4.5.4) which conflicts. Not needed without camera; add back later if needed.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        -o Dpkg::Options::="--force-overwrite" \
        python3-serial \
        python3-colcon-common-extensions \
        ros-humble-slam-toolbox \
        ros-humble-xacro \
        ros-humble-robot-state-publisher \
        ros-humble-joint-state-publisher \
        ros-humble-tf2 \
        ros-humble-tf2-ros \
        ros-humble-tf2-geometry-msgs \
        ros-humble-navigation2 \
        ros-humble-nav2-bringup \
        ros-humble-nav2-msgs \
        ros-humble-nav2-costmap-2d \
        ros-humble-visualization-msgs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir -i https://pypi.org/simple/ pyserial pillow

# ── Clone external ROS2 packages ──────────────────────────────────────────────
RUN mkdir -p /root/ros2_ws/src && \
    cd /root/ros2_ws/src && \
    git clone https://github.com/ldrobotSensorTeam/ldlidar_stl_ros2.git

# ── Copy robot workspace source ────────────────────────────────────────────────
COPY ./src /root/ros2_ws/src

# ── Build the workspace ────────────────────────────────────────────────────────
# Excluded:
#   multirobot_map_merge — blocked by OpenCV 4.8.1 vs 4.5.4 header conflict
#   explore_lite_msgs    — requires rosidl_generator_rs (Rust), not in this container
#   explore_lite         — depends on explore_lite_msgs
SHELL ["/bin/bash", "-lc"]
RUN source /opt/ros/install/setup.bash && \
    source /opt/ros/humble/setup.bash && \
    cd /root/ros2_ws && \
    colcon build --symlink-install \
        --packages-ignore multirobot_map_merge explore_lite_msgs explore_lite

# ── Write source_all.bash and add to .bashrc ──────────────────────────────────
RUN cat > /root/ros2_ws/source_all.bash << 'EOF'
#!/bin/bash
# Source all ROS2 layers in the correct order.
# nano_llm: ROS2 CLI at /opt/ros/install, Humble packages at /opt/ros/humble
source /opt/ros/install/setup.bash
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
EOF
RUN chmod +x /root/ros2_ws/source_all.bash && \
    echo 'source /root/ros2_ws/source_all.bash' >> /root/.bashrc

# ── Entrypoint ─────────────────────────────────────────────────────────────────
COPY ./entrypoint.sh /root/ros2_ws/entrypoint.sh
RUN chmod +x /root/ros2_ws/entrypoint.sh
ENTRYPOINT ["/root/ros2_ws/entrypoint.sh"]
CMD ["bash"]
