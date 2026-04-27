#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# setup_vila.sh — One-time setup inside the dustynv/nano_llm container
# ──────────────────────────────────────────────────────────────────────────────
# Run this ONCE after first `docker exec -it jetson_bot_vila bash`:
#   bash /root/ros2_ws/setup_vila.sh

set -e

echo "══════════════════════════════════════════════════════════════"
echo "  Jetson Bot SLAM + VILA 2.7B — First-Time Container Setup"
echo "══════════════════════════════════════════════════════════════"

# ── 0. Fix expired ROS2 GPG key ──────────────────────────────────────────────
echo ""
echo "▸ Refreshing ROS2 apt signing key..."
rm -f /usr/share/keyrings/ros-archive-keyring.gpg
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "  ✔ GPG key refreshed"

# ── 1. Fix any broken packages from previous runs ────────────────────────────
echo ""
echo "▸ Fixing any broken packages..."
dpkg --configure -a 2>/dev/null || true
apt-get install -f -y 2>/dev/null || true

# ── 2. Install ROS2 Humble packages ──────────────────────────────────────────
# IMPORTANT: This container has OpenCV 4.8.1 (custom nano_llm build).
# Packages that pull in libopencv-dev (4.5.4 from apt) will conflict.
# We use --force-overwrite to handle header file collisions.
# cv-bridge / image-proc / vision-opencv are intentionally excluded here —
# they're not needed without a camera, and nano_llm already has Python OpenCV.
echo ""
echo "▸ Installing ROS2 Humble packages..."
DEBIAN_FRONTEND=noninteractive apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
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

echo "  ✔ Core packages installed"

# ── 3. Clone external packages ───────────────────────────────────────────────
echo ""
echo "▸ Setting up external packages..."
cd /root/ros2_ws/src

# STL-27L LiDAR driver
if [ ! -d "ldlidar_stl_ros2" ]; then
    echo "  Cloning ldlidar_stl_ros2..."
    git clone https://github.com/ldrobotSensorTeam/ldlidar_stl_ros2.git
fi

# ── 4. Device permissions ─────────────────────────────────────────────────────
echo ""
echo "▸ Setting device permissions..."
chmod 777 /dev/roboclaw_left  2>/dev/null || echo "  ⚠ /dev/roboclaw_left not found (plug in RoboClaw)"
chmod 777 /dev/roboclaw_right 2>/dev/null || echo "  ⚠ /dev/roboclaw_right not found (plug in RoboClaw)"
chmod 777 /dev/lidar          2>/dev/null || echo "  ⚠ /dev/lidar not found (plug in LiDAR)"

# ── 5. Python dependencies ────────────────────────────────────────────────────
echo ""
echo "▸ Installing Python dependencies..."
pip3 install --no-cache-dir -i https://pypi.org/simple/ pyserial pillow

# ── 6. Build the workspace ────────────────────────────────────────────────────
echo ""
echo "▸ Building ROS2 workspace..."
cd /root/ros2_ws
source /opt/ros/install/setup.bash
source /opt/ros/humble/setup.bash

# Excluded packages:
#   multirobot_map_merge — blocked by OpenCV 4.8.1 vs 4.5.4 header conflict
#   explore_lite_msgs    — requires rosidl_generator_rs (Rust), not in this container
#   explore_lite         — depends on explore_lite_msgs
colcon build --symlink-install \
    --packages-ignore multirobot_map_merge explore_lite_msgs explore_lite

source install/setup.bash

# ── 7. Write convenience source script ────────────────────────────────────────
echo ""
echo "▸ Writing source_all.bash..."
cat > /root/ros2_ws/source_all.bash << 'EOF'
#!/bin/bash
# Source all ROS2 layers in the correct order.
# Usage: source /root/ros2_ws/source_all.bash
#
# nano_llm container layout:
#   /opt/ros/install  — nano_llm custom ROS2 (has ros2 CLI binary)
#   /opt/ros/humble   — standard Humble apt packages (slam_toolbox, xacro, etc)
#   ~/ros2_ws/install — our robot workspace overlay
source /opt/ros/install/setup.bash
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
EOF
chmod +x /root/ros2_ws/source_all.bash

# Add to .bashrc if not already there
if ! grep -q "source_all.bash" /root/.bashrc 2>/dev/null; then
    echo 'source /root/ros2_ws/source_all.bash' >> /root/.bashrc
    echo "  ✔ Added source_all.bash to ~/.bashrc"
fi

# ── 8. Pre-download VILA model (disabled — uncomment when ready for AI labelling)
# echo ""
# echo "▸ Pre-downloading VILA 2.7B model (this may take a few minutes)..."
# python3 -c "
# from nano_llm import NanoLLM
# print('Downloading VILA 2.7B...')
# model = NanoLLM.from_pretrained('Efficient-Large-Model/VILA-2.7b', api='mlc', quantization='q4f16_ft')
# print('Model downloaded and ready!')
# del model
# " 2>&1 || echo "⚠  Model download skipped (will download on first inference)"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Every new shell: source /root/ros2_ws/source_all.bash"
echo "  (already added to ~/.bashrc — new shells get it automatically)"
echo ""
echo "  Run the robot stack:"
echo "    ros2 launch jetson_bot_slam ai_slam_explore.launch.py"
echo "══════════════════════════════════════════════════════════════"
