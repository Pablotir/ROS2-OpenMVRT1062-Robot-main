#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# setup_vila.sh — One-time setup inside the dustynv/nano_llm container
# ──────────────────────────────────────────────────────────────────────────────
# Run this ONCE after first `docker exec -it jetson_bot_vila bash`:
#   bash /root/ros2_ws/setup_vila.sh

set -e

echo "══════════════════════════════════════════════════════════════"
echo "  Jetson Bot SLAM + VILA 2.7B + STL-27L LiDAR — First-Time Setup"
echo "══════════════════════════════════════════════════════════════"

# ── 0. Fix expired ROS2 GPG key ──────────────────────────────────────────────
echo ""
echo "▸ Refreshing ROS2 apt signing key..."
rm -f /usr/share/keyrings/ros-archive-keyring.gpg
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "  ✔ GPG key refreshed"

# ── 1. Install ROS2 Humble packages from apt ─────────────────────────────────
echo ""
echo "▸ Installing ROS2 Humble packages..."
apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-usb-cam \
    ros-humble-rtabmap-ros \
    ros-humble-slam-toolbox \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-xacro \
    ros-humble-robot-state-publisher \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-cv-bridge \
    ros-humble-vision-opencv \
    ros-humble-image-proc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Fix any dpkg conflicts (container's opencv-dev 4.8.1 vs apt's 4.5.4 headers)
dpkg --configure -a --force-overwrite 2>/dev/null || true

# ── 2. Clone external packages ───────────────────────────────────────────────
echo ""
echo "▸ Setting up external packages..."
cd /root/ros2_ws/src

# STL-27L LiDAR driver
if [ ! -d "ldlidar_stl_ros2" ]; then
    echo "  Cloning ldlidar_stl_ros2..."
    git clone https://github.com/ldrobotSensorTeam/ldlidar_stl_ros2.git
fi

# ── 3. Serial port permissions for LiDAR ─────────────────────────────────────
echo ""
echo "▸ Setting serial port permissions..."
chmod 777 /dev/arduino 2>/dev/null || echo "  ⚠ /dev/arduino not found (Arduino)"
chmod 777 /dev/lidar 2>/dev/null || echo "  ⚠ /dev/lidar not found (LiDAR)"

# ── 4. Python dependencies ───────────────────────────────────────────────────
echo ""
echo "▸ Installing Python dependencies..."
pip3 install --no-cache-dir -i https://pypi.org/simple/ \
    pyserial pillow "numpy<2,>=1.26" opencv-python-headless

# ── 5. Build the workspace ───────────────────────────────────────────────────
echo ""
echo "▸ Building ROS2 workspace..."
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source /opt/ros/install/setup.bash

colcon build --symlink-install \
    --packages-select \
    ldlidar_stl_ros2 \
    robot_control jetson_bot_slam

source install/setup.bash

# ── 6. Write a convenience source script ─────────────────────────────────────
cat > /root/ros2_ws/source_all.bash << 'EOF'
#!/bin/bash
# Source all three ROS2 layers in the correct order
source /opt/ros/humble/setup.bash
source /opt/ros/install/setup.bash
source /root/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
EOF
chmod +x /root/ros2_ws/source_all.bash
echo 'source /root/ros2_ws/source_all.bash' >> /root/.bashrc

# ── 7. Pre-download VILA model (optional) ────────────────────────────────────
echo ""
echo "▸ Pre-downloading VILA 2.7B model (this may take a few minutes)..."
python3 -c "
from nano_llm import NanoLLM
print('Downloading VILA 2.7B...')
model = NanoLLM.from_pretrained('Efficient-Large-Model/VILA-2.7b', api='awq', quantization='Efficient-Large-Model/VILA-2.7b')
print('Model downloaded and ready!')
del model
" 2>&1 || echo "⚠  Model download skipped (will download on first inference)"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Setup complete! Run the full stack with:"
echo ""
echo "    source /root/ros2_ws/source_all.bash"
echo "    ros2 launch jetson_bot_slam ai_slam_explore.launch.py"
echo "══════════════════════════════════════════════════════════════"
