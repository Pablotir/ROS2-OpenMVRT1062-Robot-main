#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# setup_vila.sh — One-time setup inside the dustynv/nano_llm container
# ──────────────────────────────────────────────────────────────────────────────
# Run this ONCE after first `docker exec -it jetson_bot_vila bash`:
#   bash /root/ros2_ws/setup_vila.sh

set -e

echo "══════════════════════════════════════════════════════════════"
echo "  Jetson Bot SLAM + VILA 2.7B — First-Time Setup"
echo "══════════════════════════════════════════════════════════════"

# ── 0. Fix expired ROS2 GPG key ──────────────────────────────────────────────
echo ""
echo "▸ Refreshing ROS2 apt signing key..."
rm -f /usr/share/keyrings/ros-archive-keyring.gpg
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "  ✔ GPG key refreshed"

# ── 1. Install ROS2 Humble packages from apt ─────────────────────────────────
# These install to /opt/ros/humble/ alongside the container's source-built
# ROS2 core at /opt/ros/install/. Both must be sourced at runtime.
echo ""
echo "▸ Installing ROS2 Humble packages..."
apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-usb-cam \
    ros-humble-rtabmap-ros \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-xacro \
    ros-humble-robot-state-publisher \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-cv-bridge \
    ros-humble-vision-opencv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Fix any dpkg conflicts (container's opencv-dev 4.8.1 vs apt's 4.5.4 headers)
dpkg --configure -a --force-overwrite 2>/dev/null || true

# explore_lite (build from source — not in apt for Humble)
echo ""
echo "▸ Setting up explore_lite..."
cd /root/ros2_ws/src
if [ ! -d "m-explore-ros2" ]; then
    git clone https://github.com/robo-friends/m-explore-ros2.git
fi

# ── 2. Python dependencies ───────────────────────────────────────────────────
# Use -i to bypass the container's broken jetson.webredirect.org pip index
echo ""
echo "▸ Installing Python dependencies..."
pip3 install --no-cache-dir -i https://pypi.org/simple/ \
    pyserial pillow "numpy<2,>=1.26" opencv-python-headless

# ── 3. Build the workspace ───────────────────────────────────────────────────
# Source BOTH ROS2 installations:
#   /opt/ros/humble/       — apt-installed packages (Nav2, RTAB-Map, etc.)
#   /opt/ros/install/      — container's source-built ROS2 core
echo ""
echo "▸ Building ROS2 workspace..."
cd /root/ros2_ws
source /opt/ros/humble/setup.bash
source /opt/ros/install/setup.bash

colcon build --symlink-install \
    --packages-select explore_lite_msgs explore_lite robot_control jetson_bot_slam

source install/setup.bash

# ── 4. Write a convenience source script ─────────────────────────────────────
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

# ── 5. Pre-download VILA model (optional) ────────────────────────────────────
echo ""
echo "▸ Pre-downloading VILA 2.7B model (this may take a few minutes)..."
python3 -c "
from nano_llm import NanoLLM
print('Downloading VILA 2.7B...')
model = NanoLLM.from_pretrained('Efficient-Large-Model/VILA-2.7b', api='awq')
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
