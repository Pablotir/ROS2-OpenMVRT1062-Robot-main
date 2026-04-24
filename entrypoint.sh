#!/bin/bash
set -e

# Source ROS2 — nano_llm container uses $ROS_ROOT/install/setup.bash
# NOT the standard /opt/ros/humble/setup.bash path
ros_source_env() {
    if [ -f "$1" ]; then
        echo "sourcing $1"
        source "$1"
    fi
}

# Mirror the container's own /ros_entrypoint.sh logic
ros_source_env "${ROS_ROOT}/install/setup.bash"
# Fallbacks in case ROS_ROOT is unset
ros_source_env "/opt/ros/humble/install/setup.bash"
ros_source_env "/opt/ros/humble/setup.bash"

# Source the user workspace overlay
ros_source_env "/root/ros2_ws/install/setup.bash"

export AMENT_PREFIX_PATH=${AMENT_PREFIX_PATH}:${CMAKE_PREFIX_PATH}

# Execute command
exec "$@"
