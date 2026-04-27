#!/bin/bash
set -e

# Source ROS2 environment.
# source_all.bash is baked into the image by ros2.Dockerfile and handles
# all three layers: /opt/ros/install → /opt/ros/humble → ~/ros2_ws/install
if [ -f "/root/ros2_ws/source_all.bash" ]; then
    source /root/ros2_ws/source_all.bash
else
    # Fallback for plain nano_llm image (before docker compose build)
    [ -f "${ROS_ROOT}/install/setup.bash" ] && source "${ROS_ROOT}/install/setup.bash"
    [ -f "/opt/ros/humble/setup.bash" ]     && source "/opt/ros/humble/setup.bash"
    [ -f "/root/ros2_ws/install/setup.bash" ] && source "/root/ros2_ws/install/setup.bash"
    export AMENT_PREFIX_PATH=${AMENT_PREFIX_PATH}:${CMAKE_PREFIX_PATH}
fi

exec "$@"
