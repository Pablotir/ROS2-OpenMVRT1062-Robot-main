#!/bin/bash
set -e

# Source ROS2
source /opt/ros/iron/setup.bash

# Source workspace if exists
if [ -f /root/ros2_ws/install/setup.bash ]; then
    source /root/ros2_ws/install/setup.bash
fi

# Export PYTHONPATH so cv_bridge is found
export PYTHONPATH=/opt/ros/iron/lib/python3.10/site-packages:$PYTHONPATH

# Execute command
exec "$@"

