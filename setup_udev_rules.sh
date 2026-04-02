#!/bin/bash
# Run this script with SUDO on the Jetson HOST (outside the container)

echo "Setting up udev rules for the Jetson robot hardware..."

cat <<EOF > /etc/udev/rules.d/99-robot.rules
# Arduino Mega
# CH340 serial chip
SUBSYSTEM=="tty", ENV{ID_SERIAL}=="1a86_USB2.0-Serial", SYMLINK+="arduino", MODE="0666"

# LDROBOT STL-27L LiDAR
# Silicon Labs CP2102
SUBSYSTEM=="tty", ENV{ID_SERIAL}=="Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001", SYMLINK+="lidar", MODE="0666"
EOF

echo "Reloading udev rules..."
udevadm control --reload-rules
udevadm trigger

echo "Done! You can verify by running: ls -la /dev/arduino /dev/lidar"
