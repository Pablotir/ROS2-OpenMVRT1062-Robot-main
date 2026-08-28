#!/bin/bash
# Run this script with SUDO on the Jetson HOST (outside the container)
# ──────────────────────────────────────────────────────────────────────
# NOTE: Both RoboClaws report the same USB serial number (Jetson hub serial),
# so we match by PHYSICAL USB PORT PATH instead.
#
# LEFT  RoboClaw → Jetson port 1-2.4  (USB-C port)
# RIGHT RoboClaw → Jetson port 1-2.2  (USB-A port)
#
# This means: ALWAYS plug each RoboClaw into the same physical port.
# Label the ports on the Jetson to avoid confusion.
# ──────────────────────────────────────────────────────────────────────

echo "Setting up udev rules for the Jetson robot hardware..."

cat <<EOF > /etc/udev/rules.d/99-robot.rules
# ============================================================
# LDROBOT STL-27L LiDAR
# Silicon Labs CP210x UART Bridge (idVendor=10c4, idProduct=ea60)
# Always the only CP210x device on this robot.
# ============================================================
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", SYMLINK+="lidar", MODE="0666"

# ============================================================
# RoboClaw 2x15A — LEFT Side Controller
# (M1 = Rear-Left wheel, M2 = Front-Left wheel)
# Physical Jetson USB port: 1-2.4
# ============================================================
SUBSYSTEM=="tty", KERNELS=="1-2.4", SYMLINK+="roboclaw_left", MODE="0666"

# ============================================================
# RoboClaw 2x15A — RIGHT Side Controller
# (M1 = Rear-Right wheel, M2 = Front-Right wheel)
# Physical Jetson USB port: 1-2.2
# ============================================================
SUBSYSTEM=="tty", KERNELS=="1-2.2", SYMLINK+="roboclaw_right", MODE="0666"

# ============================================================
# SO-ARM101 Servo Motor Controller
# If your arm adapter uses CH340 (idVendor=1a86, idProduct=7523):
# SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", SYMLINK+="arm_controller", MODE="0666"
# Or if matched by physical USB port:
# Run: udevadm info -a -n /dev/ttyACM0 (or /dev/ttyUSB0) | grep -m 1 'KERNELS=="'
# And set KERNELS=="..." below:
# ============================================================
# SUBSYSTEM=="tty", KERNELS=="1-2.3", SYMLINK+="arm_controller", MODE="0666"

# NOTE: Arduino Mega mapping REMOVED — retired, replaced by RoboClaws.
EOF

echo "Reloading udev rules..."
udevadm control --reload-rules
udevadm trigger

echo ""
echo "Done! Verify existing devices with:"
echo "  ls -la /dev/roboclaw_left /dev/roboclaw_right /dev/lidar /dev/arm_controller 2>/dev/null"

