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
# LDROBOT STL-27L LiDAR (Silicon Labs CP2102)
# ============================================================
SUBSYSTEM=="tty", ENV{ID_SERIAL}=="Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001", SYMLINK+="lidar", MODE="0666"

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
# TO FILL IN once arm is connected — run:
#   udevadm info -a -n /dev/ttyACM2 | grep 'KERNELS'
# Then replace FILL_IN_ARM_PORT below with the port (e.g. 1-2.3)
# ============================================================
# SUBSYSTEM=="tty", KERNELS=="FILL_IN_ARM_PORT", SYMLINK+="arm_controller", MODE="0666"

# NOTE: Arduino Mega mapping REMOVED — retired, replaced by RoboClaws.
EOF

echo "Reloading udev rules..."
udevadm control --reload-rules
udevadm trigger

echo ""
echo "Done! Verify with:"
echo "  ls -la /dev/roboclaw_left /dev/roboclaw_right /dev/lidar"
