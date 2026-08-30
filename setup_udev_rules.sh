#!/bin/bash
# setup_udev_rules.sh — Run with SUDO on the Jetson HOST (outside the container)
# ─────────────────────────────────────────────────────────────────────────────
# Sets up udev permissions for:
#   1. Intel RealSense D405 (libusb access & power state management)
#   2. LDROBOT STL-27L LiDAR (CP210x UART bridge -> /dev/lidar)
#   3. RoboClaw 2x15A motor controllers (Left & Right -> /dev/roboclaw_*)
#   4. SO-ARM101 Servo Controller (-> /dev/arm_controller)
# ─────────────────────────────────────────────────────────────────────────────

if [ "$EUID" -ne 0 ]; then
  echo "[ERROR] Please run this script with sudo: sudo ./setup_udev_rules.sh"
  exit 1
fi

echo "Installing Intel RealSense udev rules on Jetson host..."
cat <<'EOF' > /etc/udev/rules.d/99-realsense-libusb.rules
# Intel RealSense D405 / D400 series rules
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b5b", MODE:="0666", GROUP:="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b5c", MODE:="0666", GROUP:="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b3a", MODE:="0666", GROUP:="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b07", MODE:="0666", GROUP:="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0ad3", MODE:="0666", GROUP:="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b4b", MODE:="0666", GROUP:="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", MODE:="0666", GROUP:="plugdev"
EOF

echo "Installing Jetson Robot hardware udev rules..."
cat <<'EOF' > /etc/udev/rules.d/99-robot.rules
# ============================================================
# LDROBOT STL-27L LiDAR
# Silicon Labs CP210x UART Bridge (idVendor=10c4, idProduct=ea60)
# ============================================================
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", SYMLINK+="lidar", MODE="0666"

# ============================================================
# RoboClaw 2x15A — LEFT Side Controller
# ============================================================
SUBSYSTEM=="tty", KERNELS=="1-2.4", SYMLINK+="roboclaw_left", MODE="0666"

# ============================================================
# RoboClaw 2x15A — RIGHT Side Controller
# ============================================================
SUBSYSTEM=="tty", KERNELS=="1-2.2", SYMLINK+="roboclaw_right", MODE="0666"

# ============================================================
# SO-ARM101 Servo Motor Controller (Feetech / CH343 / CH9102 / FTDI)
# ============================================================
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", SYMLINK+="arm_controller", MODE="0666"
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", SYMLINK+="arm_controller", MODE="0666"
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", SYMLINK+="arm_controller", MODE="0666"
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d4", SYMLINK+="arm_controller", MODE="0666"
KERNEL=="ttyACM*", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", SYMLINK+="arm_controller", MODE="0666"
EOF

echo "Reloading udev rules..."
udevadm control --reload-rules
udevadm trigger

echo ""
echo "=========================================================="
echo " [SUCCESS] udev rules installed and reloaded!"
echo " IMPORTANT: Unplug the RealSense D405 USB cable, then"
echo "            plug it back into the Jetson to refresh power."
echo "=========================================================="
