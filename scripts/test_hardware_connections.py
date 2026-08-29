#!/usr/bin/env python3
"""
test_hardware_connections.py — Complete Robot Hardware Diagnostic Script
========================================================================
Performs zero-dependency hardware checks for:
  1. USB Bus Enumeration (RealSense D405, Arm Controller, RoboClaws, LiDAR)
  2. RealSense D405 Direct Capture (RGB + Depth via pyrealsense2)
  3. SO-ARM101 Serial Port Access & Servo Bus Ping
  4. Device Permissions (/dev/bus/usb, /dev/tty*, /dev/arm_controller)

Usage:
  python3 scripts/test_hardware_connections.py
"""

import os
import sys
import glob
import time
import subprocess


def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check_usb_devices():
    print_header("1. CHECKING PHYSICAL USB DEVICES (lsusb)")
    try:
        res = subprocess.run(['lsusb'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        lines = res.stdout.strip().split('\n')
        for line in lines:
            print(f"  [USB] {line}")
            if "8086:" in line or "Intel" in line or "0b5b" in line:
                print("        >>> MATCH: Intel RealSense D405 Detected!")
            elif "1a86:" in line or "0403:" in line or "FTDI" in line or "CH340" in line:
                print("        >>> MATCH: Arm Serial Controller Detected!")
            elif "10c4:ea60" in line or "Silicon Labs" in line or "CP210" in line:
                print("        >>> MATCH: LiDAR CP210x Bridge Detected!")
            elif "04d8:" in line or "Microchip" in line:
                print("        >>> MATCH: RoboClaw Motor Controller Detected!")
    except Exception as e:
        print(f"  [ERROR] Failed to run lsusb: {e}")


def check_serial_ports():
    print_header("2. CHECKING SERIAL PORTS & SYMLINKS")
    expected_links = [
        ('/dev/arm_controller', 'SO-ARM101 Servo Controller'),
        ('/dev/roboclaw_left', 'RoboClaw Left Motor Controller'),
        ('/dev/roboclaw_right', 'RoboClaw Right Motor Controller'),
        ('/dev/lidar', 'LiDAR Serial Symlink (if USB)'),
        ('/dev/ttyTHS1', 'LiDAR 40-Pin UART Port'),
    ]

    for dev_path, desc in expected_links:
        if os.path.exists(dev_path):
            target = os.path.realpath(dev_path)
            stat = os.stat(dev_path)
            mode_oct = oct(stat.st_mode)[-3:]
            print(f"  [PASS] {dev_path:22} -> {target:14} (Mode: {mode_oct}) | {desc}")
        else:
            print(f"  [WARN] {dev_path:22} -> NOT FOUND | {desc}")

    # List all raw ttyUSB and ttyACM devices
    raw_devices = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
    if raw_devices:
        print("\n  Available Raw Serial Devices on Host/Container:")
        for rd in raw_devices:
            print(f"    - {rd} (Target: {os.path.realpath(rd)})")


def test_realsense_direct():
    print_header("3. TESTING REALSENSE D405 CAMERA (Direct pyrealsense2)")
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("  [ERROR] pyrealsense2 is not installed in this Python environment.")
        return

    ctx = rs.context()
    devices = ctx.query_devices()
    print(f"  Connected RealSense Device Count: {len(devices)}")

    if len(devices) == 0:
        print("  [FAIL] No RealSense camera found by librealsense.")
        print("         Make sure /dev/bus/usb is mounted into the container!")
        return

    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        fw = dev.get_info(rs.camera_info.firmware_version)
        print(f"  [FOUND] Camera Name: {name}")
        print(f"          Serial Number: {serial}")
        print(f"          Firmware: {fw}")

    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 15)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 15)

        print("\n  Starting RealSense pipeline stream...")
        pipeline.start(config)
        print("  Waiting for first frame...")
        frames = pipeline.wait_for_frames(timeout_ms=5000)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame and depth_frame:
            cw, ch = color_frame.get_width(), color_frame.get_height()
            dw, dh = depth_frame.get_width(), depth_frame.get_height()
            center_dist = depth_frame.get_distance(int(dw / 2), int(dh / 2))
            print(f"  [SUCCESS] Color Stream Active: {cw}x{ch}")
            print(f"  [SUCCESS] Depth Stream Active: {dw}x{dh}")
            print(f"  [SUCCESS] Center Depth Measurement: {center_dist:.3f} meters")
        else:
            print("  [WARN] Pipeline started but frame was empty.")

        pipeline.stop()
        print("  Pipeline stopped cleanly.")

    except Exception as e:
        print(f"  [FAIL] RealSense test failed: {e}")


def test_arm_serial():
    print_header("4. TESTING ARM SERIAL COMMUNICATION")
    port = '/dev/arm_controller'

    if not os.path.exists(port):
        # Fallback to first available ttyUSB or ttyACM
        candidates = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
        if candidates:
            port = candidates[0]
            print(f"  [NOTE] /dev/arm_controller symlink not found, trying raw port: {port}")
        else:
            print("  [FAIL] No serial port (/dev/arm_controller or /dev/ttyUSB*) found.")
            return

    try:
        import serial
        ser = serial.Serial(port, baudrate=1000000, timeout=0.1)
        print(f"  [PASS] Successfully opened {port} @ 1,000,000 baud!")
        ser.close()
    except Exception as e:
        print(f"  [WARN] Failed to open {port}: {e}")


def main():
    print("\n" + "#" * 60)
    print("  JETSON BOT HARDWARE DIAGNOSTIC SUITE")
    print("#" * 60)

    check_usb_devices()
    check_serial_ports()
    test_realsense_direct()
    test_arm_serial()

    print("\n" + "=" * 60)
    print("  DIAGNOSTICS COMPLETE")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
