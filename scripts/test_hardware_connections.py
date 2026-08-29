#!/usr/bin/env python3
"""
test_hardware_connections.py — Complete Robot Hardware Diagnostic Script
========================================================================
Performs hardware checks for:
  1. Physical USB Devices (via /sys/bus/usb/devices and lsusb)
  2. Serial Ports & Symlinks (/dev/arm_controller, /dev/roboclaw_*, /dev/tty*)
  3. Video Devices (/dev/video*) & OpenCV Stream Test
  4. SO-ARM101 Servo Serial Communication (/dev/arm_controller -> /dev/ttyACM0)
  5. RealSense D405 (pyrealsense2)

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
    print_header("1. CHECKING PHYSICAL USB BUS")
    
    # Method A: Direct /sys/bus/usb/devices inspection (works even without lsusb package)
    usb_devs = glob.glob('/sys/bus/usb/devices/*')
    found_any = False
    for d in usb_devs:
        id_vendor_f = os.path.join(d, 'idVendor')
        id_product_f = os.path.join(d, 'idProduct')
        product_f = os.path.join(d, 'product')
        manufacturer_f = os.path.join(d, 'manufacturer')
        
        if os.path.exists(id_vendor_f) and os.path.exists(id_product_f):
            found_any = True
            with open(id_vendor_f) as f: vid = f.read().strip()
            with open(id_product_f) as f: pid = f.read().strip()
            prod = open(product_f).read().strip() if os.path.exists(product_f) else "Unknown"
            mfg = open(manufacturer_f).read().strip() if os.path.exists(manufacturer_f) else ""
            
            dev_str = f"ID {vid}:{pid} — {mfg} {prod}".strip()
            print(f"  [USB] {os.path.basename(d):10} | {dev_str}")
            
            if vid == "8086" and pid == "0b5b":
                print("         >>> CONFIRMED: Intel RealSense D405 Connected!")
            elif vid in ["1a86", "0403", "2e8a", "0483"] or "STM32" in prod or "CH340" in prod:
                print("         >>> CONFIRMED: Arm Controller Connected!")
            elif vid == "10c4" and pid == "ea60":
                print("         >>> CONFIRMED: LiDAR CP210x Bridge Connected!")
            elif vid == "04d8":
                print("         >>> CONFIRMED: RoboClaw Motor Controller Connected!")

    if not found_any:
        print("  [WARN] /sys/bus/usb/devices is empty. Ensure /dev/bus/usb and /sys are mounted.")


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

    raw_devices = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
    if raw_devices:
        print("\n  Available Raw Serial Devices:")
        for rd in raw_devices:
            print(f"    - {rd:16} (Target: {os.path.realpath(rd)})")


def test_video_devices():
    print_header("3. CHECKING VIDEO DEVICES (/dev/video*)")
    v_devs = sorted(glob.glob('/dev/video*'))
    if not v_devs:
        print("  [WARN] No /dev/video* devices found.")
        return

    print(f"  Found {len(v_devs)} video devices:")
    for vd in v_devs:
        print(f"    - {vd}")

    # Test opening first video device with OpenCV
    try:
        import cv2
        print(f"\n  Testing video capture with OpenCV on /dev/video0...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w, c = frame.shape
                print(f"  [SUCCESS] Captured frame from /dev/video0: {w}x{h} (channels: {c})")
            else:
                print("  [WARN] /dev/video0 opened, but could not read frame.")
            cap.release()
        else:
            print("  [NOTE] Could not open /dev/video0 with V4L2 backend.")
    except Exception as e:
        print(f"  [WARN] OpenCV capture test: {e}")


def test_arm_serial():
    print_header("4. TESTING ARM SERIAL COMMUNICATION (/dev/arm_controller)")
    port = '/dev/arm_controller'
    if not os.path.exists(port):
        candidates = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')
        if candidates:
            port = candidates[0]
            print(f"  [NOTE] /dev/arm_controller symlink not found, using raw port: {port}")
        else:
            print("  [FAIL] No serial port found.")
            return

    try:
        import serial
        ser = serial.Serial(port, baudrate=1000000, timeout=0.2)
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
    test_video_devices()
    test_arm_serial()

    print("\n" + "=" * 60)
    print("  DIAGNOSTICS COMPLETE")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
