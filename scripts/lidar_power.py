#!/usr/bin/env python3
"""
lidar_power.py — Software Motor & Laser Control for STL-27L via UART
====================================================================
Sends direct serial start/stop commands over the Jetson UART port (/dev/ttyTHS1)
using your existing 4-pin wiring.

Wiring (Jetson 40-Pin Header):
  - Pin 2  -> STL-27L VCC (5V)
  - Pin 6  -> STL-27L GND
  - Pin 8  -> STL-27L RX (Jetson UART1_TX)
  - Pin 10 -> STL-27L TX (Jetson UART1_RX)

Serial Commands (at 921600 baud):
  - '0' (ASCII 0x30) -> Stops LiDAR motor rotation and pauses laser output
  - '1' (ASCII 0x31) -> Starts LiDAR motor rotation and resumes streaming

Usage:
  python3 lidar_power.py --off    # Stop LiDAR motor completely
  python3 lidar_power.py --on     # Start LiDAR motor
"""

import sys
import time
import argparse

DEFAULT_PORT = '/dev/ttyTHS1'
DEFAULT_BAUD = 921600


def set_lidar_state(turn_on: bool, port: str = DEFAULT_PORT, baud: int = DEFAULT_BAUD) -> bool:
    try:
        import serial
    except ImportError:
        print("[ERROR] pyserial is not installed. Install with: pip3 install pyserial")
        return False

    cmd = b'0' if not turn_on else b'1'
    action_text = "STOPPED (Motor Paused)" if not turn_on else "STARTED (Spinning)"

    try:
        ser = serial.Serial(
            port=port,
            baudrate=baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.5
        )

        # Send command byte multiple times to ensure buffer clears
        for _ in range(3):
            ser.write(cmd)
            time.sleep(0.05)

        ser.flush()
        ser.close()
        print(f"[LiDAR UART] Sent command '{cmd.decode('ascii')}' on {port} -> LiDAR {action_text}")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to access {port}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="STL-27L UART Motor Start/Stop Controller")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--on', action='store_true', help='Start LiDAR motor')
    group.add_argument('--off', action='store_true', help='Stop LiDAR motor')
    parser.add_argument('--port', type=str, default=DEFAULT_PORT, help=f'UART port (default: {DEFAULT_PORT})')
    parser.add_argument('--baud', type=int, default=DEFAULT_BAUD, help=f'Baud rate (default: {DEFAULT_BAUD})')

    args = parser.parse_args()
    success = set_lidar_state(turn_on=args.on, port=args.port, baud=args.baud)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
