#!/usr/bin/env python3
"""
test_lidar_gpio.py — Diagnostic tool for LiDAR on Jetson GPIO UART (/dev/ttyTHS1)
Supports LDROBOT STL-27L / LD19 / LD06 protocol (Packet Header: 0x54).
"""

import sys
import time
import argparse
import serial
import struct

def parse_args():
    parser = argparse.ArgumentParser(description="Test LiDAR over Jetson GPIO UART.")
    parser.add_argument("--port", default="/dev/ttyTHS1", help="Serial port (default: /dev/ttyTHS1)")
    parser.add_argument("--baud", type=int, default=921600, help="Baud rate (default: 921600 for STL-27L, try 230400 for LD19/LD06)")
    parser.add_argument("--timeout", type=float, default=2.0, help="Serial timeout in seconds")
    return parser.parse_args()

def crc8(data: bytes) -> int:
    crc_table = [
        0x00, 0x4d, 0x9a, 0xd7, 0x79, 0x34, 0xe3, 0xae, 0xf2, 0xbf, 0x68, 0x25, 0x8b, 0xc6, 0x11, 0x5c,
        0xa9, 0xe4, 0x33, 0x7e, 0xd0, 0x9d, 0x4a, 0x07, 0x5b, 0x16, 0xc1, 0x8c, 0x22, 0x6f, 0xb8, 0xf5,
        0x1f, 0x52, 0x85, 0xc8, 0x66, 0x2b, 0xfc, 0xb1, 0xed, 0xa0, 0x77, 0x3a, 0x94, 0xd9, 0x0e, 0x43,
        0xb6, 0xfb, 0x2c, 0x61, 0xcf, 0x82, 0x55, 0x18, 0x44, 0x09, 0xde, 0x93, 0x3d, 0x70, 0xa7, 0xea,
        0x3e, 0x73, 0xa4, 0xe9, 0x47, 0x0a, 0xdd, 0x90, 0xcc, 0x81, 0x56, 0x1b, 0xb5, 0xf8, 0x2f, 0x62,
        0x97, 0xda, 0x0d, 0x40, 0xee, 0xa3, 0x74, 0x39, 0x65, 0x28, 0xff, 0xb2, 0x1c, 0x51, 0x86, 0xcb,
        0x21, 0x6c, 0xbb, 0xf6, 0x58, 0x15, 0xc2, 0x8f, 0xd3, 0x9e, 0x49, 0x04, 0xaa, 0xe7, 0x30, 0x7d,
        0x88, 0xc5, 0x12, 0x5f, 0xf1, 0xbc, 0x6b, 0x26, 0x7a, 0x37, 0xe0, 0xad, 0x03, 0x4e, 0x99, 0xd4,
        0x7c, 0x31, 0xe6, 0xab, 0x05, 0x48, 0x9f, 0xd2, 0x8e, 0xc3, 0x14, 0x59, 0xf7, 0xba, 0x6d, 0x20,
        0xd5, 0x98, 0x4f, 0x02, 0xac, 0xe1, 0x36, 0x7b, 0x27, 0x6a, 0xbd, 0xf0, 0x5e, 0x13, 0xc4, 0x89,
        0x63, 0x2e, 0xf9, 0xb4, 0x1a, 0x57, 0x80, 0xcd, 0x91, 0xdc, 0x0b, 0x46, 0xe8, 0xa5, 0x72, 0x3f,
        0xca, 0x87, 0x50, 0x1d, 0xb3, 0xfe, 0x29, 0x64, 0x38, 0x75, 0xa2, 0xef, 0x41, 0x0c, 0xdb, 0x96,
        0x42, 0x0f, 0xd8, 0x95, 0x3b, 0x76, 0xa1, 0xec, 0xb0, 0xfd, 0x2a, 0x67, 0xc9, 0x84, 0x53, 0x1e,
        0xeb, 0xa6, 0x71, 0x3c, 0x92, 0xdf, 0x08, 0x45, 0x19, 0x54, 0x83, 0xce, 0x60, 0x2d, 0xfa, 0xb7,
        0x5d, 0x10, 0xc7, 0x8a, 0x24, 0x69, 0xbe, 0xf3, 0xaf, 0xe2, 0x35, 0x78, 0xd6, 0x9b, 0x4c, 0x01,
        0xf4, 0xb9, 0x6e, 0x23, 0x8d, 0xc0, 0x17, 0x5a, 0x06, 0x4b, 0x9c, 0xd1, 0x7f, 0x32, 0xe5, 0xa8
    ]
    crc = 0
    for byte in data:
        crc = crc_table[(crc ^ byte) & 0xFF]
    return crc

def main():
    args = parse_args()
    print("=" * 60)
    print(" Jetson GPIO LiDAR Diagnostic Tool")
    print(f" Port: {args.port} | Baud: {args.baud}")
    print("=" * 60)

    try:
        ser = serial.Serial(args.port, baudrate=args.baud, timeout=args.timeout)
    except serial.SerialException as e:
        print(f"\n❌ Failed to open serial port {args.port}: {e}")
        print("\nTroubleshooting tips:")
        print(" 1. Check permissions: sudo chmod 666 /dev/ttyTHS1")
        print(" 2. Make sure nvgetty is stopped: sudo systemctl stop nvgetty")
        print(" 3. Verify user is in dialout group: sudo usermod -a -G dialout $USER")
        sys.exit(1)

    print(f"\n Connected to {args.port}. Listening for packet header (0x54 0x2C)...")
    
    packet_count = 0
    valid_packets = 0
    start_time = time.time()
    last_print = time.time()

    # Buffer for stream alignment
    buf = bytearray()
    PACKET_LEN = 47 # Standard LDLiDAR packet length

    try:
        while True:
            raw = ser.read(128)
            if not raw:
                print(f"⚠️  No data received on {args.port} for {args.timeout}s. Check wiring:")
                print("   • Pin 4  (5V)  -> LiDAR VCC / 5V")
                print("   • Pin 6  (GND) -> LiDAR GND")
                print("   • Pin 10 (RXD) -> LiDAR TX (UART Out)")
                print("   • Pin 8  (TXD) -> LiDAR RX (Optional/PWM)")
                time.sleep(1)
                continue

            buf.extend(raw)

            while len(buf) >= PACKET_LEN:
                # Find packet start: 0x54 (Header) and 0x2C (VerLen = 44 bytes payload)
                if buf[0] != 0x54 or buf[1] != 0x2C:
                    buf.pop(0)
                    continue

                packet = bytes(buf[:PACKET_LEN])
                del buf[:PACKET_LEN]
                packet_count += 1

                # Verify CRC
                calc_crc = crc8(packet[:-1])
                pkt_crc = packet[-1]

                if calc_crc == pkt_crc:
                    valid_packets += 1
                    # Parse radar speed (2 bytes, deg/s)
                    radar_speed = struct.unpack("<H", packet[2:4])[0]
                    # Start angle (2 bytes, 0.01 deg)
                    start_angle = struct.unpack("<H", packet[4:6])[0] / 100.0
                    # End angle (2 bytes, 0.01 deg)
                    end_angle = struct.unpack("<H", packet[42:44])[0] / 100.0
                    # Timestamp (2 bytes, ms)
                    timestamp = struct.unpack("<H", packet[44:46])[0]

                    # Parse points (12 measurement points: 2 bytes distance + 1 byte intensity)
                    distances = []
                    for i in range(12):
                        offset = 6 + i * 3
                        dist_mm = struct.unpack("<H", packet[offset:offset+2])[0]
                        intensity = packet[offset+2]
                        if dist_mm > 0:
                            distances.append(dist_mm)

                    now = time.time()
                    if now - last_print >= 0.5:
                        hz = valid_packets / (now - start_time) if (now - start_time) > 0 else 0
                        rpm = (radar_speed / 360.0) * 60.0
                        avg_dist = sum(distances) / len(distances) if distances else 0
                        min_dist = min(distances) if distances else 0
                        max_dist = max(distances) if distances else 0

                        sys.stdout.write(
                            f"\r[OK] Speed: {radar_speed:4d}°/s ({rpm:4.1f} RPM) | "
                            f"Angle: {start_angle:5.1f}° -> {end_angle:5.1f}° | "
                            f"Valid Pkts: {valid_packets} ({hz:4.1f} Hz) | "
                            f"Dist: min={min_dist:4d}mm, avg={avg_dist:4.0f}mm, max={max_dist:4d}mm   "
                        )
                        sys.stdout.flush()
                        last_print = now
                else:
                    # CRC error
                    pass

    except KeyboardInterrupt:
        print("\n\nTest stopped by user.")
        ser.close()

if __name__ == "__main__":
    main()
