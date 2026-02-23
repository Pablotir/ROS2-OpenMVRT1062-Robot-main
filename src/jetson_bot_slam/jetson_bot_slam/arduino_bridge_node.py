#!/usr/bin/env python3
"""
arduino_bridge_node.py
======================
Bridges the Jetson to the Arduino motor controller over serial.

Publishes
---------
/wheel_ticks        std_msgs/Int32MultiArray  [BL, BR, FL, FR] raw encoder counts
/ultrasonic_range   sensor_msgs/Range         HC-SR04 reading (metres)

Subscribes
----------
/arduino_cmd        std_msgs/String           Raw command string forwarded to Arduino
                                              (used internally by motor_driver_node)

The Arduino streams "DATA pos1 pos2 pos3 pos4 dist_m" at ~3 Hz.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import serial
import threading

from std_msgs.msg import Int32MultiArray, String
from sensor_msgs.msg import Range


class ArduinoBridgeNode(Node):
    def __init__(self):
        super().__init__('arduino_bridge')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('ultrasonic_fov', 0.2618)   # ~15° half-angle [rad]
        self.declare_parameter('ultrasonic_min', 0.02)     # 2 cm
        self.declare_parameter('ultrasonic_max', 4.00)     # 4 m

        port       = self.get_parameter('serial_port').value
        baud       = self.get_parameter('baud_rate').value
        self._fov  = self.get_parameter('ultrasonic_fov').value
        self._dmin = self.get_parameter('ultrasonic_min').value
        self._dmax = self.get_parameter('ultrasonic_max').value

        # ── QoS ───────────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self._ticks_pub = self.create_publisher(
            Int32MultiArray, 'wheel_ticks', sensor_qos)

        self._range_pub = self.create_publisher(
            Range, 'ultrasonic_range', sensor_qos)

        self._raw_pub = self.create_publisher(
            String, 'arduino_raw', 10)

        # ── Subscriber ────────────────────────────────────────────────────────
        self.create_subscription(
            String, 'arduino_cmd', self._cmd_callback, 10)

        # ── Serial ────────────────────────────────────────────────────────────
        self._ser = None
        self._serial_lock = threading.Lock()
        self._connect(port, baud)

        # Serial reader runs in its own thread so it never blocks the ROS spin
        self._running = True
        self._reader_thread = threading.Thread(
            target=self._read_loop, daemon=True)
        self._reader_thread.start()

        self.get_logger().info(
            f'arduino_bridge ready on {port} @ {baud} baud')

    # ── Serial helpers ────────────────────────────────────────────────────────

    def _connect(self, port: str, baud: int):
        try:
            self._ser = serial.Serial(port, baud, timeout=0.1)
            import time; time.sleep(2)          # let Arduino reset
            self.get_logger().info(f'Serial connected: {port}')
        except serial.SerialException as e:
            self.get_logger().error(f'Cannot open {port}: {e}')
            self._ser = None

    def _read_loop(self):
        """Background thread: read lines, parse DATA messages, publish."""
        while self._running:
            if self._ser is None or not self._ser.is_open:
                import time; time.sleep(1)
                continue
            try:
                with self._serial_lock:
                    line = self._ser.readline()
            except serial.SerialException as e:
                self.get_logger().warn(f'Serial read error: {e}')
                import time; time.sleep(0.5)
                continue

            if not line:
                continue

            try:
                text = line.decode('utf-8', errors='ignore').strip()
            except Exception:
                continue

            if not text:
                continue

            # Publish raw line for debugging
            raw = String()
            raw.data = text
            self._raw_pub.publish(raw)

            if text.startswith('DATA '):
                self._parse_data_line(text)

    def _parse_data_line(self, text: str):
        """
        Expected format:
            DATA <pos1> <pos2> <pos3> <pos4> <distance_m>
        Motor mapping: pos1=BL, pos2=BR, pos3=FL, pos4=FR
        """
        parts = text.split()
        if len(parts) < 6:
            return
        try:
            pos1 = int(parts[1])    # BL
            pos2 = int(parts[2])    # BR
            pos3 = int(parts[3])    # FL
            pos4 = int(parts[4])    # FR
            dist = float(parts[5])  # metres
        except ValueError:
            return

        # --- Wheel ticks [BL, BR, FL, FR] ---
        ticks_msg = Int32MultiArray()
        ticks_msg.data = [pos1, pos2, pos3, pos4]
        self._ticks_pub.publish(ticks_msg)

        # --- Ultrasonic range ---
        rng = Range()
        rng.header.stamp    = self.get_clock().now().to_msg()
        rng.header.frame_id = 'ultrasonic_link'
        rng.radiation_type  = Range.ULTRASOUND
        rng.field_of_view   = self._fov
        rng.min_range       = self._dmin
        rng.max_range       = self._dmax
        rng.range           = dist
        self._range_pub.publish(rng)

    # ── Command subscriber ────────────────────────────────────────────────────

    def _cmd_callback(self, msg: String):
        """Forward any command string directly to the Arduino."""
        cmd = msg.data.strip()
        if not cmd:
            return
        if self._ser is None or not self._ser.is_open:
            self.get_logger().warn('Serial not open – dropping command')
            return
        with self._serial_lock:
            self._ser.write(f'{cmd}\n'.encode('utf-8'))
            self._ser.flush()

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def destroy_node(self):
        self._running = False
        if self._ser and self._ser.is_open:
            # Ensure the robot stops
            try:
                with self._serial_lock:
                    self._ser.write(b'STOP\n')
                    self._ser.flush()
            except Exception:
                pass
            self._ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArduinoBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
