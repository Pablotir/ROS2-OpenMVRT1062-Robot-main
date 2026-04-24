#!/usr/bin/env python3
"""
roboclaw_node.py
================
Unified ROS2 node for dual RoboClaw 2x15A motor controllers on a mecanum robot.

Replaces both arduino_bridge_node.py and motor_driver_node.py.

Hardware layout
---------------
Left  RoboClaw (/dev/roboclaw_left,  address 0x80):
    M1 = Rear-Left   (RL)
    M2 = Front-Left  (FL)

Right RoboClaw (/dev/roboclaw_right, address 0x80):
    M1 = Rear-Right  (RR)
    M2 = Front-Right (FR)

Subscribes
----------
/cmd_vel    geometry_msgs/Twist     Velocity target from exploration controller

Publishes
---------
/wheel_ticks    std_msgs/Int32MultiArray    [RL, RR, FL, FR] raw encoder counts
                                            (compatible with mecanum_odometry_node)

Data flow
---------
exploration_controller ──► /cmd_vel ──► roboclaw_node ──► RoboClaw (HW PID)
roboclaw_node          ──► /wheel_ticks ──► mecanum_odometry ──► /odom + TF

Mecanum inverse kinematics (wheel angular velocity, rad/s)
----------------------------------------------------------
    v_FL = (vx - vy - (Lx + Ly) × ω) / R
    v_FR = (vx + vy + (Lx + Ly) × ω) / R
    v_RL = (vx + vy - (Lx + Ly) × ω) / R
    v_RR = (vx - vy + (Lx + Ly) × ω) / R

Conversion to QPPS:
    qpps = wheel_rad_per_s × ticks_per_rev / (2π)

RoboClaw handles motor direction inversions internally (configured during
autotune). Positive QPPS = forward for all motors.
"""

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from std_msgs.msg import Int32MultiArray

from jetson_bot_slam.roboclaw_driver import RoboclawDriver


class RoboclawNode(Node):
    def __init__(self):
        super().__init__('roboclaw_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('left_port',         '/dev/roboclaw_left')
        self.declare_parameter('right_port',        '/dev/roboclaw_right')
        self.declare_parameter('baudrate',          115200)
        self.declare_parameter('address',           0x80)

        self.declare_parameter('wheel_radius',      0.0508)    # m (2 in)
        self.declare_parameter('half_wheelbase',    0.1270)    # Lx (m)
        self.declare_parameter('half_track_width',  0.2172)    # Ly (m)
        self.declare_parameter('ticks_per_rev',     1440)
        self.declare_parameter('max_qpps',          2300)      # conservative cap

        self.declare_parameter('control_hz',        20.0)      # motor + encoder loop
        self.declare_parameter('diag_hz',           0.2)       # diagnostics (every 5s)
        self.declare_parameter('cmd_vel_timeout',   0.5)       # s — stale cmd_vel guard
        self.declare_parameter('max_speed_mps',     0.6)       # m/s velocity cap

        left_port  = self.get_parameter('left_port').value
        right_port = self.get_parameter('right_port').value
        baudrate   = self.get_parameter('baudrate').value
        self._addr = self.get_parameter('address').value

        R    = self.get_parameter('wheel_radius').value
        Lx   = self.get_parameter('half_wheelbase').value
        Ly   = self.get_parameter('half_track_width').value
        TPR  = self.get_parameter('ticks_per_rev').value

        self._max_qpps     = self.get_parameter('max_qpps').value
        self._cmd_timeout  = self.get_parameter('cmd_vel_timeout').value
        self._max_spd      = self.get_parameter('max_speed_mps').value
        hz                 = self.get_parameter('control_hz').value
        diag_hz            = self.get_parameter('diag_hz').value

        # ── Derived constants ─────────────────────────────────────────────────
        self._R    = R
        self._LxLy = Lx + Ly
        # rad/s → QPPS conversion factor
        self._rad_to_qpps = TPR / (2.0 * math.pi)

        # ── State ─────────────────────────────────────────────────────────────
        self._vx    = 0.0
        self._vy    = 0.0
        self._omega = 0.0
        self._last_cmd_time = 0.0
        self._stopped = True
        self._connected_left  = False
        self._connected_right = False

        # ── Open serial connections ───────────────────────────────────────────
        self._rc_left  = RoboclawDriver(left_port, baudrate)
        self._rc_right = RoboclawDriver(right_port, baudrate)

        self._connected_left = self._rc_left.open()
        if self._connected_left:
            self.get_logger().info(f'LEFT  RoboClaw opened: {left_port}')
        else:
            self.get_logger().error(f'FAILED to open LEFT RoboClaw: {left_port}')

        self._connected_right = self._rc_right.open()
        if self._connected_right:
            self.get_logger().info(f'RIGHT RoboClaw opened: {right_port}')
        else:
            self.get_logger().error(f'FAILED to open RIGHT RoboClaw: {right_port}')

        # ── Publishers ────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5)

        self._ticks_pub = self.create_publisher(
            Int32MultiArray, 'wheel_ticks', sensor_qos)

        # ── Subscriber ────────────────────────────────────────────────────────
        self.create_subscription(Twist, 'cmd_vel', self._cmd_vel_cb, 10)

        # ── Timers ────────────────────────────────────────────────────────────
        self.create_timer(1.0 / hz, self._control_loop)
        if diag_hz > 0:
            self.create_timer(1.0 / diag_hz, self._diag_loop)

        self.get_logger().info(
            f'roboclaw_node ready | R={R} | Lx+Ly={self._LxLy:.3f} | '
            f'TPR={TPR} | max_qpps={self._max_qpps} | {hz} Hz control')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cmd_vel_cb(self, msg: Twist):
        """Cache the latest velocity command."""
        self._vx    = max(-self._max_spd, min(self._max_spd, msg.linear.x))
        self._vy    = max(-self._max_spd, min(self._max_spd, msg.linear.y))
        self._omega = max(-1.5, min(1.5, msg.angular.z))
        self._last_cmd_time = time.time()

    # ── Control loop (20 Hz) ──────────────────────────────────────────────────

    def _control_loop(self):
        """Read encoders, send motor commands."""
        # ── Read encoders ─────────────────────────────────────────────────
        enc_rl, enc_rr, enc_fl, enc_fr = 0, 0, 0, 0

        if self._connected_left:
            ok1, enc_rl, _ = self._rc_left.read_encoder_m1(self._addr)
            ok2, enc_fl, _ = self._rc_left.read_encoder_m2(self._addr)
            if not ok1 or not ok2:
                self.get_logger().warn('LEFT encoder read failed', throttle_duration_sec=5.0)

        if self._connected_right:
            ok1, enc_rr, _ = self._rc_right.read_encoder_m1(self._addr)
            ok2, enc_fr, _ = self._rc_right.read_encoder_m2(self._addr)
            if not ok1 or not ok2:
                self.get_logger().warn('RIGHT encoder read failed', throttle_duration_sec=5.0)

        # Publish [RL, RR, FL, FR] — matches old [BL, BR, FL, FR] format
        ticks_msg = Int32MultiArray()
        ticks_msg.data = [enc_rl, enc_rr, enc_fl, enc_fr]
        self._ticks_pub.publish(ticks_msg)

        # ── Motor commands ────────────────────────────────────────────────
        now = time.time()
        stale = (now - self._last_cmd_time) > self._cmd_timeout
        zero  = (self._vx == 0.0 and self._vy == 0.0 and self._omega == 0.0)

        if stale or zero:
            if not self._stopped:
                self._stop_all()
                self._stopped = True
            return

        self._stopped = False

        # ── Mecanum inverse kinematics ────────────────────────────────────
        v_fl = (self._vx - self._vy - self._LxLy * self._omega) / self._R
        v_fr = (self._vx + self._vy + self._LxLy * self._omega) / self._R
        v_rl = (self._vx + self._vy - self._LxLy * self._omega) / self._R
        v_rr = (self._vx - self._vy + self._LxLy * self._omega) / self._R

        # Convert rad/s → QPPS
        qpps_fl = int(v_fl * self._rad_to_qpps)
        qpps_fr = int(v_fr * self._rad_to_qpps)
        qpps_rl = int(v_rl * self._rad_to_qpps)
        qpps_rr = int(v_rr * self._rad_to_qpps)

        # Clamp to max QPPS
        mx = self._max_qpps
        qpps_fl = max(-mx, min(mx, qpps_fl))
        qpps_fr = max(-mx, min(mx, qpps_fr))
        qpps_rl = max(-mx, min(mx, qpps_rl))
        qpps_rr = max(-mx, min(mx, qpps_rr))

        # Send to controllers
        # Left:  M1=RL, M2=FL
        # Right: M1=RR, M2=FR
        if self._connected_left:
            if not self._rc_left.speed_m1_m2(self._addr, qpps_rl, qpps_fl):
                self.get_logger().warn('LEFT speed cmd failed', throttle_duration_sec=5.0)

        if self._connected_right:
            if not self._rc_right.speed_m1_m2(self._addr, qpps_rr, qpps_fr):
                self.get_logger().warn('RIGHT speed cmd failed', throttle_duration_sec=5.0)

    def _stop_all(self):
        """Send zero speed to all motors."""
        if self._connected_left:
            self._rc_left.speed_m1_m2(self._addr, 0, 0)
        if self._connected_right:
            self._rc_right.speed_m1_m2(self._addr, 0, 0)

    # ── Diagnostics (every 5s) ────────────────────────────────────────────────

    def _diag_loop(self):
        """Log battery voltage, currents, and temperature."""
        parts = []

        if self._connected_left:
            ok, volts = self._rc_left.read_main_battery(self._addr)
            if ok:
                parts.append(f'L_batt={volts:.1f}V')
            ok, m1a, m2a = self._rc_left.read_currents(self._addr)
            if ok:
                parts.append(f'L_curr=M1:{m1a:.2f}A M2:{m2a:.2f}A')
            ok, temp = self._rc_left.read_temperature(self._addr)
            if ok:
                parts.append(f'L_temp={temp:.1f}°C')

        if self._connected_right:
            ok, volts = self._rc_right.read_main_battery(self._addr)
            if ok:
                parts.append(f'R_batt={volts:.1f}V')
            ok, m1a, m2a = self._rc_right.read_currents(self._addr)
            if ok:
                parts.append(f'R_curr=M1:{m1a:.2f}A M2:{m2a:.2f}A')
            ok, temp = self._rc_right.read_temperature(self._addr)
            if ok:
                parts.append(f'R_temp={temp:.1f}°C')

        if parts:
            self.get_logger().info(f'[DIAG] {" | ".join(parts)}')

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def destroy_node(self):
        self.get_logger().info('Shutting down — stopping all motors')
        self._stop_all()
        self._rc_left.close()
        self._rc_right.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RoboclawNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
