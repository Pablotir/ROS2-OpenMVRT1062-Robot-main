#!/usr/bin/env python3
"""
motor_driver_node.py
====================
Translates geometry_msgs/Twist (from Nav2 / teleop) into Arduino B-commands,
implementing velocity control on top of the Arduino's position control loop.

Strategy
--------
The Arduino runs RUN_TO_POSITION mode.  We approximate velocity control by
sending an incremental position command every `control_period` seconds:

    degrees_to_move = wheel_angular_velocity [rad/s]
                      × control_period [s]
                      × (180 / π)

Subscribes
----------
/cmd_vel        geometry_msgs/Twist     velocity target from Nav2 / teleop

Publishes
---------
/arduino_cmd    std_msgs/String         B-commands forwarded to arduino_bridge

Mecanum inverse kinematics  (v in rad/s, positive = physical forward)
-----------------------------------------------------------------------
    v_FL = (vx - vy - (Lx + Ly) × ω) / R      ← negate for B motor 3
    v_FR = (vx + vy + (Lx + Ly) × ω) / R      ← negate for B motor 4
    v_BL = (vx + vy - (Lx + Ly) × ω) / R
    v_BR = (vx - vy + (Lx + Ly) × ω) / R

Arduino B-command sign mapping
-------------------------------
    Motor 1 (BL): degrees = +v_BL × dt × (180/π)
    Motor 2 (BR): degrees = +v_BR × dt × (180/π)
    Motor 3 (FL): degrees = -v_FL × dt × (180/π)   ← FL internally inverted
    Motor 4 (FR): degrees = -v_FR × dt × (180/π)   ← FR internally inverted
"""

import math
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class MotorDriverNode(Node):
    def __init__(self):
        super().__init__('motor_driver')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('wheel_radius',     0.0508)   # m
        self.declare_parameter('half_wheelbase',   0.1270)   # Lx (m)
        self.declare_parameter('half_track_width', 0.2172)   # Ly (m)
        self.declare_parameter('control_hz',       10.0)     # cmd frequency
        self.declare_parameter('cmd_vel_timeout',  0.5)      # s – stale guard
        self.declare_parameter('max_speed_mps',    0.5)      # m/s velocity cap
        self.declare_parameter('max_deg_per_step', 720.0)    # safety deg cap

        R    = self.get_parameter('wheel_radius').value
        Lx   = self.get_parameter('half_wheelbase').value
        Ly   = self.get_parameter('half_track_width').value
        hz   = self.get_parameter('control_hz').value
        self._cmd_timeout  = self.get_parameter('cmd_vel_timeout').value
        self._max_spd      = self.get_parameter('max_speed_mps').value
        self._max_deg      = self.get_parameter('max_deg_per_step').value

        self._R    = R
        self._LxLy = Lx + Ly
        self._dt   = 1.0 / hz

        # ── State ─────────────────────────────────────────────────────────────
        self._vx    = 0.0
        self._vy    = 0.0
        self._omega = 0.0
        self._last_cmd_time: float = 0.0
        self._stopped = True    # tracks whether a STOP was already sent

        # ── Pub / Sub ─────────────────────────────────────────────────────────
        self._cmd_pub = self.create_publisher(String, 'arduino_cmd', 10)

        self.create_subscription(
            Twist, 'cmd_vel', self._cmd_vel_callback, 10)

        # ── Control timer ─────────────────────────────────────────────────────
        self.create_timer(self._dt, self._control_loop)

        self.get_logger().info(
            f'motor_driver ready | R={R} m | Lx+Ly={Lx+Ly:.3f} m | '
            f'{hz} Hz control loop')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cmd_vel_callback(self, msg: Twist):
        """Cache the latest velocity command."""
        # Clamp linear velocities
        vx = max(-self._max_spd, min(self._max_spd, msg.linear.x))
        vy = max(-self._max_spd, min(self._max_spd, msg.linear.y))
        # Clamp angular: max ~1 rad/s
        omega = max(-1.5, min(1.5, msg.angular.z))

        self._vx    = vx
        self._vy    = vy
        self._omega = omega
        self._last_cmd_time = time.time()

    def _control_loop(self):
        """
        Called at `control_hz` Hz.
        Converts cached cmd_vel to an Arduino B-command and publishes it.
        """
        now = time.time()
        stale = (now - self._last_cmd_time) > self._cmd_timeout

        # If command is stale or zero, send one STOP and do nothing more
        if stale or (self._vx == 0.0 and self._vy == 0.0 and self._omega == 0.0):
            if not self._stopped:
                self._publish_raw('STOP')
                self._stopped = True
            return

        self._stopped = False

        # ── Mecanum inverse kinematics ────────────────────────────────────────
        # Wheel angular velocities [rad/s] (positive = physical forward)
        v_FL = (self._vx - self._vy - self._LxLy * self._omega) / self._R
        v_FR = (self._vx + self._vy + self._LxLy * self._omega) / self._R
        v_BL = (self._vx + self._vy - self._LxLy * self._omega) / self._R
        v_BR = (self._vx - self._vy + self._LxLy * self._omega) / self._R

        # ── Convert to degree increments for this control step ────────────────
        scale = self._dt * (180.0 / math.pi)

        deg_BL = +v_BL * scale   # motor 1 – direct
        deg_BR = +v_BR * scale   # motor 2 – direct
        deg_FL = -v_FL * scale   # motor 3 – Arduino FL is inverted
        deg_FR = -v_FR * scale   # motor 4 – Arduino FR is inverted

        # Safety cap on per-step movement
        max_d = self._max_deg
        deg_BL = max(-max_d, min(max_d, deg_BL))
        deg_BR = max(-max_d, min(max_d, deg_BR))
        deg_FL = max(-max_d, min(max_d, deg_FL))
        deg_FR = max(-max_d, min(max_d, deg_FR))

        # Filter very small values (dead zone < 0.5°)
        def _fmt(v: float) -> str:
            if abs(v) < 0.5:
                return '0'
            return f'{v:.1f}'

        cmd = (
            f'B 1 {_fmt(deg_BL)} '
            f'2 {_fmt(deg_BR)} '
            f'3 {_fmt(deg_FL)} '
            f'4 {_fmt(deg_FR)}'
        )
        self._publish_raw(cmd)

    # ── Helper ────────────────────────────────────────────────────────────────

    def _publish_raw(self, cmd: str):
        msg = String()
        msg.data = cmd
        self._cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MotorDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Make sure the robot stops on shutdown
        stop_msg = String()
        stop_msg.data = 'STOP'
        node._cmd_pub.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
