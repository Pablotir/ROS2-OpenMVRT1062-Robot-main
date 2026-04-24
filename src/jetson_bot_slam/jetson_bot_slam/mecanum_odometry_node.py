#!/usr/bin/env python3
"""
mecanum_odometry_node.py
========================
Computes robot pose from mecanum-drive wheel encoder ticks and publishes
nav_msgs/Odometry + broadcasts the odom → base_link TF.

Subscribes
----------
/wheel_ticks    std_msgs/Int32MultiArray   [RL, RR, FL, FR] raw encoder counts
                                           (published by roboclaw_node)

Publishes
---------
/odom           nav_msgs/Odometry
/tf             odom → base_link  (via tf2_ros TransformBroadcaster)

Robot geometry (all in metres)
-------------------------------
Wheel radius  R  = 0.0508  m   (2 in)
Ticks per rev    = 1440
Half wheelbase   Lx = 0.1270 m   (5.0 in, front-to-back axle separation / 2)
Half track width Ly = 0.2172 m   (8.55 in, left-to-right axle separation / 2)
These can be overridden via ROS2 parameters.

Sign conventions
----------------
Dual RoboClaw 2x15A controllers handle motor direction inversions internally
(configured during autotune in BasicMicro Motion Studio).
All encoder counts increase for forward motion — no software inversions needed.

    Physical forward displacement per wheel:
        d_FL =  +Δenc_FL * meters_per_tick
        d_FR =  +Δenc_FR * meters_per_tick
        d_RL =  +Δenc_RL * meters_per_tick
        d_RR =  +Δenc_RR * meters_per_tick

    Robot motion (mecanum forward kinematics):
        Δx     = (d_FL + d_FR + d_RL + d_RR) / 4
        Δy     = (-d_FL + d_FR + d_RL - d_RR) / 4    (+y = left in ROS2)
        Δθ     = (-d_FL + d_FR - d_RL + d_RR) / (4 * (Lx + Ly))
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
from std_msgs.msg import Int32MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster


def euler_to_quaternion(yaw: float) -> Quaternion:
    """Convert a yaw angle (radians) to a geometry_msgs/Quaternion."""
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class MecanumOdometryNode(Node):
    def __init__(self):
        super().__init__('mecanum_odometry')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('wheel_radius',       0.0508)   # m (4 in diameter)
        self.declare_parameter('ticks_per_rev',      1440)
        self.declare_parameter('half_wheelbase',     0.1270)   # Lx  (m)
        self.declare_parameter('half_track_width',   0.2172)   # Ly  (m)
        self.declare_parameter('odom_frame',         'odom')
        self.declare_parameter('base_frame',         'base_link')
        self.declare_parameter('publish_tf',         True)
        self.declare_parameter('tf_republish_hz',    20.0)     # TF keep-alive rate

        R   = self.get_parameter('wheel_radius').value
        TPR = self.get_parameter('ticks_per_rev').value
        Lx  = self.get_parameter('half_wheelbase').value
        Ly  = self.get_parameter('half_track_width').value

        self._odom_frame = self.get_parameter('odom_frame').value
        self._base_frame = self.get_parameter('base_frame').value
        self._pub_tf     = self.get_parameter('publish_tf').value
        tf_hz            = self.get_parameter('tf_republish_hz').value

        # meters of arc per encoder tick
        self._m_per_tick = (2.0 * math.pi * R) / TPR
        self._Lx = Lx
        self._Ly = Ly

        # ── State ─────────────────────────────────────────────────────────────
        # Robot pose in the odom frame
        self._x     = 0.0
        self._y     = 0.0
        self._theta = 0.0

        # Latest velocities (updated by encoder callback, re-used by keep-alive)
        self._vx    = 0.0
        self._vy    = 0.0
        self._omega = 0.0

        # Previous encoder counts (initialised on first message)
        self._prev_enc: list[int] | None = None   # [RL, RR, FL, FR]
        self._prev_time = None
        self._got_first_data = False

        # ── QoS ───────────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ── Pub / Sub ─────────────────────────────────────────────────────────
        self._odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self._tf_br    = TransformBroadcaster(self)

        self.create_subscription(
            Int32MultiArray, 'wheel_ticks', self._tick_callback, sensor_qos)

        # ── 20 Hz TF keep-alive timer ─────────────────────────────────────────
        # Re-publishes odom TF + odom msg at 20 Hz with fresh timestamps,
        # filling gaps between Arduino DATA messages (~1-3 Hz).
        # This prevents the TF starvation cascade that blocks Nav2 + RTAB-Map.
        if tf_hz > 0:
            self.create_timer(1.0 / tf_hz, self._tf_keepalive)
            self.get_logger().info(f'TF keep-alive enabled at {tf_hz:.0f} Hz')

        self.get_logger().info(
            f'mecanum_odometry ready | R={R} m | Lx={Lx} m | Ly={Ly} m | '
            f'{TPR} ticks/rev | {self._m_per_tick*1e3:.4f} mm/tick')

    # ── Callback ──────────────────────────────────────────────────────────────

    def _tick_callback(self, msg: Int32MultiArray):
        """Process a new encoder reading and update pose estimate."""
        if len(msg.data) < 4:
            self.get_logger().warn('Expected 4 encoder values, got fewer')
            return

        enc = list(msg.data)   # [RL, RR, FL, FR]
        now = self.get_clock().now()

        # ── First message: initialise and return ──────────────────────────────
        if self._prev_enc is None:
            self._prev_enc  = enc
            self._prev_time = now
            self._got_first_data = True
            return

        dt = (now - self._prev_time).nanoseconds * 1e-9
        if dt <= 0.0:
            return

        # ── Encoder deltas ────────────────────────────────────────────────────
        d_enc_RL = enc[0] - self._prev_enc[0]
        d_enc_RR = enc[1] - self._prev_enc[1]
        d_enc_FL = enc[2] - self._prev_enc[2]
        d_enc_FR = enc[3] - self._prev_enc[3]

        self._prev_enc  = enc
        self._prev_time = now

        # ── Physical displacements (metres) ───────────────────────────────────
        # RoboClaw handles motor direction inversions internally.
        # All encoders count up for forward motion — no software inversion.
        d_RL = +d_enc_RL * self._m_per_tick
        d_RR = +d_enc_RR * self._m_per_tick
        d_FL = +d_enc_FL * self._m_per_tick
        d_FR = +d_enc_FR * self._m_per_tick

        # ── Mecanum forward kinematics ────────────────────────────────────────
        dx_body = (d_FL + d_FR + d_RL + d_RR) / 4.0          # forward
        dy_body = (-d_FL + d_FR + d_RL - d_RR) / 4.0         # lateral (+left)
        dtheta  = (-d_FL + d_FR - d_RL + d_RR) / (4.0 * (self._Lx + self._Ly))

        # ── Integrate into odom frame (mid-point integration) ─────────────────
        mid_theta = self._theta + dtheta / 2.0
        self._x     += dx_body * math.cos(mid_theta) - dy_body * math.sin(mid_theta)
        self._y     += dx_body * math.sin(mid_theta) + dy_body * math.cos(mid_theta)
        self._theta += dtheta

        # Normalise heading to [-π, π]
        self._theta = math.atan2(math.sin(self._theta), math.cos(self._theta))

        # ── Cache velocities (re-used by keep-alive timer) ────────────────────
        self._vx    = dx_body / dt
        self._vy    = dy_body / dt
        self._omega = dtheta  / dt

        # ── Publish odom + TF ─────────────────────────────────────────────────
        self._publish_odom(now)

    # ── Shared odom + TF publisher ────────────────────────────────────────────

    def _publish_odom(self, stamp):
        """Publish Odometry message and broadcast odom→base_link TF."""
        odom = Odometry()
        odom.header.stamp    = stamp.to_msg()
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id  = self._base_frame

        odom.pose.pose.position.x  = self._x
        odom.pose.pose.position.y  = self._y
        odom.pose.pose.position.z  = 0.0
        odom.pose.pose.orientation = euler_to_quaternion(self._theta)

        # Diagonal position covariance (x, y, z, roll, pitch, yaw)
        pos_cov = [0.0] * 36
        pos_cov[0]  = 0.001   # x–x
        pos_cov[7]  = 0.002   # y–y  (higher: lateral slip uncertainty)
        pos_cov[14] = 1e6     # z–z  (unknown – planar robot)
        pos_cov[21] = 1e6     # roll  (not observed)
        pos_cov[28] = 1e6     # pitch (not observed)
        pos_cov[35] = 0.005   # yaw–yaw
        odom.pose.covariance = pos_cov

        vel_cov = [0.0] * 36
        vel_cov[0]  = 0.001
        vel_cov[7]  = 0.002
        vel_cov[14] = 1e6
        vel_cov[21] = 1e6
        vel_cov[28] = 1e6
        vel_cov[35] = 0.005
        odom.twist.twist.linear.x  = self._vx
        odom.twist.twist.linear.y  = self._vy
        odom.twist.twist.angular.z = self._omega
        odom.twist.covariance = vel_cov

        self._odom_pub.publish(odom)

        # ── Broadcast TF odom → base_link ────────────────────────────────────
        if self._pub_tf:
            tf = TransformStamped()
            tf.header.stamp            = stamp.to_msg()
            tf.header.frame_id         = self._odom_frame
            tf.child_frame_id          = self._base_frame
            tf.transform.translation.x = self._x
            tf.transform.translation.y = self._y
            tf.transform.translation.z = 0.0
            tf.transform.rotation      = euler_to_quaternion(self._theta)
            self._tf_br.sendTransform(tf)

    # ── TF keep-alive ─────────────────────────────────────────────────────────

    def _tf_keepalive(self):
        """Re-broadcast odom→base_link TF at 20 Hz with fresh timestamps.

        The RoboClaw encoder reads happen at ~20 Hz, but this timer acts
        as a safety net to keep TF fresh if an encoder read is delayed.
        Nav2 and SLAM Toolbox need continuous TF availability.
        """
        if not self._got_first_data:
            return   # Don't publish until we have at least one encoder reading
        self._publish_odom(self.get_clock().now())


def main(args=None):
    rclpy.init(args=args)
    node = MecanumOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
