#!/usr/bin/env python3
"""
exploration_controller.py — Smooth reactive exploration for slam_toolbox
=========================================================================
Drives the robot through the environment so slam_toolbox can build a map.
slam_toolbox only needs the robot to MOVE; it handles mapping on its own.

Behaviour (NO states, NO stop-and-go)
--------------------------------------
Every 100 ms the controller:
  1. Reads the full 360° LiDAR scan.
  2. Finds the most open direction (largest clearance).
  3. Steers toward that direction with smooth proportional control.
  4. Scales forward speed by how much clearance is ahead.
  5. Emergency-stops ONLY if an obstacle is within 10 cm.

The result is smooth, continuous motion that flows around obstacles —
exactly like the slam_toolbox demo videos.

Why no backup/turn states?
--------------------------
The old controller did: FORWARD → stop → BACKING → TURNING → FORWARD.
This wasted time and produced zero exploration progress because the
clearance thresholds were too close together (0.35 m trigger vs 0.39 m
clear), causing immediate re-triggering.

A reactive controller never stops unless something is dangerously close.
It just steers toward open space while maintaining forward speed.

Subscribes
----------
  /scan              sensor_msgs/LaserScan   STL-27L 360° LiDAR

Publishes
---------
  /cmd_vel           geometry_msgs/Twist     (linear.x + angular.z ONLY)
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


# Number of angular sectors to divide the scan into
N_SECTORS = 24   # 360° / 24 = 15° per sector


class ExplorationController(Node):
    def __init__(self):
        super().__init__('exploration_controller')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('move_speed',           0.20)
        self.declare_parameter('turn_speed',           0.55)
        self.declare_parameter('obstacle_distance',    0.40)
        self.declare_parameter('emergency_stop_dist',  0.10)
        self.declare_parameter('sensor_timeout',       8.0)

        self._move_spd   = self.get_parameter('move_speed').value
        self._turn_spd   = self.get_parameter('turn_speed').value
        self._obs_dist   = self.get_parameter('obstacle_distance').value
        self._estop_dist = self.get_parameter('emergency_stop_dist').value
        self._timeout    = self.get_parameter('sensor_timeout').value

        # ── State ─────────────────────────────────────────────────────────────
        self._scan_received = False
        self._start_t = self.get_clock().now().nanoseconds * 1e-9
        self._last_scan_t = None
        self._log_tick = 0

        # ── Publisher ─────────────────────────────────────────────────────────
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── Subscriber ────────────────────────────────────────────────────────
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(LaserScan, '/scan', self._on_scan, scan_qos)

        # ── 10 Hz control loop ────────────────────────────────────────────────
        self.create_timer(0.1, self._control_loop)

        self.get_logger().info(
            f'Exploration controller (smooth reactive) ready | '
            f'speed={self._move_spd} m/s | turn={self._turn_spd} rad/s | '
            f'obstacle={self._obs_dist} m | estop={self._estop_dist} m')

    # ── LiDAR callback ────────────────────────────────────────────────────────

    def _on_scan(self, msg: LaserScan):
        self._latest_scan = msg
        if not self._scan_received:
            self.get_logger().info(
                f'First LiDAR scan: {len(msg.ranges)} rays')
            self._scan_received = True
        self._last_scan_t = self.get_clock().now().nanoseconds * 1e-9

    # ── Main control loop ─────────────────────────────────────────────────────

    def _control_loop(self):
        now = self.get_clock().now().nanoseconds * 1e-9

        # ── Wait for LiDAR ────────────────────────────────────────────────
        if not self._scan_received:
            self._cmd_pub.publish(Twist())
            self._log_tick += 1
            if self._log_tick % 10 == 0:
                elapsed = now - self._start_t
                if elapsed > self._timeout:
                    self.get_logger().warn(
                        f'NO LIDAR after {elapsed:.0f} s — robot halted')
                else:
                    self.get_logger().info(
                        f'Waiting for LiDAR ({elapsed:.1f}/{self._timeout:.0f} s)')
            return

        # ── Check for stale scan ──────────────────────────────────────────
        if self._last_scan_t and (now - self._last_scan_t) > 2.0:
            self._cmd_pub.publish(Twist())
            if self._log_tick % 10 == 0:
                self.get_logger().warn('LiDAR stale — robot halted')
            self._log_tick += 1
            return

        scan = self._latest_scan

        # ── Build sector clearance map ────────────────────────────────────
        # Divide 360° into N_SECTORS bins, compute min range per sector
        sector_min = [float('inf')] * N_SECTORS
        sector_angle = 2.0 * math.pi / N_SECTORS  # radians per sector

        closest_any = float('inf')

        for i, r in enumerate(scan.ranges):
            if not (scan.range_min <= r <= scan.range_max):
                continue
            if math.isnan(r) or math.isinf(r):
                continue

            angle = scan.angle_min + i * scan.angle_increment
            # Normalize to [0, 2π)
            angle = angle % (2.0 * math.pi)

            sector_idx = int(angle / sector_angle) % N_SECTORS
            if r < sector_min[sector_idx]:
                sector_min[sector_idx] = r

            if r < closest_any:
                closest_any = r

        # ── Emergency stop ────────────────────────────────────────────────
        if closest_any < self._estop_dist:
            self._cmd_pub.publish(Twist())
            self._log_tick += 1
            if self._log_tick % 10 == 0:
                self.get_logger().warn(
                    f'EMERGENCY STOP — obstacle at {closest_any:.2f} m')
            return

        # ── Find the best direction to go ─────────────────────────────────
        # For each sector, the "value" is a blend of:
        #   - clearance in that direction
        #   - forward bias (prefer going straight when possible)
        #   - continuity bonus (prefer sectors with open neighbors)

        # Sector 0 = angle 0 (forward), sector N/4 = 90° left, etc.
        forward_sector = 0

        scores = [0.0] * N_SECTORS
        for s in range(N_SECTORS):
            # Base = clearance (clamped to avoid inf domination)
            clearance = min(sector_min[s], 3.0)

            # Neighbor average (smooth over adjacent sectors)
            prev_c = min(sector_min[(s - 1) % N_SECTORS], 3.0)
            next_c = min(sector_min[(s + 1) % N_SECTORS], 3.0)
            smooth_clearance = (clearance + prev_c + next_c) / 3.0

            # Forward preference: boost sectors near forward
            # Sector angle relative to forward, normalized to [-π, π]
            sector_center = s * sector_angle
            # How far from forward (0) this sector is
            angle_from_fwd = abs(math.atan2(
                math.sin(sector_center),
                math.cos(sector_center)))
            # Bias: 1.0 when forward, 0.0 when backward
            fwd_bias = 1.0 - (angle_from_fwd / math.pi)

            # Combined score
            scores[s] = smooth_clearance + fwd_bias * 0.5

            # Penalize blocked sectors
            if clearance < self._obs_dist:
                scores[s] *= 0.1

        # Best sector
        best_sector = max(range(N_SECTORS), key=lambda s: scores[s])
        best_angle = best_sector * sector_angle
        # Normalize to [-π, π] (0 = forward)
        best_angle = math.atan2(math.sin(best_angle), math.cos(best_angle))

        # ── Compute forward speed ─────────────────────────────────────────
        # front clearance = min of the 3 sectors around forward (±22.5°)
        front_sectors = [0, 1, N_SECTORS - 1]  # forward ± 1 sector
        front_clear = min(sector_min[s] for s in front_sectors)

        if front_clear < self._estop_dist:
            fwd = 0.0
        elif front_clear < self._obs_dist:
            # Proportional slowdown: 0 at estop, full speed at obs_dist
            ratio = (front_clear - self._estop_dist) / (self._obs_dist - self._estop_dist)
            fwd = self._move_spd * ratio * 0.5   # half-speed in tight spaces
        else:
            fwd = self._move_spd

        # If best direction is significantly to the side, slow down more
        # to let the robot turn before driving into something
        turn_urgency = abs(best_angle) / math.pi  # 0 = forward, 1 = backward
        if turn_urgency > 0.4:
            fwd *= max(0.0, 1.0 - turn_urgency)

        # ── Compute turn rate ─────────────────────────────────────────────
        # Proportional: turn faster when the goal direction is more lateral
        turn = self._turn_spd * (best_angle / math.pi)  # smooth, proportional

        # If front is blocked, turn more aggressively toward the open side
        if front_clear < self._obs_dist:
            if abs(best_angle) > 0.1:
                turn = self._turn_spd * (1.0 if best_angle > 0 else -1.0) * 0.8
            else:
                # Front is blocked but best is also forward — find ANY open side
                left_clear  = min(sector_min[s] for s in range(N_SECTORS // 6, N_SECTORS // 3))
                right_clear = min(sector_min[s] for s in range(2 * N_SECTORS // 3, 5 * N_SECTORS // 6))
                if left_clear > right_clear:
                    turn = self._turn_spd * 0.8
                else:
                    turn = -self._turn_spd * 0.8

        # ── Publish ───────────────────────────────────────────────────────
        twist = Twist()
        twist.linear.x = fwd
        twist.angular.z = turn
        self._cmd_pub.publish(twist)

        # ── Status log every 2 s ─────────────────────────────────────────
        self._log_tick += 1
        if self._log_tick % 20 == 0:
            best_deg = math.degrees(best_angle)
            self.get_logger().info(
                f'front={front_clear:.2f} m | '
                f'best_dir={best_deg:+.0f}° | '
                f'cmd=({fwd:.2f} fwd, {turn:+.2f} turn) | '
                f'closest={closest_any:.2f} m')


def main(args=None):
    rclpy.init(args=args)
    node = ExplorationController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
