#!/usr/bin/env python3
"""
exploration_controller.py  — LiDAR + ultrasonic exploration
=============================================================
Drives the robot autonomously using the STL-27L 360° LiDAR as the PRIMARY
obstacle sensor and the rear-mounted HC-SR04 ultrasonic as a BACKUP for
reverse maneuvers.

State machine
-------------
  FORWARD  → drive forward at move_speed
           → if front_min < obstacle_distance → BACKING

  BACKING  → reverse for backup_s seconds to create clearance
           → if rear too close (LiDAR rear zone OR ultrasonic) → emergency stop reverse
           → then → TURNING

  TURNING  → spin toward the zone with the most open space
           → until BOTH: min_turn_s elapsed AND front_min >= clear_distance
           → then → FORWARD

Exploration strategy
--------------------
  When an obstacle is detected in front, the robot picks the turn direction
  based on which side (left vs right LiDAR zone) has more open space.
  This replaces the old fixed left-hand rule and produces better coverage.

  If still blocked after max_turn_s, switches direction (corner escape),
  then forces forward after a second timeout.

360° LiDAR zones
-----------------
  FRONT:  ±60°  (0° = dead ahead)
  LEFT:   60° to 120°
  RIGHT:  -60° to -120° (i.e. 240° to 300°)
  REAR:   120° to 240°

Subscribes
----------
  /scan              sensor_msgs/LaserScan   STL-27L 360° LiDAR (PRIMARY)
  /ultrasonic_range  sensor_msgs/Range        rear HC-SR04 (BACKUP, best-effort)

Publishes
---------
  /cmd_vel                    geometry_msgs/Twist
  /robot/movement_complete    std_msgs/Bool

Parameters
----------
move_speed          float   0.20   m/s forward speed
turn_speed          float   0.55   rad/s turn speed
obstacle_distance   float   0.30   m — start backing/turning below this (front zone)
emergency_stop_dist float   0.08   m — enter emergency reverse below this (any zone)
rear_safety_dist    float   0.15   m — stop reversing if rear obstacle (LiDAR or ultrasonic)
backup_s            float   2.0    s — reverse duration before turning
min_turn_s          float   3.0    s — minimum turn time before checking clear
max_turn_s          float   10.0   s — switch direction if still blocked (corner escape)
label_every         int     5      turns between AI label requests
sensor_timeout      float   8.0    s — halt if no LiDAR data for this long
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Range

FORWARD        = 'forward'
BACKING        = 'backing'
EMERGENCY_BACK = 'emergency_back'
TURNING        = 'turning'


class ExplorationController(Node):
    def __init__(self):
        super().__init__('exploration_controller')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('move_speed',           0.20)
        self.declare_parameter('turn_speed',           0.55)
        self.declare_parameter('obstacle_distance',    0.30)
        self.declare_parameter('emergency_stop_dist',  0.08)
        self.declare_parameter('rear_safety_dist',     0.15)
        self.declare_parameter('backup_s',             2.0)
        self.declare_parameter('min_turn_s',           3.0)
        self.declare_parameter('max_turn_s',           10.0)
        self.declare_parameter('label_every',          5)
        self.declare_parameter('sensor_timeout',       8.0)

        self._move_spd      = self.get_parameter('move_speed').value
        self._turn_spd      = self.get_parameter('turn_speed').value
        self._obs_dist      = self.get_parameter('obstacle_distance').value
        self._estop_dist    = self.get_parameter('emergency_stop_dist').value
        self._rear_safe     = self.get_parameter('rear_safety_dist').value
        self._backup_s      = self.get_parameter('backup_s').value
        self._min_turn_s    = self.get_parameter('min_turn_s').value
        self._max_turn_s    = self.get_parameter('max_turn_s').value
        self._label_every   = int(self.get_parameter('label_every').value)
        self._sensor_timeout = self.get_parameter('sensor_timeout').value

        # clear_dist: how far ahead must be open before we resume FORWARD.
        self._clear_dist = self._obs_dist * 1.1

        # ── Publishers ────────────────────────────────────────────────────
        self._cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self._done_pub = self.create_publisher(Bool,  '/robot/movement_complete', 10)

        # ── Subscribers ───────────────────────────────────────────────────
        # LiDAR — primary sensor (RELIABLE QoS, standard for LaserScan)
        self.create_subscription(LaserScan, '/scan', self._on_scan, 10)

        # Rear ultrasonic — backup (BEST_EFFORT to match arduino_bridge)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(Range, '/ultrasonic_range', self._on_ultrasonic, sensor_qos)

        # ── State ─────────────────────────────────────────────────────────
        self._state            = FORWARD
        self._turn_dir         = 1.0      # +1 left, -1 right (chosen dynamically)
        self._turn_count       = 0
        self._flipped          = False
        self._phase_start_t    = None
        self._current_twist    = Twist()
        self._log_tick         = 0
        self._start_t          = self.get_clock().now().nanoseconds * 1e-9

        # LiDAR zone minimums (updated every /scan callback)
        self._front_min        = 9.9
        self._left_min         = 9.9
        self._right_min        = 9.9
        self._rear_min         = 9.9
        self._scan_received    = False
        self._last_scan_t      = None

        # Rear ultrasonic (backup)
        self._rear_ultra_range = 9.9
        self._ultra_received   = False

        # 10 Hz control loop
        self.create_timer(0.1, self._control_loop)

        self.get_logger().info(
            f'Exploration controller (LiDAR + ultrasonic) ready | '
            f'speed={self._move_spd} m/s | turn={self._turn_spd} rad/s | '
            f'obstacle={self._obs_dist} m | clear={self._clear_dist:.2f} m | '
            f'estop={self._estop_dist} m | rear_safe={self._rear_safe} m | '
            f'backup_s={self._backup_s} s | min_turn={self._min_turn_s} s | '
            f'max_turn={self._max_turn_s} s')

    # ── LiDAR callback ────────────────────────────────────────────────────────

    def _on_scan(self, msg: LaserScan):
        """Process 360° LaserScan into 4 obstacle zones."""
        n = len(msg.ranges)
        if n == 0:
            return

        front_vals = []
        left_vals = []
        right_vals = []
        rear_vals = []

        for i, r in enumerate(msg.ranges):
            if not (msg.range_min <= r <= msg.range_max):
                continue   # skip inf / NaN / out-of-range

            # Angle of this ray (radians, 0 = forward, + = CCW left)
            angle = msg.angle_min + i * msg.angle_increment
            # Normalize to [-pi, pi]
            angle = math.atan2(math.sin(angle), math.cos(angle))
            deg = math.degrees(angle)

            if -60.0 <= deg <= 60.0:
                front_vals.append(r)
            elif 60.0 < deg <= 120.0:
                left_vals.append(r)
            elif -120.0 <= deg < -60.0:
                right_vals.append(r)
            else:
                rear_vals.append(r)

        self._front_min = min(front_vals) if front_vals else 9.9
        self._left_min  = min(left_vals)  if left_vals  else 9.9
        self._right_min = min(right_vals) if right_vals else 9.9
        self._rear_min  = min(rear_vals)  if rear_vals  else 9.9

        if not self._scan_received:
            self.get_logger().info(
                f'First LiDAR scan received: {n} rays | '
                f'front={self._front_min:.2f} left={self._left_min:.2f} '
                f'right={self._right_min:.2f} rear={self._rear_min:.2f}')
        self._scan_received = True
        self._last_scan_t = self.get_clock().now().nanoseconds * 1e-9

    # ── Rear ultrasonic callback ──────────────────────────────────────────────

    def _on_ultrasonic(self, msg: Range):
        """Store rear ultrasonic reading (backup for reverse safety)."""
        if msg.min_range <= msg.range <= msg.max_range:
            self._rear_ultra_range = msg.range
            self._ultra_received = True

    # ── Effective rear distance (min of LiDAR rear zone + ultrasonic) ─────────

    def _rear_distance(self) -> float:
        """Return the closest rear obstacle from either sensor."""
        rear = self._rear_min
        if self._ultra_received:
            rear = min(rear, self._rear_ultra_range)
        return rear

    # ── 10 Hz control loop ────────────────────────────────────────────────────

    def _control_loop(self):
        now = self.get_clock().now().nanoseconds * 1e-9

        # ── Guard: hold still until first LiDAR scan arrives ──────────────
        if not self._scan_received:
            self._cmd_pub.publish(Twist())
            waiting_s = now - self._start_t
            if self._log_tick % 10 == 0:
                if waiting_s > self._sensor_timeout:
                    self.get_logger().warn(
                        f'*** NO LIDAR DATA after {waiting_s:.0f} s — ROBOT HALTED ***\n'
                        f'  Check: (1) STL-27L connected to /dev/ttyUSB1\n'
                        f'         (2) ldlidar_stl_ros2 node is running\n'
                        f'             run: ros2 topic echo /scan --once')
                else:
                    self.get_logger().info(
                        f'[SENSOR] waiting for first LiDAR scan '
                        f'({waiting_s:.1f}/{self._sensor_timeout:.0f} s)')
            self._log_tick += 1
            return

        # ── 1 Hz status line ──────────────────────────────────────────────
        self._log_tick += 1
        if self._log_tick % 10 == 0:
            if self._state == FORWARD:
                action = f'DRIVING FORWARD  @ {self._move_spd} m/s'
            elif self._state == BACKING:
                elapsed = now - self._phase_start_t if self._phase_start_t else 0.0
                action = f'BACKING UP       ({elapsed:.1f}/{self._backup_s:.1f} s)'
            elif self._state == EMERGENCY_BACK:
                action = f'EMERGENCY REVERSE (front={self._front_min:.2f} m)'
            else:  # TURNING
                elapsed = now - self._phase_start_t if self._phase_start_t else 0.0
                dirstr  = 'LEFT' if self._turn_dir > 0 else 'RIGHT'
                action  = f'TURNING {dirstr:<5} ({elapsed:.1f}/{self._max_turn_s:.1f} s max)'

            ultra_str = f'{self._rear_ultra_range:.2f}m' if self._ultra_received else 'n/a'
            self.get_logger().info(
                f'[LIDAR] front={self._front_min:.2f} left={self._left_min:.2f} '
                f'right={self._right_min:.2f} rear={self._rear_min:.2f}  '
                f'[ULTRA rear={ultra_str}]  '
                f'[ACTION] {action}')

        # ── Guard: halt if LiDAR has gone stale ───────────────────────────
        if self._last_scan_t is not None:
            stale_s = now - self._last_scan_t
            if stale_s > 2.0:  # LiDAR at 10Hz, 2s = 20 missed frames
                self._cmd_pub.publish(Twist())
                if self._log_tick % 10 == 0:
                    self.get_logger().warn(
                        f'[LIDAR STALE] No scan for {stale_s:.1f} s — robot halted')
                return

        # ── Emergency: front obstacle extremely close ─────────────────────
        if self._front_min < self._estop_dist and self._state != EMERGENCY_BACK:
            self.get_logger().warn(
                f'EMERGENCY — front={self._front_min:.2f} m — reversing')
            self._state = EMERGENCY_BACK
            self._current_twist = Twist()
            self._current_twist.linear.x = -self._move_spd

        if self._state == EMERGENCY_BACK:
            rear_dist = self._rear_distance()
            if rear_dist < self._rear_safe:
                # Can't reverse either — just stop
                self.get_logger().warn(
                    f'BOXED IN — front={self._front_min:.2f} rear={rear_dist:.2f} — stopped')
                self._cmd_pub.publish(Twist())
                return
            if self._front_min >= self._obs_dist:
                self.get_logger().info(
                    f'Emergency clear at {self._front_min:.2f} m — turning')
                self._start_turn(now)
            else:
                self._current_twist = Twist()
                self._current_twist.linear.x = -self._move_spd
            self._cmd_pub.publish(self._current_twist)
            return

        # ── FORWARD state ─────────────────────────────────────────────────
        if self._state == FORWARD:
            if self._front_min < self._obs_dist:
                self.get_logger().info(
                    f'>>> OBSTACLE at {self._front_min:.2f} m — '
                    f'starting {self._backup_s} s reverse')
                self._state = BACKING
                self._phase_start_t = now
                self._flipped = False
                self._current_twist = Twist()
                self._current_twist.linear.x = -self._move_spd
            else:
                self._current_twist = Twist()
                self._current_twist.linear.x = self._move_spd

        # ── BACKING state ─────────────────────────────────────────────────
        elif self._state == BACKING:
            elapsed = now - self._phase_start_t

            # Check rear safety while reversing
            rear_dist = self._rear_distance()
            if rear_dist < self._rear_safe:
                self.get_logger().warn(
                    f'Rear obstacle at {rear_dist:.2f} m — stopping reverse early')
                self._start_turn(now)
            elif elapsed >= self._backup_s:
                self.get_logger().info(
                    f'>>> BACKUP DONE — picking best turn direction')
                self._start_turn(now)
            else:
                self._current_twist = Twist()
                self._current_twist.linear.x = -self._move_spd

        # ── TURNING state ─────────────────────────────────────────────────
        elif self._state == TURNING:
            elapsed = now - self._phase_start_t
            dirstr = 'LEFT' if self._turn_dir > 0 else 'RIGHT'

            # Corner escape: still blocked after max_turn_s — switch direction
            if elapsed >= self._max_turn_s and not self._flipped:
                self.get_logger().warn(
                    f'>>> CORNER ESCAPE — still blocked after {self._max_turn_s:.1f} s '
                    f'(front={self._front_min:.2f} m) — switching to {"RIGHT" if self._turn_dir > 0 else "LEFT"}')
                self._flipped = True
                self._turn_dir *= -1.0
                self._phase_start_t = now
                self._current_twist = Twist()
                self._current_twist.angular.z = self._turn_dir * self._turn_spd
            elif elapsed >= self._max_turn_s and self._flipped:
                # Double escape timeout — force forward
                self.get_logger().warn(
                    f'>>> FORCED FORWARD — still blocked after double escape')
                self._finish_turn()
                return
            elif elapsed >= self._min_turn_s and self._front_min >= self._clear_dist:
                self._finish_turn()
                return

        self._cmd_pub.publish(self._current_twist)

    # ── Turn helpers ──────────────────────────────────────────────────────────

    def _start_turn(self, now: float):
        """Begin turning toward the side with more open space."""
        self._state = TURNING
        self._phase_start_t = now
        self._flipped = False

        # Pick direction: turn toward the more open side
        if self._left_min >= self._right_min:
            self._turn_dir = 1.0   # left (CCW)
            self.get_logger().info(
                f'Turning LEFT (left={self._left_min:.2f} > right={self._right_min:.2f})')
        else:
            self._turn_dir = -1.0  # right (CW)
            self.get_logger().info(
                f'Turning RIGHT (right={self._right_min:.2f} > left={self._left_min:.2f})')

        self._current_twist = Twist()
        self._current_twist.angular.z = self._turn_dir * self._turn_spd

    def _finish_turn(self):
        """Turn complete — resume forward and optionally trigger AI."""
        self._current_twist = Twist()
        self._cmd_pub.publish(self._current_twist)

        self._turn_count += 1
        self._state = FORWARD
        self._flipped = False

        self.get_logger().info(
            f'Path clear at {self._front_min:.2f} m — resuming forward '
            f'(turn #{self._turn_count})')

        if self._turn_count % self._label_every == 0:
            self.get_logger().info(f'Turn {self._turn_count} — requesting AI scene label')
            done = Bool()
            done.data = True
            self._done_pub.publish(done)


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
