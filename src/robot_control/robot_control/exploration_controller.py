#!/usr/bin/env python3
"""
exploration_controller.py  — ultrasonic-reactive exploration
=============================================================
Drives the robot autonomously using the HC-SR04 ultrasonic sensor.
NO dependency on AI for movement — AI is only used for scene labelling.
V8
State machine
-------------
  FORWARD  → drive forward at move_speed
           → if filtered_range < obstacle_distance  → BACKING

  BACKING  → reverse for backup_s seconds to create clearance
           → then → TURNING

  TURNING  → spin in place (alternating direction)
           → until BOTH: min_turn_s elapsed AND filtered_range >= clear_distance
           → then → FORWARD

Exploration strategy
--------------------
  Always turn LEFT (CCW) when an obstacle is encountered.
  This implements a loose left-hand rule — the robot naturally
  traces room walls and works its way around the space systematically
  rather than zigzagging back and forth.

  If the path is STILL blocked after max_turn_s (default 8 s ≈ 250°),
  the robot switches to turning RIGHT for that turn only (corner escape),
  then resets back to left-preference for the next obstacle.

Subscribes:  /ultrasonic_range  (BEST_EFFORT QoS — matches arduino_bridge)
Publishes:   /cmd_vel, /robot/movement_complete

Parameters
----------
move_speed          float   0.15   m/s forward speed (slower = more reaction time)
turn_speed          float   0.55   rad/s turn speed
obstacle_distance   float   1.00   m — start backing/turning below this
emergency_stop_dist float   0.10   m — enter emergency reverse below this (any state)
backup_clear_dist   float   0.30   m — reverse until this distance is reached in emergency
backup_s            float   2.0    s — reverse duration before turning (normal obstacle)
min_turn_s          float   3.0    s — minimum turn time before checking clear (~90 deg at 0.55 rad/s)
max_turn_s          float   10.0   s — switch to right turn if still blocked (corner escape)
label_every         int     5      turns between AI label requests
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range

FORWARD        = 'forward'
BACKING        = 'backing'         # timed reverse after normal obstacle detect
EMERGENCY_BACK = 'emergency_back'  # distance-based reverse after entering no-go zone
TURNING        = 'turning'


class ExplorationController(Node):
    def __init__(self):
        super().__init__('exploration_controller')

        self.declare_parameter('move_speed',           0.15)
        self.declare_parameter('turn_speed',           0.55)
        self.declare_parameter('obstacle_distance',    1.00)
        self.declare_parameter('emergency_stop_dist',  0.10)
        self.declare_parameter('backup_clear_dist',    0.30)
        self.declare_parameter('backup_s',             2.0)
        self.declare_parameter('min_turn_s',           3.0)
        self.declare_parameter('max_turn_s',           10.0)
        self.declare_parameter('label_every',          5)
        # How long to wait for the first sensor reading before raising an alarm.
        # Set long enough for the arduino_bridge to connect (serial + 2 s reset).
        self.declare_parameter('sensor_timeout',       8.0)

        self._move_spd    = self.get_parameter('move_speed').value
        self._turn_spd    = self.get_parameter('turn_speed').value
        self._obs_dist    = self.get_parameter('obstacle_distance').value
        self._estop_dist   = self.get_parameter('emergency_stop_dist').value
        self._backup_clr  = self.get_parameter('backup_clear_dist').value
        self._backup_s    = self.get_parameter('backup_s').value
        self._min_turn_s  = self.get_parameter('min_turn_s').value
        self._max_turn_s  = self.get_parameter('max_turn_s').value
        self._label_every    = int(self.get_parameter('label_every').value)
        self._sensor_timeout = self.get_parameter('sensor_timeout').value

        # clear_dist: how far ahead must be open before we resume FORWARD.
        # Use 1.1x instead of 1.5x — 1.5x (1.5 m) is often unreachable in a
        # typical room and causes the robot to spin indefinitely.
        self._clear_dist  = self._obs_dist * 1.1

        self._cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self._done_pub = self.create_publisher(Bool,  '/robot/movement_complete', 10)

        # BEST_EFFORT QoS — must match arduino_bridge publisher
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(Range, '/ultrasonic_range', self._on_range, sensor_qos)

        # State
        self._state          = FORWARD
        self._turn_dir       = 1.0      # always start left (CCW); flipped only for corner escape
        self._turn_count     = 0
        self._flipped        = False    # True when we switched to right for corner escape
        self._filtered_range = 9.9      # glitch-filtered range
        self._raw_range      = None     # None until first valid reading arrives
        self._phase_start_t  = None     # when current BACKING or TURNING phase started
        self._current_twist  = Twist()
        self._log_tick       = 0        # counts control-loop ticks for 1 Hz status prints
        self._last_state     = None     # detect state changes for transition logging
        self._start_t        = self.get_clock().now().nanoseconds * 1e-9
        self._range_received = 0        # total Range msgs received (diagnostic)
        self._last_range_t   = None     # wall-clock time of last accepted reading
        # Arduino sends DATA at ~3 Hz (333 ms) but HiTechnic I2C reads in the same
        # loop cause gaps of 2-3 s.  Set stale threshold above that so we don't
        # incorrectly halt mid-manoeuvre.
        self._sensor_stale_s = 4.0      # halt if no reading for this many seconds

        # 10 Hz control loop
        self.create_timer(0.1, self._control_loop)

        self.get_logger().info(
            f'Exploration controller ready | '
            f'speed={self._move_spd} m/s | turn={self._turn_spd} rad/s | '
            f'obstacle={self._obs_dist} m | clear={self._clear_dist:.2f} m | '
            f'estop={self._estop_dist} m | backup_to={self._backup_clr} m | '
            f'backup_s={self._backup_s} s | min_turn={self._min_turn_s} s | '
            f'max_turn={self._max_turn_s} s | always turning LEFT first')

    # ── Sensor callback ───────────────────────────────────────────────────────

    def _on_range(self, msg: Range):
        """Accept in-range readings with targeted HC-SR04 glitch filter.

        The HC-SR04 blind-spot glitch always spikes to NEAR MAX RANGE (e.g. 3.9-4.0 m).
        It does NOT produce intermediate values like 3.0 m.  So we only reject a reading
        when it BOTH jumps > 2 m AND lands within 0.2 m of the sensor's max_range.

        This correctly accepts a legitimate backing-away jump like 0.04 → 3.0 m
        (safe — robot is moving away from the wall) while still catching 0.04 → 3.9 m
        blind-spot spikes.  The old ">2.0 m" rule was too broad and froze the sensor
        every time the robot reversed into open space.
        """
        self._range_received += 1
        if self._range_received <= 5 or self._range_received % 30 == 0:
            self.get_logger().info(
                f'[CONTROLLER] Range received #{self._range_received}: '
                f'{msg.range:.3f} m  (min={msg.min_range} max={msg.max_range})')

        if not (msg.min_range <= msg.range <= msg.max_range):
            return

        # First valid reading ever — seed and go.
        if self._raw_range is None:
            self._raw_range      = msg.range
            self._filtered_range = msg.range
            self._last_range_t   = self.get_clock().now().nanoseconds * 1e-9
            self.get_logger().info(f'First ultrasonic reading: {msg.range:.2f} m')
            return

        jump     = abs(msg.range - self._raw_range)
        near_max = msg.range >= (msg.max_range - 0.20)   # within 0.2 m of sensor ceiling

        if jump > 2.0 and near_max:
            # HC-SR04 blind-spot false-max spike: large jump AND lands at sensor ceiling.
            # Keep _raw_range at last accepted value so the cascade bug cannot repeat.
            self.get_logger().debug(
                f'Range glitch ignored (near-max spike): '
                f'{self._raw_range:.2f} → {msg.range:.2f} m (jump={jump:.2f})')
            return

        self._raw_range      = msg.range
        self._filtered_range = msg.range
        self._last_range_t   = self.get_clock().now().nanoseconds * 1e-9

    # ── 10 Hz control loop ────────────────────────────────────────────────────

    def _control_loop(self):
        now = self.get_clock().now().nanoseconds * 1e-9

        # ── Guard: hold still until the first sensor reading arrives ─────────
        if self._raw_range is None:
            # Publish a stop command every tick so the robot stays still
            self._cmd_pub.publish(Twist())
            waiting_s = now - self._start_t
            # Warn at 1 Hz
            if self._log_tick % 10 == 0:
                if waiting_s > self._sensor_timeout:
                    self.get_logger().warn(
                        f'*** NO ULTRASONIC DATA after {waiting_s:.0f} s — ROBOT HALTED ***\n'
                        f'  Check: (1) HC-SR04 wired to Arduino pins TRIG=12 ECHO=13\n'
                        f'         (2) arduino_bridge is publishing to /ultrasonic_range\n'
                        f'             run: ros2 topic echo /ultrasonic_range\n'
                        f'         (3) arduino_raw topic shows DATA lines:\n'
                        f'             run: ros2 topic echo /arduino_raw')
                else:
                    self.get_logger().info(
                        f'[SENSOR] waiting for first ultrasonic reading '
                        f'({waiting_s:.1f}/{self._sensor_timeout:.0f} s)')
            self._log_tick += 1
            return

        # ── 1 Hz status line ─────────────────────────────────────────────────
        self._log_tick += 1
        if self._log_tick % 10 == 0:
            if self._state == FORWARD:
                action = f'DRIVING FORWARD  @ {self._move_spd} m/s'
            elif self._state == BACKING:
                elapsed = now - self._phase_start_t if self._phase_start_t else 0.0
                action = f'BACKING UP       ({elapsed:.1f}/{self._backup_s:.1f} s)'
            elif self._state == EMERGENCY_BACK:
                action = f'EMERGENCY REVERSE @ {self._move_spd} m/s  (until {self._backup_clr} m)'
            else:  # TURNING
                elapsed = now - self._phase_start_t if self._phase_start_t else 0.0
                dirstr  = 'LEFT' if self._turn_dir > 0 else 'RIGHT'
                action  = f'TURNING {dirstr:<5} ({elapsed:.1f}/{self._max_turn_s:.1f} s max)'
            sensor_str = f'{self._filtered_range:.2f} m' if self._raw_range is not None else 'waiting…'
            self.get_logger().info(
                f'[SENSOR] ultrasonic={sensor_str}  '
                f'[ACTION] {action}')

        # ── Guard: halt if sensor has gone stale ─────────────────────────────
        if self._last_range_t is not None:
            stale_s = now - self._last_range_t
            if stale_s > self._sensor_stale_s:
                self._cmd_pub.publish(Twist())
                if self._log_tick % 10 == 0:
                    self.get_logger().warn(
                        f'[SENSOR STALE] No ultrasonic data for {stale_s:.1f} s '
                        f'— robot halted. Check /arduino_raw for DATA lines.')
                self._log_tick += 1
                return

        # ── Emergency: entered no-go zone ─────────────────────────────────────
        if self._filtered_range < self._estop_dist and self._state != EMERGENCY_BACK:
            self.get_logger().warn(
                f'EMERGENCY BACK — range={self._filtered_range:.2f} m — reversing until {self._backup_clr} m')
            self._state         = EMERGENCY_BACK
            self._current_twist = Twist()
            self._current_twist.linear.x = -self._move_spd

        if self._state == EMERGENCY_BACK:
            if self._filtered_range >= self._backup_clr:
                self.get_logger().info(
                    f'Emergency clear at {self._filtered_range:.2f} m — turning left')
                self._state         = TURNING
                self._phase_start_t = now
                self._flipped       = False
                self._turn_dir      = 1.0   # always left after emergency
                self._current_twist = Twist()
                self._current_twist.angular.z = self._turn_dir * self._turn_spd
            else:
                # Still too close — keep reversing
                self._current_twist = Twist()
                self._current_twist.linear.x = -self._move_spd
            self._cmd_pub.publish(self._current_twist)
            return

        if self._state == FORWARD:
            if self._filtered_range < self._obs_dist:
                self.get_logger().info(
                    f'>>> OBSTACLE DETECTED at {self._filtered_range:.2f} m '
                    f'(threshold={self._obs_dist} m) — starting {self._backup_s} s reverse')
                self._state         = BACKING
                self._phase_start_t = now
                self._flipped       = False
                self._turn_dir      = 1.0   # reset to left every new obstacle
                self._current_twist = Twist()
                self._current_twist.linear.x = -self._move_spd
            else:
                self._current_twist = Twist()
                self._current_twist.linear.x = self._move_spd

        elif self._state == BACKING:
            elapsed = now - self._phase_start_t
            remaining = max(0.0, self._backup_s - elapsed)
            if self._log_tick % 5 == 0:  # 0.5 Hz while backing so you can see progress
                self.get_logger().info(
                    f'  BACKING: {elapsed:.1f}/{self._backup_s:.1f} s  '
                    f'range={self._filtered_range:.2f} m  '
                    f'({remaining:.1f} s left)')
            if elapsed >= self._backup_s:
                self.get_logger().info(
                    f'>>> BACKUP DONE — spinning LEFT until path > {self._clear_dist:.2f} m clear')
                self._state         = TURNING
                self._phase_start_t = now
                self._current_twist = Twist()
                self._current_twist.angular.z = self._turn_dir * self._turn_spd

        elif self._state == TURNING:
            elapsed = now - self._phase_start_t
            dirstr  = 'LEFT' if self._turn_dir > 0 else 'RIGHT'
            # First escape: still blocked after max_turn_s — switch to RIGHT once
            if elapsed >= self._max_turn_s and not self._flipped:
                self.get_logger().warn(
                    f'>>> CORNER ESCAPE — still blocked after {self._max_turn_s:.1f} s '
                    f'(range={self._filtered_range:.2f} m) — switching to RIGHT turn')
                self._flipped       = True
                self._turn_dir      = -1.0
                self._phase_start_t = now
                self._current_twist = Twist()
                self._current_twist.angular.z = self._turn_dir * self._turn_spd
            # Second escape: corner-escape turn ALSO timed out — force forward to
            # break out of any fully-enclosed spin loop.
            elif elapsed >= self._max_turn_s and self._flipped:
                self.get_logger().warn(
                    f'>>> FORCED FORWARD — still blocked after double escape '
                    f'(range={self._filtered_range:.2f} m) — driving forward anyway')
                self._finish_turn()
                return
            elif elapsed >= self._min_turn_s and self._filtered_range >= self._clear_dist:
                self._finish_turn()
                return

        self._cmd_pub.publish(self._current_twist)

    # ── Turn finished ─────────────────────────────────────────────────────────

    def _finish_turn(self):
        self._current_twist = Twist()
        self._cmd_pub.publish(self._current_twist)

        self._turn_dir   = 1.0   # always reset to left for next obstacle
        self._flipped    = False
        self._turn_count += 1
        self._state       = FORWARD

        self.get_logger().info(
            f'Path clear at {self._filtered_range:.2f} m — resuming forward '
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
