#!/usr/bin/env python3
"""
exploration_controller.py  — ultrasonic-reactive exploration
=============================================================
Drives the robot autonomously using the HC-SR04 ultrasonic sensor.
NO dependency on AI for movement — AI is only used for scene labelling.

Strategy
--------
  1. Drive forward continuously.
  2. If ultrasonic range < obstacle_distance → stop, turn away, resume.
  3. Alternate turn direction each time to avoid spinning in circles.
  4. After every `label_every` steps, publish /robot/movement_complete
     so ai_navigator can grab a frame and label the scene.

Subscribes
----------
/ultrasonic_range        sensor_msgs/Range

Publishes
---------
/cmd_vel                 geometry_msgs/Twist
/robot/movement_complete std_msgs/Bool   (triggers AI labelling every N steps)

Parameters
----------
move_speed          float   0.25   m/s forward speed
turn_speed          float   0.60   rad/s turn speed
obstacle_distance   float   0.55   m — turn if closer than this
turn_duration       float   1.2    s — how long to turn before resuming forward
label_every         int     5      steps between AI scene label requests
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range


class ExplorationController(Node):
    def __init__(self):
        super().__init__('exploration_controller')

        self.declare_parameter('move_speed',        0.25)
        self.declare_parameter('turn_speed',        0.60)
        self.declare_parameter('obstacle_distance', 0.55)
        self.declare_parameter('turn_duration',     1.2)
        self.declare_parameter('label_every',       5)

        self._move_spd   = self.get_parameter('move_speed').value
        self._turn_spd   = self.get_parameter('turn_speed').value
        self._obs_dist   = self.get_parameter('obstacle_distance').value
        self._turn_dur   = self.get_parameter('turn_duration').value
        self._label_every = int(self.get_parameter('label_every').value)

        self._cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self._done_pub = self.create_publisher(Bool,  '/robot/movement_complete', 10)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(Range, '/ultrasonic_range', self._on_range, sensor_qos)

        # State
        self._turning        = False   # True while executing a turn
        self._turn_direction = 1.0     # 1.0 = CCW (left), -1.0 = CW (right)
        self._step_count     = 0
        self._last_range     = 9.9    # assume clear on start
        self._turn_timer     = None
        self._current_twist  = Twist()  # republished every cycle

        # 10 Hz control loop — runs always, publishes self._current_twist
        self.create_timer(0.1, self._control_loop)

        self.get_logger().info(
            f'Exploration controller ready | '
            f'speed={self._move_spd} m/s | obstacle threshold={self._obs_dist} m')

    def _on_range(self, msg: Range):
        if msg.range >= msg.min_range and msg.range <= msg.max_range:
            self._last_range = msg.range

    def _control_loop(self):
        # Always republish _current_twist at 10 Hz so motor_driver never times out
        if self._turning:
            self._cmd_pub.publish(self._current_twist)
            return

        if self._last_range < self._obs_dist:
            self.get_logger().info(
                f'Obstacle at {self._last_range:.2f} m — turning '
                f'{"left" if self._turn_direction > 0 else "right"}')
            self._start_turn()
        else:
            self._current_twist = Twist()
            self._current_twist.linear.x = self._move_spd
            self._cmd_pub.publish(self._current_twist)

    def _start_turn(self):
        self._turning = True
        # Set turn twist — _control_loop will keep republishing it at 10 Hz
        self._current_twist = Twist()
        self._current_twist.angular.z = self._turn_direction * self._turn_spd
        # Flip direction for next obstacle
        self._turn_direction *= -1.0
        self._turn_timer = self.create_timer(self._turn_dur, self._finish_turn)

    def _finish_turn(self):
        if self._turn_timer:
            self._turn_timer.cancel()
            self._turn_timer = None

        self._current_twist = Twist()  # stop
        self._cmd_pub.publish(self._current_twist)
        self._turning = False
        self._step_count += 1

        if self._step_count % self._label_every == 0:
            self.get_logger().info(f'Step {self._step_count} — requesting AI scene label')
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
    node._cmd_pub.publish(Twist())  # stop on exit
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
