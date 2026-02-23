#!/usr/bin/env python3
"""
exploration_controller.py  (updated)
=====================================
Translates AI navigation decisions from ai_navigator into proper ROS2
cmd_vel Twist messages consumed by jetson_bot_slam's motor_driver_node.

BEFORE: published raw '/llm/motor_plan' strings with wrong mecanum signs.
NOW:    publishes geometry_msgs/Twist on '/cmd_vel' — motor_driver_node
        handles the full mecanum inverse kinematics correctly.

Subscribes
----------
/ai/direction            std_msgs/String   'forward' | 'left' | 'right' | 'stop'

Publishes
---------
/cmd_vel                 geometry_msgs/Twist   velocity for motor_driver_node
/robot/movement_complete std_msgs/Bool         signals ai_navigator to take next frame

Parameters
----------
move_speed      (float, default 0.25)  m/s forward/backward speed
turn_speed      (float, default 0.50)  rad/s rotation speed
move_duration   (float, default 3.0)   seconds per directional step
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist


class ExplorationController(Node):
    def __init__(self):
        super().__init__('exploration_controller')

        self.declare_parameter('move_speed',    0.25)   # m/s
        self.declare_parameter('turn_speed',    0.50)   # rad/s
        self.declare_parameter('move_duration', 3.0)    # seconds

        self._move_spd = self.get_parameter('move_speed').value
        self._turn_spd = self.get_parameter('turn_speed').value
        self._move_dur = self.get_parameter('move_duration').value

        # ── Publishers ────────────────────────────────────────────────────────
        self._cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self._done_pub = self.create_publisher(Bool,  '/robot/movement_complete', 10)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(String, '/ai/direction', self._on_direction, 10)

        # ── Internal state ────────────────────────────────────────────────────
        self._current_twist = Twist()   # zero by default
        self._publish_timer = None      # republishes at 10 Hz while moving
        self._stop_timer    = None      # one-shot: stop after move_duration

        self.get_logger().info(
            f'Exploration controller ready | '
            f'speed={self._move_spd} m/s | turn={self._turn_spd} rad/s | '
            f'step={self._move_dur} s')

    # ── Direction callback ────────────────────────────────────────────────────

    def _on_direction(self, msg: String):
        direction = msg.data.strip().lower()
        self.get_logger().info(f'Direction received: {direction}')

        self._cancel_timers()

        twist = Twist()

        if 'stop' in direction:
            self._publish_zero()
            return
        elif 'forward' in direction:
            twist.linear.x  =  self._move_spd
        elif 'backward' in direction or 'back' in direction:
            twist.linear.x  = -self._move_spd
        elif 'strafe_left' in direction or 'slide_left' in direction:
            twist.linear.y  =  self._move_spd    # mecanum lateral
        elif 'strafe_right' in direction or 'slide_right' in direction:
            twist.linear.y  = -self._move_spd
        elif 'left' in direction:
            twist.angular.z =  self._turn_spd    # CCW turn
        elif 'right' in direction:
            twist.angular.z = -self._turn_spd    # CW turn
        else:
            twist.linear.x  =  self._move_spd    # default: nudge forward

        self._current_twist = twist
        self._start_movement()

    # ── Movement helpers ──────────────────────────────────────────────────────

    def _start_movement(self):
        """Publish twist at 10 Hz for move_duration seconds, then stop."""
        # 10 Hz keeps motor_driver's 0.5 s cmd_vel timeout well satisfied
        self._publish_timer = self.create_timer(0.1, self._republish)
        self._stop_timer    = self.create_timer(self._move_dur, self._finish_movement)

    def _republish(self):
        self._cmd_pub.publish(self._current_twist)

    def _finish_movement(self):
        self._cancel_timers()
        self._publish_zero()
        self.get_logger().info('Step complete — signalling ai_navigator')
        done = Bool()
        done.data = True
        self._done_pub.publish(done)

    def _publish_zero(self):
        self._cmd_pub.publish(Twist())

    def _cancel_timers(self):
        if self._publish_timer is not None:
            self._publish_timer.cancel()
            self._publish_timer = None
        if self._stop_timer is not None:
            self._stop_timer.cancel()
            self._stop_timer = None


def main(args=None):
    rclpy.init(args=args)
    node = ExplorationController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._cancel_timers()
        node._publish_zero()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
