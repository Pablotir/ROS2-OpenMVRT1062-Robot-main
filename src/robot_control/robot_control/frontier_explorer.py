#!/usr/bin/env python3
"""
frontier_explorer.py — DISABLED
================================
This node is not used. The active exploration stack is:

  slam_toolbox          — builds /map from /scan + /odom
  exploration_controller — drives the robot (reads /scan, writes /cmd_vel)

This file exists only so the package builds without errors.
"""

import rclpy
from rclpy.node import Node


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')
        self.get_logger().info(
            'frontier_explorer is disabled. '
            'Motion is handled by exploration_controller.')


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
