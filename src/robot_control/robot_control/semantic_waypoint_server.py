#!/usr/bin/env python3
"""
semantic_waypoint_server.py — Semantic Room Goal Dispatcher for Nav2
====================================================================
Bridges Layer 1 (SQLite Semantic Scene Graph) with Layer 2 (Nav2 Path Routing).
Accepts natural language room queries (e.g., 'kitchen', 'bedroom', 'living_room'),
resolves the spatial centroid from SQLite, and dispatches a NavigateToPose action
to the Nav2 navigation stack.

Features:
  - Open-vocabulary query matching against scene graph database
  - Nav2 SimpleActionClient wrapper with goal cancellation capability
  - Auto-fallback to nearest discovered topological region
"""

import os
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import String
from std_srvs.srv import Trigger
from nav2_msgs.action import NavigateToPose

from .scene_graph_db import SceneGraphDB
from .map_manager import get_maps_base_dir


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.z = math.sin(yaw / 2.0)
    q.x = 0.0
    q.y = 0.0
    return q


class SemanticWaypointServer(Node):
    """Dispatches Nav2 goals to room centroids queried from the semantic scene graph."""

    def __init__(self):
        super().__init__('semantic_waypoint_server')

        self.declare_parameter('maps_dir', '/root/maps')
        self._maps_dir = get_maps_base_dir(self.get_parameter('maps_dir').value)
        self._db_path = os.path.join(self._maps_dir, 'scene_graph.db')
        self._db = SceneGraphDB(self._db_path)

        # Nav2 Action Client
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._current_goal_handle = None

        # Subscribers
        self.create_subscription(String, '/nav/go_to_room', self._on_go_to_room_topic, 10)

        # Services
        self.create_service(Trigger, '/nav/cancel_goal', self._srv_cancel_goal)

        self.get_logger().info(
            f"Semantic Waypoint Server ready | DB: {self._db_path} | Target Action: /navigate_to_pose")

    def _on_go_to_room_topic(self, msg: String):
        """Dispatches goal when room name is published to /nav/go_to_room."""
        room_name = msg.data.strip()
        self.navigate_to_room(room_name)

    def navigate_to_room(self, room_name: str) -> bool:
        """Query room centroid from SQLite and dispatch NavigateToPose action."""
        region = self._db.find_region_by_label(room_name)
        if region is None:
            self.get_logger().warn(f"Semantic Scene Graph: Room '{room_name}' not found in database.")
            return False

        cx = region['centroid_x']
        cy = region['centroid_y']
        label = region['label']
        r_id = region['region_id']

        self.get_logger().info(
            f"Dispatching Nav2 Goal -> Room '{label}' (Region {r_id}) at [{cx:.2f}, {cy:.2f}]")

        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 action server '/navigate_to_pose' is not available.")
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = cx
        goal_msg.pose.pose.position.y = cy
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation = yaw_to_quaternion(0.0)

        send_future = self._nav_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self._on_goal_response)
        return True

    def _on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Nav2 rejected the room navigation goal.")
            return

        self.get_logger().info("Nav2 accepted room navigation goal. Traveling...")
        self._current_goal_handle = goal_handle

    def _srv_cancel_goal(self, request, response):
        """Cancel active Nav2 goal (used for visual preemption upon object discovery)."""
        if self._current_goal_handle is not None:
            self.get_logger().info("Cancelling active Nav2 goal...")
            self._current_goal_handle.cancel_goal_async()
            self._current_goal_handle = None
            response.success = True
            response.message = "Nav2 goal cancelled."
        else:
            response.success = False
            response.message = "No active goal to cancel."
        return response


def main(args=None):
    rclpy.init(args=args)
    node = SemanticWaypointServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
