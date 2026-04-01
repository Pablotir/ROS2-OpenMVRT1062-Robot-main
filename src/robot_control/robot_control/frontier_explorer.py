#!/usr/bin/env python3
"""
frontier_explorer.py
====================
Python-native frontier-based exploration node for ROS2/Nav2.

Reads the RTAB-Map occupancy grid (/map) and automatically navigates
the robot to unexplored frontiers using the Nav2 NavigateToPose action.

Algorithm
---------
1. Subscribe to /map (nav_msgs/OccupancyGrid).
2. Every `planner_frequency` seconds:
   a. Find all frontier cells — free cells (0) adjacent to unknown cells (-1).
   b. Cluster frontier cells into contiguous frontier groups.
   c. Score each cluster: gain = size, cost = distance_from_robot.
      Best frontier = argmax(gain_scale * size - potential_scale * distance).
   d. If best frontier is far enough away and bigger than min_frontier_size,
      send a NavigateToPose action goal to Nav2.
3. If no frontiers remain → exploration is complete, node logs and stops.

Subscribes
----------
  /map              nav_msgs/OccupancyGrid    RTAB-Map 2D occupancy grid
  /odom             nav_msgs/Odometry         Robot pose (fallback if no TF)

Requires
--------
  Nav2 stack running (controller_server, planner_server, bt_navigator)

Parameters
----------
  planner_frequency   float   0.33    Hz — how often to re-evaluate frontiers
  min_frontier_size   float   0.75    m  — ignore frontiers smaller than this
  potential_scale     float   3.0        — cost weight for distance
  gain_scale          float   1.0        — gain weight for frontier size
  robot_base_frame    str     base_link
  progress_timeout    float   30.0    s  — cancel goal if no progress after this
  startup_delay       float   15.0    s  — wait for Nav2 to fully boot before sending
"""

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Bool


# ── Occupancy values ──────────────────────────────────────────────────────────
FREE     = 0
UNKNOWN  = -1
OCCUPIED_THRESHOLD = 50    # cells ≥ this value are considered occupied


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('planner_frequency',   0.33)
        self.declare_parameter('min_frontier_size',   0.75)
        self.declare_parameter('potential_scale',     3.0)
        self.declare_parameter('gain_scale',          1.0)
        self.declare_parameter('robot_base_frame',    'base_link')
        self.declare_parameter('progress_timeout',    30.0)
        self.declare_parameter('startup_delay',       15.0)

        self._freq          = self.get_parameter('planner_frequency').value
        self._min_size_m    = self.get_parameter('min_frontier_size').value
        self._pot_scale     = self.get_parameter('potential_scale').value
        self._gain_scale    = self.get_parameter('gain_scale').value
        self._timeout       = self.get_parameter('progress_timeout').value
        self._startup_delay = self.get_parameter('startup_delay').value

        # ── State ─────────────────────────────────────────────────────────────
        self._map: OccupancyGrid | None = None
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._navigating = False
        self._goal_handle = None
        self._goal_sent_t = 0.0
        self._enabled = True
        self._start_time = time.monotonic()
        self._nav2_ready = False

        # Blacklist recently-failed goals so we don't retry the same spot
        self._blacklist: list[tuple[float, float]] = []
        self._blacklist_radius = 0.5   # metres

        # ── Nav2 action client ────────────────────────────────────────────────
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # ── Map subscriber (transient_local so we get the last published map) ─
        map_qos = QoSProfile(
            reliability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(OccupancyGrid, '/map', self._on_map, map_qos)

        # ── Odometry subscriber (robot pose) ──────────────────────────────────
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(Odometry, '/odom', self._on_odom, odom_qos)

        # ── Pause/resume topic (mirrors explore_lite interface) ───────────────
        self.create_subscription(Bool, '/explore/resume', self._on_resume, 10)

        # ── Planning timer ────────────────────────────────────────────────────
        period = 1.0 / max(self._freq, 0.01)
        self.create_timer(period, self._plan)

        self.get_logger().info(
            f'Frontier Explorer ready | freq={self._freq:.2f} Hz | '
            f'min_size={self._min_size_m} m | '
            f'potential_scale={self._pot_scale} | gain_scale={self._gain_scale} | '
            f'startup_delay={self._startup_delay:.0f} s')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_map(self, msg: OccupancyGrid):
        self._map = msg

    def _on_odom(self, msg: Odometry):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y

    def _on_resume(self, msg: Bool):
        self._enabled = msg.data
        self.get_logger().info(
            f'Exploration {"resumed" if self._enabled else "paused"}')

    # ── Main planning callback ────────────────────────────────────────────────

    def _plan(self):
        if not self._enabled or self._map is None:
            return

        # Wait for Nav2 to finish booting before sending the first goal
        if not self._nav2_ready:
            elapsed = time.monotonic() - self._start_time
            if elapsed < self._startup_delay:
                return
            self._nav2_ready = True
            self.get_logger().info('Startup delay elapsed — beginning exploration')

        # Check if current goal has timed out
        if self._navigating:
            elapsed = time.monotonic() - self._goal_sent_t
            if elapsed > self._timeout:
                self.get_logger().warn(
                    f'Goal timeout after {elapsed:.0f} s — blacklisting and replanning')
                if self._goal_handle is not None:
                    self._goal_handle.cancel_goal_async()
                self._navigating = False
            else:
                return   # still navigating, wait

        frontiers = self._find_frontiers()
        if not frontiers:
            self.get_logger().info(
                'No frontiers found — exploration complete!')
            self._enabled = False
            return

        best = self._select_best_frontier(frontiers)
        if best is None:
            self.get_logger().info('All frontiers are blacklisted — clearing blacklist')
            self._blacklist.clear()
            return

        bx, by = best
        dist = math.hypot(bx - self._robot_x, by - self._robot_y)
        self.get_logger().info(
            f'Sending goal to frontier ({bx:.2f}, {by:.2f}) | dist={dist:.2f} m')
        self._send_goal(bx, by)

    # ── Frontier discovery ────────────────────────────────────────────────────

    def _find_frontiers(self) -> list[list[tuple[float, float]]]:
        """Return a list of frontier clusters, each a list of (x, y) world coords."""
        m = self._map
        w, h   = m.info.width, m.info.height
        res    = m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y
        data   = m.data

        def idx(r, c): return r * w + c
        def world(r, c):
            return ox + (c + 0.5) * res, oy + (r + 0.5) * res

        # Find frontier cells: FREE cells that have at least one UNKNOWN neighbour
        frontier_mask = [False] * (w * h)
        for row in range(1, h - 1):
            for col in range(1, w - 1):
                i = idx(row, col)
                if data[i] != FREE:
                    continue
                neighbours = [
                    data[idx(row - 1, col)],
                    data[idx(row + 1, col)],
                    data[idx(row, col - 1)],
                    data[idx(row, col + 1)],
                ]
                if UNKNOWN in neighbours:
                    frontier_mask[i] = True

        # BFS cluster frontier cells
        visited = [False] * (w * h)
        clusters = []
        min_cells = max(1, int(self._min_size_m / res))

        for start in range(w * h):
            if not frontier_mask[start] or visited[start]:
                continue
            cluster_pts = []
            queue = [start]
            visited[start] = True
            while queue:
                cur = queue.pop()
                r, c = divmod(cur, w)
                cluster_pts.append(world(r, c))
                for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if 0 <= nr < h and 0 <= nc < w:
                        ni = idx(nr, nc)
                        if frontier_mask[ni] and not visited[ni]:
                            visited[ni] = True
                            queue.append(ni)
            if len(cluster_pts) >= min_cells:
                clusters.append(cluster_pts)

        return clusters

    def _is_blacklisted(self, x: float, y: float) -> bool:
        for bx, by in self._blacklist:
            if math.hypot(x - bx, y - by) < self._blacklist_radius:
                return True
        return False

    def _select_best_frontier(
            self, clusters: list[list[tuple[float, float]]]
    ) -> tuple[float, float] | None:
        """Score and return the centroid of the best frontier cluster."""
        best_score = -float('inf')
        best_xy = None

        for cluster in clusters:
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            dist = math.hypot(cx - self._robot_x, cy - self._robot_y)
            if dist < 0.3:
                continue   # already there, skip
            if self._is_blacklisted(cx, cy):
                continue
            size = len(cluster)
            score = self._gain_scale * size - self._pot_scale * dist
            if score > best_score:
                best_score = score
                best_xy = (cx, cy)

        return best_xy

    # ── Nav2 goal sending ─────────────────────────────────────────────────────

    def _send_goal(self, x: float, y: float):
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('Nav2 action server not available yet')
            return

        # Compute orientation facing toward the goal
        dx = x - self._robot_x
        dy = y - self._robot_y
        yaw = math.atan2(dy, dx)

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        # Convert yaw to quaternion (rotation about Z only)
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self._navigating = True
        self._goal_sent_t = time.monotonic()
        self._current_goal = (x, y)

        future = self._nav_client.send_goal_async(goal)
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.get_logger().warn('Goal rejected by Nav2 — blacklisting')
            self._blacklist.append(self._current_goal)
            self._navigating = False
            return
        self.get_logger().info('Goal accepted — navigating...')
        result_future = self._goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_result(self, future):
        self._navigating = False
        status = future.result().status
        if status == 4:   # STATUS_SUCCEEDED
            self.get_logger().info('Navigation succeeded — replanning for next frontier')
        elif status == 6:  # STATUS_CANCELED
            self.get_logger().warn('Navigation cancelled')
        else:
            self.get_logger().warn(f'Navigation failed (status={status}) — blacklisting goal')
            self._blacklist.append(self._current_goal)


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
