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
   a. Find all frontier cells — free cells within 2 cells of unknown space.
      This 2-cell-deep check handles RTAB-Map ray tracing, which always
      places a wall (occupied cell) between free and unknown space.
   b. Map-edge free cells are also frontiers (space beyond grid is unknown).
   c. Cluster frontier cells into contiguous groups.
   d. Score each cluster: gain = size, cost = distance_from_robot.
      Best frontier = argmax(gain_scale * size - potential_scale * distance).
   e. If best frontier is far enough away and big enough, send a
      NavigateToPose action goal to Nav2.
3. If no frontiers are found for several cycles, perform a one-time
   spin-in-place (~360°) to grow the map before retrying.
4. If no frontiers remain after spinning → exploration is complete.

Subscribes
----------
  /map              nav_msgs/OccupancyGrid    RTAB-Map 2D occupancy grid
  /odom             nav_msgs/Odometry         Robot pose (fallback if no TF)

Publishes
---------
  /cmd_vel          geometry_msgs/Twist       Spin-in-place bootstrap only

Requires
--------
  Nav2 stack running (controller_server, planner_server, bt_navigator)

Parameters
----------
  planner_frequency   float   0.33    Hz — how often to re-evaluate frontiers
  min_frontier_size   float   0.30    m  — ignore frontiers smaller than this
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
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Bool


# ── Occupancy values ──────────────────────────────────────────────────────────
FREE     = 0
UNKNOWN  = -1

# ── Spin-in-place bootstrap ──────────────────────────────────────────────────
_SPIN_TRIGGER     = 5      # consecutive empty checks before spinning
_SPIN_SPEED       = 0.5    # rad/s
_SPIN_DURATION    = 13.0   # seconds (~full 360° + margin)
_SPIN_PUB_HZ     = 20     # publish rate — must outpace velocity_smoother (10 Hz)


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('planner_frequency',   0.33)
        self.declare_parameter('min_frontier_size',   0.30)
        self.declare_parameter('potential_scale',     5.0)
        self.declare_parameter('gain_scale',          1.0)
        self.declare_parameter('robot_base_frame',    'base_link')
        self.declare_parameter('progress_timeout',    30.0)
        self.declare_parameter('startup_delay',       10.0)

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
        self._no_frontier_count = 0

        # Goal cooldown to prevent slamming Nav2 during recovery
        self._last_fail_t = 0.0
        self._fail_cooldown = 5.0

        # Spin-in-place bootstrap state
        self._spin_done = False
        self._spinning = False
        self._spin_start_t = 0.0

        # Blacklist recently-failed goals
        self._blacklist: list[tuple[float, float]] = []
        self._blacklist_radius = 0.5

        # Stats counter
        self._stats_counter = 0

        # ── Nav2 action client ────────────────────────────────────────────────
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # ── cmd_vel publisher (for spin-in-place only) ────────────────────────
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── Spin timer (created on demand, publishes at 20 Hz) ────────────────
        self._spin_timer = None

        # ── Map subscriber (transient_local so we get the last published map) ─
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
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

        # ── Pause/resume topic ────────────────────────────────────────────────
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

    # ── Spin-in-place helpers ─────────────────────────────────────────────────

    def _start_spin(self):
        """Begin spinning in place at high publish rate."""
        self._spinning = True
        self._spin_start_t = time.monotonic()
        self.get_logger().info(
            f'Starting spin-in-place ({_SPIN_SPEED:.1f} rad/s, '
            f'{_SPIN_DURATION:.0f}s, {_SPIN_PUB_HZ} Hz publish)')
        # Create a fast timer that publishes the spin command
        self._spin_timer = self.create_timer(
            1.0 / _SPIN_PUB_HZ, self._spin_tick)

    def _spin_tick(self):
        """Called at 20 Hz during spin to outpace velocity_smoother."""
        elapsed = time.monotonic() - self._spin_start_t
        if elapsed < _SPIN_DURATION:
            twist = Twist()
            twist.angular.z = _SPIN_SPEED
            self._cmd_pub.publish(twist)
        else:
            # Spin complete — stop and clean up
            twist = Twist()
            self._cmd_pub.publish(twist)
            if self._spin_timer is not None:
                self._spin_timer.cancel()
                self._spin_timer = None
            self._spinning = False
            self._spin_done = True
            self._no_frontier_count = 0
            self.get_logger().info(
                'Spin-in-place complete — resuming frontier search')

    # ── Main planning callback ────────────────────────────────────────────────

    def _plan(self):
        if not self._enabled or self._map is None:
            return

        # Wait for Nav2 to finish booting
        if not self._nav2_ready:
            elapsed = time.monotonic() - self._start_time
            if elapsed < self._startup_delay:
                return
            self._nav2_ready = True
            self.get_logger().info('Startup delay elapsed — beginning exploration')

        # Don't plan while spinning
        if self._spinning:
            return

        # Cooldown after a navigation failure to let Nav2 recover
        if not self._navigating and time.monotonic() - self._last_fail_t < self._fail_cooldown:
            return

        # Check if current goal has timed out
        if self._navigating:
            elapsed = time.monotonic() - self._goal_sent_t
            if elapsed > self._timeout:
                self.get_logger().warn(
                    f'Goal timeout after {elapsed:.0f} s — blacklisting and replanning')
                self._blacklist.append(self._current_goal)
                if self._goal_handle is not None:
                    self._goal_handle.cancel_goal_async()
                self._navigating = False
                self._last_fail_t = time.monotonic()
            else:
                return

        # ── Log map stats for diagnostics ─────────────────────────────────
        self._log_map_stats()

        frontiers = self._find_frontiers()
        if not frontiers:
            self._no_frontier_count += 1

            # Trigger spin-in-place if stuck and haven't spun yet
            if not self._spin_done and self._no_frontier_count >= _SPIN_TRIGGER:
                self.get_logger().info(
                    f'No frontiers for {self._no_frontier_count} checks — '
                    'starting spin-in-place to grow the map...')
                self._start_spin()
                return

            if self._no_frontier_count % 10 == 1:
                self.get_logger().info(
                    f'No frontiers found (check #{self._no_frontier_count}) — '
                    'waiting for map to grow...')
            return
        self._no_frontier_count = 0

        best = self._select_best_frontier(frontiers)
        if best is None:
            self.get_logger().info('All frontiers blacklisted — clearing blacklist')
            self._blacklist.clear()
            return

        bx, by = best
        dist = math.hypot(bx - self._robot_x, by - self._robot_y)
        self.get_logger().info(
            f'Sending goal to frontier ({bx:.2f}, {by:.2f}) | dist={dist:.2f} m | '
            f'{len(frontiers)} clusters found')
        self._send_goal(bx, by)

    # ── Map diagnostics ──────────────────────────────────────────────────────

    def _log_map_stats(self):
        """Log map dimensions and cell-type counts periodically."""
        self._stats_counter += 1
        if self._stats_counter % 10 != 1:
            return

        m = self._map
        total = m.info.width * m.info.height
        data = m.data
        n_free = sum(1 for v in data if v == FREE)
        n_unk  = sum(1 for v in data if v == UNKNOWN)
        n_occ  = total - n_free - n_unk
        self.get_logger().info(
            f'Map stats: {m.info.width}x{m.info.height} '
            f'({m.info.width * m.info.resolution:.1f}x'
            f'{m.info.height * m.info.resolution:.1f} m) | '
            f'free={n_free} unk={n_unk} occ={n_occ}')

    # ── Frontier discovery (2-cell-deep adjacency) ────────────────────────────

    def _find_frontiers(self) -> list[list[tuple[float, float]]]:
        """Return frontier clusters. Each cluster is a list of (x, y) coords.

        A frontier cell is a FREE cell that is within 2 cells of UNKNOWN space.
        This handles RTAB-Map ray tracing, where occupied cells (walls) always
        separate free from unknown: FREE → OCCUPIED → UNKNOWN.

        Map-edge cells treat out-of-bounds as UNKNOWN (implicit frontier).
        """
        m = self._map
        w, h   = m.info.width, m.info.height
        res    = m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y
        data   = m.data

        def idx(r, c):
            return r * w + c

        def world(r, c):
            return ox + (c + 0.5) * res, oy + (r + 0.5) * res

        def get_cell(r, c):
            """Return cell value, treating out-of-bounds as UNKNOWN."""
            if r < 0 or r >= h or c < 0 or c >= w:
                return UNKNOWN
            return data[idx(r, c)]

        # Build a set of cells that are within 1 cell of UNKNOWN.
        # These are "near-unknown" cells — they may be occupied walls.
        near_unknown = set()
        for row in range(h):
            for col in range(w):
                # Check if this cell has any UNKNOWN neighbor (including OOB)
                if (get_cell(row - 1, col) == UNKNOWN or
                    get_cell(row + 1, col) == UNKNOWN or
                    get_cell(row, col - 1) == UNKNOWN or
                    get_cell(row, col + 1) == UNKNOWN):
                    near_unknown.add((row, col))
                # Also check if the cell itself is at the map edge
                # (OOB neighbors handled by get_cell returning UNKNOWN)

        # A frontier cell is a FREE cell that is:
        #   - in near_unknown (directly adjacent to unknown), OR
        #   - adjacent to a cell in near_unknown (2 cells from unknown —
        #     handles the FREE→OCCUPIED→UNKNOWN wall pattern)
        frontier_mask = [False] * (w * h)
        for row in range(h):
            for col in range(w):
                i = idx(row, col)
                if data[i] != FREE:
                    continue

                # Check: is this free cell in near_unknown?
                if (row, col) in near_unknown:
                    frontier_mask[i] = True
                    continue

                # Check: is any 4-connected neighbor in near_unknown?
                for nr, nc in ((row-1, col), (row+1, col),
                               (row, col-1), (row, col+1)):
                    if (nr, nc) in near_unknown:
                        frontier_mask[i] = True
                        break

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
                continue
            
            # Cap maximum frontier distance to 3.0 meters
            # The robot shouldn't try to navigate through huge swaths of unknown
            if dist > 5.0:
                continue
                
            if self._is_blacklisted(cx, cy):
                continue
                
            size = len(cluster)
            # Quadratic distance penalty forces the planner to pick nearby goals
            score = self._gain_scale * size - self._pot_scale * (dist ** 2)
            
            if score > best_score:
                best_score = score
                best_xy = (cx, cy)

        return best_xy

    # ── Nav2 goal sending ─────────────────────────────────────────────────────

    def _send_goal(self, x: float, y: float):
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('Nav2 action server not available yet')
            return

        dx = x - self._robot_x
        dy = y - self._robot_y
        yaw = math.atan2(dy, dx)

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
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
            self._last_fail_t = time.monotonic()
            return
        self.get_logger().info('Goal accepted — navigating...')
        result_future = self._goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_result(self, future):
        self._navigating = False
        status = future.result().status
        if status == 4:
            self.get_logger().info('Navigation succeeded — replanning for next frontier')
        elif status == 6:
            self.get_logger().warn('Navigation cancelled (e.g. by recovery behaviour)')
            self._last_fail_t = time.monotonic()
        elif status == 5:
            self.get_logger().warn('Navigation aborted/failed completely (status=5) — blacklisting goal')
            self._blacklist.append(self._current_goal)
            self._last_fail_t = time.monotonic()
        else:
            self.get_logger().warn(f'Navigation ended (status={status}) — waiting for cooldown')
            self._last_fail_t = time.monotonic()


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
