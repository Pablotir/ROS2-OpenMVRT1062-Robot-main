#!/usr/bin/env python3
"""
frontier_explorer.py
====================
Python-native frontier-based exploration node for ROS2/Nav2.

Reads the slam_toolbox occupancy grid (/map) and automatically navigates
the robot to unexplored frontiers using the Nav2 NavigateToPose action.

Algorithm
---------
1. Subscribe to /map (nav_msgs/OccupancyGrid).
2. Every `planner_frequency` seconds:
   a. Find all frontier cells — free cells within 2 cells of unknown space.
      This 2-cell-deep check handles ray tracing, which always
      places a wall (occupied cell) between free and unknown space.
   b. Map-edge free cells are also frontiers (space beyond grid is unknown).
   c. Cluster frontier cells into contiguous groups.
   d. Score each cluster: gain = size, cost = distance_from_robot.
      Best frontier = argmax(gain_scale * size - potential_scale * distance²).
   e. If best frontier is far enough away and big enough, send a
      NavigateToPose action goal to Nav2.
3. If the robot is stuck spinning (barely translating for 10s), stop with
   a fatal diagnostic printout.
4. If navigation fails 3 times consecutively, stop with diagnostic output.

Subscribes
----------
  /map              nav_msgs/OccupancyGrid    slam_toolbox 2D occupancy grid
  /odom             nav_msgs/Odometry         Robot pose (fallback if no TF)

Publishes
---------
  /cmd_vel          geometry_msgs/Twist       Emergency stop only

Requires
--------
  Nav2 stack running (controller_server, planner_server, bt_navigator)

Parameters
----------
  planner_frequency   float   0.33    Hz — how often to re-evaluate frontiers
  min_frontier_size   float   0.30    m  — ignore frontiers smaller than this
  potential_scale     float   5.0        — cost weight for distance
  gain_scale          float   1.0        — gain weight for frontier size
  robot_base_frame    str     base_link
  progress_timeout    float   30.0    s  — cancel goal if no progress after this
  startup_delay       float   10.0    s  — wait for Nav2 to fully boot before sending
"""

import math
import time
import collections
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Bool
from builtin_interfaces.msg import Time as TimeMsg

import tf2_ros


# ── Occupancy values ──────────────────────────────────────────────────────────
FREE     = 0
UNKNOWN  = -1

# ── Spin detection ────────────────────────────────────────────────────────────
_STUCK_WINDOW     = 10.0   # seconds of position history to check
_STUCK_THRESHOLD  = 0.10   # if robot moves less than 10 cm in the window, it's stuck
_STUCK_CHECK_HZ   = 2.0    # how often to sample position for spin detection


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
        self._robot_x = 0.0       # position in MAP frame (via TF)
        self._robot_y = 0.0
        self._robot_x_odom = 0.0  # fallback: position in ODOM frame
        self._robot_y_odom = 0.0
        self._navigating = False
        self._goal_handle = None
        self._goal_sent_t = 0.0
        self._current_goal = (0.0, 0.0)
        self._enabled = True
        self._start_time = time.monotonic()
        self._nav2_ready = False
        self._no_frontier_count = 0

        # Generation counter — prevents stale callbacks from
        # cancelled goals from corrupting _navigating state
        self._goal_gen = 0

        # Goal cooldown to prevent slamming Nav2 during recovery
        self._last_fail_t = 0.0
        self._fail_cooldown = 5.0

        # Blacklist recently-failed goals
        self._blacklist: list[tuple[float, float]] = []
        self._blacklist_radius = 0.5

        # Stats counter
        self._stats_counter = 0

        # Consecutive failure tracking — bail out after N failures
        self._consecutive_fails = 0
        self._max_consecutive_fails = 3
        self._error_log: list[str] = []

        # ── Spin/stuck detection ──────────────────────────────────────────────
        # Ring buffer of (time, x, y) samples in map frame
        self._pos_history: collections.deque = collections.deque(maxlen=100)
        self._stuck_reported = False

        # ── TF2 buffer ────────────────────────────────────────────────────────
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── Nav2 action client ────────────────────────────────────────────────
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # ── cmd_vel publisher (emergency stop only) ───────────────────────────
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── Map subscriber (transient_local so we get the last published map) ─
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(OccupancyGrid, '/map', self._on_map, map_qos)

        # ── Odometry subscriber (robot pose fallback) ─────────────────────────
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

        # ── Position sampling timer (for spin detection) ──────────────────────
        self.create_timer(1.0 / _STUCK_CHECK_HZ, self._sample_position)

        self.get_logger().info(
            f'Frontier Explorer ready | freq={self._freq:.2f} Hz | '
            f'min_size={self._min_size_m} m | '
            f'potential_scale={self._pot_scale} | gain_scale={self._gain_scale} | '
            f'startup_delay={self._startup_delay:.0f} s')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_map(self, msg: OccupancyGrid):
        self._map = msg

    def _on_odom(self, msg: Odometry):
        self._robot_x_odom = msg.pose.pose.position.x
        self._robot_y_odom = msg.pose.pose.position.y

    def _on_resume(self, msg: Bool):
        self._enabled = msg.data
        self.get_logger().info(
            f'Exploration {"resumed" if self._enabled else "paused"}')

    # ── TF-based position in MAP frame ────────────────────────────────────────

    def _update_robot_pose_map(self):
        """Look up robot position in map frame via TF. Falls back to odom."""
        try:
            t: TransformStamped = self._tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
            self._robot_x = t.transform.translation.x
            self._robot_y = t.transform.translation.y
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            # Fallback to odom (at startup, map→odom may not exist yet)
            self._robot_x = self._robot_x_odom
            self._robot_y = self._robot_y_odom

    # ── Spin/stuck detection ──────────────────────────────────────────────────

    def _sample_position(self):
        """Sample current position for spin detection."""
        if not self._nav2_ready or not self._navigating:
            self._pos_history.clear()
            return

        self._update_robot_pose_map()
        now = time.monotonic()
        self._pos_history.append((now, self._robot_x, self._robot_y))

        # Check if stuck (barely moving over the window)
        if len(self._pos_history) < 5:
            return

        oldest_t, oldest_x, oldest_y = self._pos_history[0]
        window = now - oldest_t
        if window < _STUCK_WINDOW:
            return  # Not enough history yet

        # Calculate total displacement over the window
        displacement = math.hypot(
            self._robot_x - oldest_x,
            self._robot_y - oldest_y)

        if displacement < _STUCK_THRESHOLD and not self._stuck_reported:
            self._stuck_reported = True
            msg = (f'STUCK/SPINNING detected: moved only {displacement:.3f} m '
                   f'in {window:.1f} s while navigating to {self._current_goal}')
            self.get_logger().error(msg)
            self._error_log.append(f'[STUCK] {msg}')

            # Cancel the current goal and blacklist it
            self._blacklist.append(self._current_goal)
            self._goal_gen += 1
            if self._goal_handle is not None:
                self._goal_handle.cancel_goal_async()
            self._navigating = False
            self._last_fail_t = time.monotonic()
            self._pos_history.clear()

            # Count this as a consecutive failure
            self._consecutive_fails += 1
            if self._consecutive_fails >= self._max_consecutive_fails:
                self._fatal_shutdown()

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
                self._goal_gen += 1  # Invalidate stale callbacks
                if self._goal_handle is not None:
                    self._goal_handle.cancel_goal_async()
                self._navigating = False
                self._last_fail_t = time.monotonic()
            else:
                return

        # Update position in map frame before planning
        self._update_robot_pose_map()

        # ── Log map stats for diagnostics ─────────────────────────────────
        self._log_map_stats()

        frontiers = self._find_frontiers()
        if not frontiers:
            self._no_frontier_count += 1
            if self._no_frontier_count % 10 == 1:
                self.get_logger().info(
                    f'No frontiers found (check #{self._no_frontier_count}) — '
                    'area appears fully mapped')
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
        This handles ray tracing, where occupied cells (walls) always
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
        near_unknown = set()
        for row in range(h):
            for col in range(w):
                if (get_cell(row - 1, col) == UNKNOWN or
                    get_cell(row + 1, col) == UNKNOWN or
                    get_cell(row, col - 1) == UNKNOWN or
                    get_cell(row, col + 1) == UNKNOWN):
                    near_unknown.add((row, col))

        # A frontier cell is a FREE cell that is:
        #   - in near_unknown (directly adjacent to unknown), OR
        #   - adjacent to a cell in near_unknown (2 cells from unknown)
        frontier_mask = [False] * (w * h)
        for row in range(h):
            for col in range(w):
                i = idx(row, col)
                if data[i] != FREE:
                    continue

                if (row, col) in near_unknown:
                    frontier_mask[i] = True
                    continue

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
        """Score and return the best frontier goal.

        The navigation target is pulled 0.4 m toward the robot from the
        frontier centroid so it's always in explored space.
        """
        best_score = -float('inf')
        best_xy = None

        for cluster in clusters:
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            dist = math.hypot(cx - self._robot_x, cy - self._robot_y)
            if dist < 0.3:
                continue

            # Cap maximum frontier distance
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

        if best_xy is None:
            return None

        # Pull the goal 0.4 m toward the robot so it's always inside
        # the explored (and therefore costmap-covered) region.
        fx, fy = best_xy
        dx = fx - self._robot_x
        dy = fy - self._robot_y
        d = math.hypot(dx, dy)
        pull_back = 0.4  # meters
        if d > pull_back + 0.3:
            fx -= (dx / d) * pull_back
            fy -= (dy / d) * pull_back

        return (fx, fy)

    # ── Nav2 goal sending ─────────────────────────────────────────────────────

    def _send_goal(self, x: float, y: float):
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('Nav2 action server not available yet')
            return

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        # Use Time(0) = "latest available transform" — avoids TF
        # extrapolation errors at startup.
        goal.pose.header.stamp = TimeMsg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        # IDENTITY quaternion — no heading preference.
        # The mecanum robot doesn't need to face any specific direction.
        # RotateToGoal/GoalAlign critics have been removed from DWB.
        goal.pose.pose.orientation.z = 0.0
        goal.pose.pose.orientation.w = 1.0

        self._navigating = True
        self._goal_sent_t = time.monotonic()
        self._current_goal = (x, y)
        self._goal_gen += 1
        gen = self._goal_gen  # Capture for closure
        self._stuck_reported = False  # Reset for new goal
        self._pos_history.clear()

        future = self._nav_client.send_goal_async(goal)
        future.add_done_callback(lambda f: self._on_goal_response(f, gen))

    def _on_goal_response(self, future, gen):
        if gen != self._goal_gen:
            return  # Stale callback from a preempted goal
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            msg = f'Goal REJECTED by Nav2 at {self._current_goal}'
            self.get_logger().warn(msg)
            self._blacklist.append(self._current_goal)
            self._navigating = False
            self._record_failure(msg)
            self._last_fail_t = time.monotonic()
            return
        self.get_logger().info('Goal accepted — navigating...')
        result_future = self._goal_handle.get_result_async()
        result_future.add_done_callback(lambda f: self._on_result(f, gen))

    def _on_result(self, future, gen):
        if gen != self._goal_gen:
            return  # Stale callback from a preempted goal — ignore
        self._navigating = False
        status = future.result().status
        if status == 4:
            self.get_logger().info('Navigation succeeded — replanning for next frontier')
            self._consecutive_fails = 0  # Reset on success
        elif status == 6:
            msg = 'Navigation cancelled (e.g. by recovery behaviour)'
            self.get_logger().warn(msg)
            self._record_failure(msg)
            self._last_fail_t = time.monotonic()
        elif status == 5:
            msg = f'Navigation ABORTED (status=5) to goal {self._current_goal}'
            self.get_logger().warn(msg)
            self._blacklist.append(self._current_goal)
            self._record_failure(msg)
            self._last_fail_t = time.monotonic()
        else:
            msg = f'Navigation ended with unexpected status={status}'
            self.get_logger().warn(msg)
            self._record_failure(msg)
            self._last_fail_t = time.monotonic()

    def _record_failure(self, msg: str):
        """Track consecutive failures and shut down if threshold exceeded."""
        self._consecutive_fails += 1
        self._error_log.append(
            f'[FAIL #{self._consecutive_fails}] {msg}')

        if self._consecutive_fails >= self._max_consecutive_fails:
            self._fatal_shutdown()

    def _fatal_shutdown(self):
        """Stop the robot, print errors clearly, and exit the node."""
        # Stop all motors immediately
        twist = Twist()
        self._cmd_pub.publish(twist)
        self._cmd_pub.publish(twist)  # Publish twice for reliability

        # Cancel any active navigation
        if self._goal_handle is not None:
            try:
                self._goal_handle.cancel_goal_async()
            except Exception:
                pass

        # Print errors clearly for easy copying
        border = '=' * 70
        lines = [
            '',
            '',
            '',
            border,
            'FRONTIER EXPLORER — FATAL: Too many consecutive failures',
            border,
            '',
            f'Consecutive failures: {self._consecutive_fails}',
            f'Robot position (map frame): ({self._robot_x:.2f}, {self._robot_y:.2f})',
            f'Last goal sent: {self._current_goal}',
            f'Blacklisted goals: {len(self._blacklist)}',
            '',
            'Error log:',
        ]
        for entry in self._error_log:
            lines.append(f'  {entry}')
        lines.append('')
        lines.append('Possible causes:')
        lines.append('  1. STUCK/SPINNING  — DWB controller choosing rotation over translation')
        lines.append('  2. Planner failure — goal is unreachable (blocked by inflation or walls)')
        lines.append('  3. TF error        — slam_toolbox map→odom transform missing or stale')
        lines.append('  4. Controller fail — no valid trajectories (all paths hit obstacles)')
        lines.append('  5. Hardware issue   — LiDAR or Arduino disconnected')
        lines.append('')
        lines.append('To debug:')
        lines.append('  • Check the full log above for [ERROR] and [WARN] lines')
        lines.append('  • Look for "No valid trajectories" — means inflation too aggressive')
        lines.append('  • Look for "Failed to make progress" — means robot barely moving')
        lines.append('  • Look for "Extrapolation" — means TF timing issue')
        lines.append(border)
        lines.append('')

        error_output = '\n'.join(lines)
        self.get_logger().fatal(error_output)

        # Also print to stdout so it's visible even if logger is noisy
        print(error_output, flush=True)

        # Shut down
        raise SystemExit(1)


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
