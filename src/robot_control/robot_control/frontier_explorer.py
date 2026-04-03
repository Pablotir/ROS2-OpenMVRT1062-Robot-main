#!/usr/bin/env python3
"""
frontier_explorer.py — Direct LiDAR frontier exploration (NO Nav2)
===================================================================
Combines frontier detection (from the SLAM map) with direct LiDAR-based
obstacle avoidance. No costmaps, no DWB, no global planner, no behavior
trees. Just: find unexplored space → drive toward it → don't hit anything.

How it works
------------
1. Read the SLAM map (/map) to find frontier clusters (free space near
   unknown space).
2. Score each frontier by size (prefer large) and direction clearance
   (prefer directions with no LiDAR obstacles).
3. Compute the desired heading toward the best frontier.
4. Drive the robot using a simple potential-field-like controller:
   - Translate toward the goal at `move_speed`
   - Steer away from any close LiDAR obstacles
   - Emergency stop if something is within `emergency_stop_dist`
   - The mecanum drive allows simultaneous strafe + turn, so the robot
     can slide around obstacles naturally.

Why not Nav2?
-------------
Nav2 (costmaps + DWB planner + NavFn + behavior trees) is designed for
navigating in a KNOWN map. For exploration, where the map is being built
in real-time, it creates more problems than it solves:
  - Costmap inflation fights with the expanding SLAM frontiers
  - DWB critics designed for diff-drive cause mecanum robots to spin
  - "No valid trajectories" when the planner can't find a clean path
  - Recovery behaviors (spin, backup) conflict with exploration goals
Direct LiDAR control is simpler, faster, and more reliable for this task.

Subscribes
----------
  /map    nav_msgs/OccupancyGrid   SLAM map from slam_toolbox
  /scan   sensor_msgs/LaserScan    360° LiDAR for obstacle avoidance
  /odom   nav_msgs/Odometry        Robot pose (fallback if no TF)

Publishes
---------
  /cmd_vel   geometry_msgs/Twist   Velocity commands (mecanum strafe supported)

Parameters
----------
  move_speed          float  0.22   m/s  — forward speed
  turn_speed          float  0.6    rad/s — max turn rate
  obstacle_distance   float  0.40   m    — start steering away at this range
  emergency_stop_dist float  0.12   m    — full stop, back up
  planner_frequency   float  2.0    Hz   — how often to re-evaluate frontiers
  min_frontier_size   float  0.30   m    — ignore tiny frontier clusters
  startup_delay       float  8.0    s    — wait for SLAM to produce first map
"""

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

import tf2_ros

# ── Occupancy values ──────────────────────────────────────────────────────────
FREE    = 0
UNKNOWN = -1


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('move_speed',          0.22)
        self.declare_parameter('turn_speed',          0.6)
        self.declare_parameter('obstacle_distance',   0.40)
        self.declare_parameter('emergency_stop_dist', 0.12)
        self.declare_parameter('planner_frequency',   2.0)
        self.declare_parameter('min_frontier_size',   0.30)
        self.declare_parameter('startup_delay',       8.0)

        self._move_spd   = self.get_parameter('move_speed').value
        self._turn_spd   = self.get_parameter('turn_speed').value
        self._obs_dist   = self.get_parameter('obstacle_distance').value
        self._estop_dist = self.get_parameter('emergency_stop_dist').value
        self._plan_freq  = self.get_parameter('planner_frequency').value
        self._min_size_m = self.get_parameter('min_frontier_size').value
        self._startup_delay = self.get_parameter('startup_delay').value

        # ── State ─────────────────────────────────────────────────────────────
        self._map: OccupancyGrid | None = None
        self._scan: LaserScan | None = None
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_yaw = 0.0
        self._start_time = time.monotonic()
        self._ready = False
        self._enabled = True

        # Current goal in map frame (set by frontier planner)
        self._goal_x = None
        self._goal_y = None
        self._no_frontier_count = 0

        # ── TF2 ──────────────────────────────────────────────────────────────
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── Publishers ────────────────────────────────────────────────────────
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── Subscribers ───────────────────────────────────────────────────────
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(OccupancyGrid, '/map', self._on_map, map_qos)

        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(LaserScan, '/scan', self._on_scan, scan_qos)

        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(Odometry, '/odom', self._on_odom, odom_qos)

        self.create_subscription(Bool, '/explore/resume', self._on_resume, 10)

        # ── Timers ────────────────────────────────────────────────────────────
        # Fast control loop (10 Hz) — obstacle avoidance + driving
        self.create_timer(0.1, self._control_loop)

        # Slower frontier planner
        plan_period = 1.0 / max(self._plan_freq, 0.1)
        self.create_timer(plan_period, self._plan_frontiers)

        self._log_tick = 0

        self.get_logger().info(
            f'Frontier Explorer (NO Nav2) ready | '
            f'speed={self._move_spd} m/s | turn={self._turn_spd} rad/s | '
            f'obstacle={self._obs_dist} m | estop={self._estop_dist} m | '
            f'planner_freq={self._plan_freq} Hz | '
            f'startup_delay={self._startup_delay:.0f} s')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_map(self, msg: OccupancyGrid):
        self._map = msg

    def _on_scan(self, msg: LaserScan):
        self._scan = msg

    def _on_odom(self, msg: Odometry):
        # Fallback position (used before TF is available)
        pass

    def _on_resume(self, msg: Bool):
        self._enabled = msg.data
        self.get_logger().info(
            f'Exploration {"resumed" if self._enabled else "paused"}')
        if not self._enabled:
            self._cmd_pub.publish(Twist())

    # ── TF ────────────────────────────────────────────────────────────────────

    def _update_pose(self):
        try:
            t = self._tf_buffer.lookup_transform('map', 'base_link',
                                                  rclpy.time.Time())
            self._robot_x = t.transform.translation.x
            self._robot_y = t.transform.translation.y
            q = t.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self._robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        except Exception:
            pass  # Keep previous values

    # ── Frontier planner (runs at planner_frequency Hz) ───────────────────────

    def _plan_frontiers(self):
        if not self._enabled or self._map is None or self._scan is None:
            return

        if not self._ready:
            if time.monotonic() - self._start_time < self._startup_delay:
                return
            self._ready = True
            self.get_logger().info('Startup delay elapsed — beginning exploration')

        self._update_pose()

        clusters = self._find_frontiers()
        if not clusters:
            self._no_frontier_count += 1
            if self._no_frontier_count % 20 == 1:
                self.get_logger().info(
                    f'No frontiers (#{self._no_frontier_count}) — '
                    'area may be fully mapped')
            # Keep driving toward last goal or forward
            return
        self._no_frontier_count = 0

        # Score and pick the best frontier
        best = self._pick_frontier(clusters)
        if best is None:
            return

        gx, gy = best
        dist = math.hypot(gx - self._robot_x, gy - self._robot_y)

        # Only update goal if it's substantially different from current
        if (self._goal_x is None or
            math.hypot(gx - self._goal_x, gy - self._goal_y) > 0.3):
            self._goal_x, self._goal_y = gx, gy
            self.get_logger().info(
                f'New frontier goal ({gx:.2f}, {gy:.2f}) | '
                f'dist={dist:.2f} m | {len(clusters)} clusters')

    def _find_frontiers(self) -> list[list[tuple[float, float]]]:
        """Find frontier clusters on the SLAM map."""
        m = self._map
        w, h = m.info.width, m.info.height
        res = m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y
        data = m.data

        def idx(r, c):
            return r * w + c

        def world(r, c):
            return ox + (c + 0.5) * res, oy + (r + 0.5) * res

        def get_cell(r, c):
            if r < 0 or r >= h or c < 0 or c >= w:
                return UNKNOWN
            return data[idx(r, c)]

        # Cells within 1 cell of unknown
        near_unknown = set()
        for row in range(h):
            for col in range(w):
                if (get_cell(row - 1, col) == UNKNOWN or
                    get_cell(row + 1, col) == UNKNOWN or
                    get_cell(row, col - 1) == UNKNOWN or
                    get_cell(row, col + 1) == UNKNOWN):
                    near_unknown.add((row, col))

        # Frontier: FREE cell within 2 cells of UNKNOWN
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

        # BFS cluster
        visited = [False] * (w * h)
        clusters = []
        min_cells = max(1, int(self._min_size_m / res))

        for start in range(w * h):
            if not frontier_mask[start] or visited[start]:
                continue
            pts = []
            queue = [start]
            visited[start] = True
            while queue:
                cur = queue.pop()
                r, c = divmod(cur, w)
                pts.append(world(r, c))
                for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if 0 <= nr < h and 0 <= nc < w:
                        ni = idx(nr, nc)
                        if frontier_mask[ni] and not visited[ni]:
                            visited[ni] = True
                            queue.append(ni)
            if len(pts) >= min_cells:
                clusters.append(pts)

        return clusters

    def _pick_frontier(self, clusters) -> tuple[float, float] | None:
        """Pick the best frontier considering size, distance, and LiDAR clearance."""
        best_score = -float('inf')
        best = None

        for cluster in clusters:
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            dist = math.hypot(cx - self._robot_x, cy - self._robot_y)

            if dist < 0.3 or dist > 8.0:
                continue

            size = len(cluster)
            clearance = self._clearance_toward(cx, cy)

            # Score: prefer large, close, unblocked frontiers
            score = size * 1.0 - dist * 3.0
            if clearance < self._obs_dist:
                score -= 500.0   # Heavily penalize blocked directions
            else:
                score += min(clearance, 3.0) * 2.0  # Bonus for clearance

            if score > best_score:
                best_score = score
                best = (cx, cy)

        return best

    def _clearance_toward(self, gx: float, gy: float) -> float:
        """Minimum LiDAR range in a ±30° cone toward the goal."""
        scan = self._scan
        if scan is None:
            return float('inf')

        dx = gx - self._robot_x
        dy = gy - self._robot_y
        goal_local = math.atan2(dy, dx) - self._robot_yaw
        goal_local = math.atan2(math.sin(goal_local), math.cos(goal_local))

        cone = math.radians(30.0)
        min_r = float('inf')

        for i, r in enumerate(scan.ranges):
            if r < scan.range_min or r > scan.range_max or math.isinf(r) or math.isnan(r):
                continue
            ray = scan.angle_min + i * scan.angle_increment
            diff = abs(math.atan2(math.sin(ray - goal_local),
                                   math.cos(ray - goal_local)))
            if diff <= cone and r < min_r:
                min_r = r

        return min_r

    # ── 10 Hz control loop — direct LiDAR driving ────────────────────────────

    def _control_loop(self):
        if not self._ready or not self._enabled or self._scan is None:
            return

        self._update_pose()
        scan = self._scan
        twist = Twist()

        # ── Build obstacle repulsion from raw LiDAR ──────────────────────
        # Sum up repulsive vectors from all nearby obstacles
        repulse_x = 0.0
        repulse_y = 0.0
        closest_front = float('inf')
        closest_any = float('inf')

        for i, r in enumerate(scan.ranges):
            if r < scan.range_min or r > scan.range_max or math.isinf(r) or math.isnan(r):
                continue

            if r < closest_any:
                closest_any = r

            ray_angle = scan.angle_min + i * scan.angle_increment

            # Check if this is roughly "in front" (±90°)
            norm_angle = math.atan2(math.sin(ray_angle), math.cos(ray_angle))
            if abs(norm_angle) < math.radians(90) and r < closest_front:
                closest_front = r

            # Only repulse from obstacles within obstacle_distance
            if r < self._obs_dist:
                # Repulsion force: inverse-square, direction away from obstacle
                strength = (self._obs_dist - r) / self._obs_dist
                strength = strength ** 2  # Quadratic for stronger close repulsion
                # Obstacle is at (r*cos(angle), r*sin(angle)) in robot frame
                # Push AWAY from it
                repulse_x -= strength * math.cos(ray_angle)
                repulse_y -= strength * math.sin(ray_angle)

        # ── Emergency stop ───────────────────────────────────────────────
        if closest_any < self._estop_dist:
            self._cmd_pub.publish(Twist())  # Full stop
            self._log_tick += 1
            if self._log_tick % 10 == 0:
                self.get_logger().warn(
                    f'EMERGENCY STOP — obstacle at {closest_any:.2f} m')
            return

        # ── Determine desired direction ──────────────────────────────────
        if self._goal_x is not None:
            # Direction to frontier goal in robot-local frame
            dx = self._goal_x - self._robot_x
            dy = self._goal_y - self._robot_y
            goal_dist = math.hypot(dx, dy)
            goal_angle_map = math.atan2(dy, dx)
            goal_angle_local = goal_angle_map - self._robot_yaw
            goal_angle_local = math.atan2(math.sin(goal_angle_local),
                                           math.cos(goal_angle_local))

            if goal_dist < 0.3:
                # Reached the goal, clear it
                self.get_logger().info(
                    f'Reached frontier goal ({self._goal_x:.2f}, {self._goal_y:.2f})')
                self._goal_x = None
                self._goal_y = None
                self._cmd_pub.publish(Twist())
                return

            # Attraction toward goal
            attract_strength = min(goal_dist, 1.0)  # Cap at 1.0
            attract_x = attract_strength * math.cos(goal_angle_local)
            attract_y = attract_strength * math.sin(goal_angle_local)
        else:
            # No frontier goal — just drive forward slowly
            attract_x = 0.3
            attract_y = 0.0

        # ── Combine attraction + repulsion ───────────────────────────────
        # Repulsion is scaled up so the robot STRONGLY avoids obstacles
        repulsion_gain = 3.0
        combined_x = attract_x + repulse_x * repulsion_gain
        combined_y = attract_y + repulse_y * repulsion_gain

        # Convert to velocity commands
        # Forward/backward (clamp to max speed)
        fwd = max(-self._move_spd, min(self._move_spd, combined_x * self._move_spd))
        # Strafe (mecanum)
        strafe = max(-self._move_spd * 0.7, min(self._move_spd * 0.7,
                      combined_y * self._move_spd))
        # Turn toward combined direction
        desired_heading = math.atan2(combined_y, combined_x)
        turn = max(-self._turn_spd, min(self._turn_spd,
                    desired_heading * 1.5))

        # If front is too close, don't go forward at all
        if closest_front < self._obs_dist and fwd > 0:
            fwd = 0.0

        twist.linear.x = fwd
        twist.linear.y = strafe
        twist.angular.z = turn

        self._cmd_pub.publish(twist)

        # ── Periodic logging ─────────────────────────────────────────────
        self._log_tick += 1
        if self._log_tick % 20 == 0:  # Every 2 seconds
            goal_str = (f'({self._goal_x:.2f}, {self._goal_y:.2f})'
                       if self._goal_x is not None else 'none')
            self.get_logger().info(
                f'pos=({self._robot_x:.2f}, {self._robot_y:.2f}) | '
                f'goal={goal_str} | '
                f'front={closest_front:.2f} m | '
                f'vel=({fwd:.2f}, {strafe:.2f}, {turn:.2f})')


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop motors on exit
        node._cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
