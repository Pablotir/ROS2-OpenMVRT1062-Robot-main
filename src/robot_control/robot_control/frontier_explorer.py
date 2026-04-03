#!/usr/bin/env python3
"""
frontier_explorer.py — SLAM-guided direct LiDAR exploration
============================================================
The only node that publishes /cmd_vel.

Two loops run in parallel:
  1. Frontier planner  (2 Hz)  — reads /map, finds the best unexplored
     direction, sets self._goal as a heading angle (in robot frame).
  2. Control loop      (10 Hz) — drives toward that heading while using
     live /scan to repel away from nearby obstacles in real time.

No Nav2. No costmaps. No action clients. No global planner.

Startup gate
------------
The node will NOT publish any velocity until:
  - A /scan has been received
  - A /map has been received
  - The map→base_link TF is available (SLAM has initialised)
  - startup_delay seconds have elapsed

Motion model (mecanum)
----------------------
  linear.x  = forward/back
  linear.y  = strafe left/right  (mecanum only)
  angular.z = turn

All three axes are used simultaneously, so the robot can slide around
obstacles rather than stopping and spinning.
"""

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

import tf2_ros

FREE    = 0
UNKNOWN = -1


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('move_speed',          0.20)   # m/s forward
        self.declare_parameter('turn_speed',          0.55)   # rad/s
        self.declare_parameter('obstacle_distance',   0.35)   # m — start repelling
        self.declare_parameter('emergency_stop_dist', 0.10)   # m — full stop
        self.declare_parameter('planner_frequency',   1.0)    # Hz
        self.declare_parameter('min_frontier_size',   0.25)   # m
        self.declare_parameter('startup_delay',       8.0)    # s

        self._move_spd   = self.get_parameter('move_speed').value
        self._turn_spd   = self.get_parameter('turn_speed').value
        self._obs_dist   = self.get_parameter('obstacle_distance').value
        self._estop_dist = self.get_parameter('emergency_stop_dist').value
        self._plan_freq  = self.get_parameter('planner_frequency').value
        self._min_size_m = self.get_parameter('min_frontier_size').value
        self._startup    = self.get_parameter('startup_delay').value

        # ── State ─────────────────────────────────────────────────────────────
        self._map:  OccupancyGrid | None = None
        self._scan: LaserScan     | None = None

        # Robot pose in map frame — set only when TF succeeds
        self._pos_x  = None   # None = TF not yet available
        self._pos_y  = None
        self._pos_yaw = 0.0

        # Current goal heading in ROBOT-LOCAL frame (radians)
        # 0 = straight ahead, +π/2 = left, -π/2 = right
        self._goal_heading: float | None = None

        self._start_time  = time.monotonic()
        self._ready       = False   # True once startup conditions all met
        self._log_tick    = 0

        # ── TF ────────────────────────────────────────────────────────────────
        self._tf_buf = tf2_ros.Buffer()
        self._tf_lis = tf2_ros.TransformListener(self._tf_buf, self)

        # ── Publisher ─────────────────────────────────────────────────────────
        self._cmd = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── Subscribers ───────────────────────────────────────────────────────
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(OccupancyGrid, '/map', self._cb_map, map_qos)

        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(LaserScan, '/scan', self._cb_scan, scan_qos)

        # ── Timers ────────────────────────────────────────────────────────────
        self.create_timer(0.1, self._control_loop)                         # 10 Hz
        self.create_timer(1.0 / max(self._plan_freq, 0.1), self._plan)    # planner

        self.get_logger().info(
            f'FrontierExplorer ready | '
            f'speed={self._move_spd} m/s | turn={self._turn_spd} rad/s | '
            f'obs={self._obs_dist} m | estop={self._estop_dist} m | '
            f'startup_delay={self._startup:.0f} s')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_map(self, msg):  self._map  = msg
    def _cb_scan(self, msg): self._scan = msg

    # ── TF lookup ─────────────────────────────────────────────────────────────

    def _update_tf(self) -> bool:
        """Update robot pose from TF. Returns True if succeeded."""
        try:
            t = self._tf_buf.lookup_transform('map', 'base_link',
                                               rclpy.time.Time())
            self._pos_x = t.transform.translation.x
            self._pos_y = t.transform.translation.y
            q = t.transform.rotation
            self._pos_yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            return True
        except Exception:
            return False

    # ── Readiness check ────────────────────────────────────────────────────────

    def _check_ready(self) -> bool:
        if self._ready:
            return True
        if self._scan is None:
            return False
        if self._map is None:
            return False
        if not self._update_tf():
            return False
        if time.monotonic() - self._start_time < self._startup:
            return False
        self._ready = True
        self.get_logger().info(
            f'All sensors ready — starting exploration | '
            f'robot at ({self._pos_x:.2f}, {self._pos_y:.2f})')
        return True

    # ── Frontier planner ──────────────────────────────────────────────────────

    def _plan(self):
        if not self._check_ready():
            return
        if not self._update_tf():
            return   # Lost TF — don't update goal, keep last heading

        clusters = self._find_frontiers()
        if not clusters:
            self._log_tick += 1
            if self._log_tick % 10 == 0:
                self.get_logger().info('No frontiers found — area may be fully mapped')
            return

        best = self._pick_frontier(clusters)
        if best is None:
            return

        gx, gy = best
        # Convert goal world position → robot-local heading angle
        dx = gx - self._pos_x
        dy = gy - self._pos_y
        goal_angle_world = math.atan2(dy, dx)
        self._goal_heading = math.atan2(
            math.sin(goal_angle_world - self._pos_yaw),
            math.cos(goal_angle_world - self._pos_yaw))

        dist = math.hypot(dx, dy)
        self.get_logger().info(
            f'Frontier → ({gx:.2f}, {gy:.2f}) | '
            f'dist={dist:.2f} m | heading={math.degrees(self._goal_heading):.0f}° local | '
            f'{len(clusters)} clusters')

    def _find_frontiers(self) -> list[list[tuple[float, float]]]:
        m   = self._map
        w, h = m.info.width, m.info.height
        res  = m.info.resolution
        ox   = m.info.origin.position.x
        oy   = m.info.origin.position.y
        data = m.data

        def get(r, c):
            if r < 0 or r >= h or c < 0 or c >= w:
                return UNKNOWN
            return data[r * w + c]

        def world(r, c):
            return ox + (c + 0.5) * res, oy + (r + 0.5) * res

        # Frontier: FREE cell adjacent to UNKNOWN
        frontier = [False] * (w * h)
        for row in range(h):
            for col in range(w):
                if get(row, col) != FREE:
                    continue
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    if get(row+dr, col+dc) == UNKNOWN:
                        frontier[row * w + col] = True
                        break

        # BFS cluster
        visited = [False] * (w * h)
        clusters = []
        min_cells = max(1, int(self._min_size_m / res))

        for start in range(w * h):
            if not frontier[start] or visited[start]:
                continue
            pts, queue = [], [start]
            visited[start] = True
            while queue:
                cur = queue.pop()
                r, c = divmod(cur, w)
                pts.append(world(r, c))
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    nr, nc = r+dr, c+dc
                    ni = nr * w + nc
                    if 0 <= nr < h and 0 <= nc < w and frontier[ni] and not visited[ni]:
                        visited[ni] = True
                        queue.append(ni)
            if len(pts) >= min_cells:
                clusters.append(pts)

        return clusters

    def _pick_frontier(self, clusters) -> tuple[float, float] | None:
        """Score each frontier cluster; return best world (x, y)."""
        best_score = -float('inf')
        best_pos   = None

        for cluster in clusters:
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            dist = math.hypot(cx - self._pos_x, cy - self._pos_y)

            if dist < 0.3 or dist > 8.0:
                continue

            # Heading to this frontier in robot-local frame
            goal_angle_world = math.atan2(cy - self._pos_y, cx - self._pos_x)
            heading_local = math.atan2(
                math.sin(goal_angle_world - self._pos_yaw),
                math.cos(goal_angle_world - self._pos_yaw))

            # LiDAR clearance in that direction
            clearance = self._lidar_clearance(heading_local)

            size  = len(cluster)
            score = size * 1.0 - dist * 2.0

            if clearance < self._obs_dist:
                score -= 300.0   # heavy penalty if blocked
            else:
                score += min(clearance, 3.0) * 3.0   # reward open paths

            if score > best_score:
                best_score = score
                best_pos   = (cx, cy)

        return best_pos

    def _lidar_clearance(self, heading_local: float) -> float:
        """Minimum LiDAR range in ±30° cone around heading_local (robot frame)."""
        scan = self._scan
        if scan is None:
            return float('inf')
        cone = math.radians(30.0)
        min_r = float('inf')
        for i, r in enumerate(scan.ranges):
            if not (scan.range_min <= r <= scan.range_max) or math.isnan(r):
                continue
            ray = scan.angle_min + i * scan.angle_increment
            diff = abs(math.atan2(math.sin(ray - heading_local),
                                   math.cos(ray - heading_local)))
            if diff <= cone and r < min_r:
                min_r = r
        return min_r

    # ── 10 Hz control loop ────────────────────────────────────────────────────

    def _control_loop(self):
        if not self._ready:
            return

        scan = self._scan
        if scan is None:
            return

        twist = Twist()

        # ── Compute obstacle repulsion from raw LiDAR ─────────────────────
        repulse_x = 0.0   # in robot frame: +x = forward
        repulse_y = 0.0   # +y = left
        closest_front = float('inf')
        closest_any   = float('inf')

        for i, r in enumerate(scan.ranges):
            if not (scan.range_min <= r <= scan.range_max) or math.isnan(r) or math.isinf(r):
                continue

            ray = scan.angle_min + i * scan.angle_increment
            norm = math.atan2(math.sin(ray), math.cos(ray))  # [-π, π]

            if r < closest_any:
                closest_any = r
            if abs(norm) < math.radians(90) and r < closest_front:
                closest_front = r

            # Repulse only from obstacles closer than obs_dist
            if r < self._obs_dist:
                strength = ((self._obs_dist - r) / self._obs_dist) ** 2
                repulse_x -= strength * math.cos(ray)
                repulse_y -= strength * math.sin(ray)

        # ── Emergency stop ────────────────────────────────────────────────
        if closest_any < self._estop_dist:
            self._cmd.publish(Twist())
            if self._log_tick % 10 == 0:
                self.get_logger().warn(
                    f'EMERGENCY STOP — obstacle at {closest_any:.2f} m')
            self._log_tick += 1
            return

        # ── Attraction toward frontier goal ───────────────────────────────
        if self._goal_heading is not None:
            # Unit vector in goal direction
            attract_x = math.cos(self._goal_heading)
            attract_y = math.sin(self._goal_heading)
        else:
            # No frontier yet — drive straight forward
            attract_x = 1.0
            attract_y = 0.0

        # ── Combine (repulsion weighted 2× attraction) ────────────────────
        nav_x = attract_x + repulse_x * 2.0
        nav_y = attract_y + repulse_y * 2.0

        # Don't drive forward if something is close ahead
        if closest_front < self._obs_dist and nav_x > 0:
            nav_x = 0.0

        # Scale to configured speeds
        fwd    = max(-self._move_spd, min(self._move_spd, nav_x * self._move_spd))
        strafe = max(-self._move_spd * 0.6, min(self._move_spd * 0.6,
                      nav_y * self._move_spd))

        # Steer: turn to face the combined nav direction
        desired_heading = math.atan2(nav_y, nav_x) if (abs(nav_x) + abs(nav_y)) > 0.01 else 0.0
        turn = max(-self._turn_spd, min(self._turn_spd, desired_heading * 1.5))

        twist.linear.x  = fwd
        twist.linear.y  = strafe
        twist.angular.z = turn
        self._cmd.publish(twist)

        # ── Status log every 2 s ──────────────────────────────────────────
        self._log_tick += 1
        if self._log_tick % 20 == 0:
            goal_deg = (f'{math.degrees(self._goal_heading):.0f}°'
                        if self._goal_heading is not None else 'none')
            pos_str  = (f'({self._pos_x:.2f}, {self._pos_y:.2f})'
                        if self._pos_x is not None else 'no-TF')
            self.get_logger().info(
                f'pos={pos_str} | '
                f'goal_heading={goal_deg} | '
                f'front={closest_front:.2f} m | '
                f'cmd=({fwd:.2f} fwd, {strafe:.2f} str, {turn:.2f} turn)')


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._cmd.publish(Twist())   # stop motors on exit
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
