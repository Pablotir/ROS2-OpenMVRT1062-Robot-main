#!/usr/bin/env python3
"""
exploration_controller.py — Map-aware smooth reactive exploration
==================================================================
Single node. Two jobs:
  1. FRONTIER PLANNER (every 3s): reads /map → picks a goal → sticks with it
  2. REACTIVE DRIVER  (10 Hz):   reads /scan → drives toward goal → avoids obstacles

Key design decisions
--------------------
- Goal persistence: once a frontier is chosen, the robot commits to it
  for at least 15 seconds. No flip-flopping between two frontiers.
- Turn-then-drive: if the goal is behind the robot (>90°), the robot
  turns in place at full speed until the goal is within ±60° of forward,
  THEN starts driving. No more 0.02 m/s crawl.
- Visited tracking: reached frontiers are marked so the robot doesn't
  circle back to them.
- Stuck detection: if the robot makes no progress for 20 seconds,
  it picks a new frontier.

Subscribes: /scan, /map
Publishes:  /cmd_vel (linear.x + linear.y + angular.z)
"""

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import tf2_ros

N_SECTORS = 24
FREE      = 0
UNKNOWN   = -1


class ExplorationController(Node):
    def __init__(self):
        super().__init__('exploration_controller')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('move_speed',           0.18)
        self.declare_parameter('turn_speed',           0.50)
        self.declare_parameter('obstacle_distance',    0.50)
        self.declare_parameter('emergency_stop_dist',  0.20)
        self.declare_parameter('robot_radius',         0.27)
        self.declare_parameter('sensor_timeout',       8.0)

        self._move_spd   = self.get_parameter('move_speed').value
        self._turn_spd   = self.get_parameter('turn_speed').value
        self._obs_dist   = self.get_parameter('obstacle_distance').value
        self._estop_dist = self.get_parameter('emergency_stop_dist').value
        self._robot_rad  = self.get_parameter('robot_radius').value
        self._timeout    = self.get_parameter('sensor_timeout').value

        # ── State ─────────────────────────────────────────────────────────────
        self._latest_scan = None
        self._scan_received = False
        self._last_scan_t   = None
        self._map           = None
        self._start_t       = time.monotonic()
        self._log_tick      = 0

        # Robot pose
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_yaw = 0.0
        self._has_tf = False

        # Goal management
        self._goal_world = None      # (x, y) in map frame
        self._goal_set_time = 0.0    # when the goal was last set
        self._goal_min_hold = 15.0   # seconds — minimum time to keep a goal
        self._visited = []           # list of (x, y) visited goals
        self._visited_radius = 0.8   # how close = "visited"

        # Stuck detection
        self._last_progress_x = 0.0
        self._last_progress_y = 0.0
        self._last_progress_t = 0.0
        self._stuck_timeout   = 3.0   # seconds with no progress → new goal

        # ── TF ────────────────────────────────────────────────────────────────
        self._tf_buf = tf2_ros.Buffer()
        self._tf_lis = tf2_ros.TransformListener(self._tf_buf, self)

        # ── Pub/Sub ───────────────────────────────────────────────────────────
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        scan_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(LaserScan, '/scan', self._on_scan, scan_qos)

        map_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(OccupancyGrid, '/map', self._on_map, map_qos)

        # ── Timers ────────────────────────────────────────────────────────────
        self.create_timer(0.1, self._control_loop)
        self.create_timer(3.0, self._plan_frontier)

        self.get_logger().info(
            f'Exploration controller ready | '
            f'speed={self._move_spd} | turn={self._turn_spd} | '
            f'obs={self._obs_dist} | estop={self._estop_dist} | '
            f'radius={self._robot_rad}')

    def _on_scan(self, msg):
        self._latest_scan = msg
        if not self._scan_received:
            self.get_logger().info(f'First LiDAR scan: {len(msg.ranges)} rays')
            self._scan_received = True
        self._last_scan_t = time.monotonic()

    def _on_map(self, msg):
        self._map = msg

    def _update_tf(self) -> bool:
        try:
            t = self._tf_buf.lookup_transform('map', 'base_link',
                                               rclpy.time.Time())
            self._robot_x = t.transform.translation.x
            self._robot_y = t.transform.translation.y
            q = t.transform.rotation
            self._robot_yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self._has_tf = True
            return True
        except Exception:
            return False

    # ═══════════════════════════════════════════════════════════════════════════
    # FRONTIER PLANNER (every 3 s)
    # ═══════════════════════════════════════════════════════════════════════════

    def _plan_frontier(self):
        if self._map is None or not self._has_tf:
            return
        self._update_tf()
        now = time.monotonic()

        # ── Check if current goal is still valid ──────────────────────────
        if self._goal_world is not None:
            gx, gy = self._goal_world
            dist = math.hypot(gx - self._robot_x, gy - self._robot_y)

            # Reached the goal?
            if dist < 0.5:
                self.get_logger().info(
                    f'REACHED goal ({gx:.1f}, {gy:.1f}) — marking visited')
                self._visited.append((gx, gy))
                self._goal_world = None
            # Still holding? Don't re-plan yet
            elif now - self._goal_set_time < self._goal_min_hold:
                return
            # Stuck? Force re-plan
            elif self._is_stuck(now):
                self.get_logger().warn(
                    f'STUCK — no progress for {self._stuck_timeout:.0f}s, '
                    f'abandoning goal ({gx:.1f}, {gy:.1f})')
                self._visited.append((gx, gy))  # Don't go back
                self._goal_world = None
            else:
                return  # Keep current goal

        # ── Find new frontier ─────────────────────────────────────────────
        clusters = self._find_frontiers()
        if not clusters:
            self._goal_world = None
            self._log_tick += 1
            if self._log_tick % 5 == 0:
                self.get_logger().info('No frontiers — area may be fully mapped')
            return

        best = self._pick_frontier(clusters)
        if best is None:
            self._goal_world = None
            return

        gx, gy = best
        dist = math.hypot(gx - self._robot_x, gy - self._robot_y)
        self._goal_world = (gx, gy)
        self._goal_set_time = now
        self._last_progress_x = self._robot_x
        self._last_progress_y = self._robot_y
        self._last_progress_t = now

        heading = self._heading_to(gx, gy)
        self.get_logger().info(
            f'NEW GOAL → ({gx:.2f}, {gy:.2f}) | '
            f'dist={dist:.2f} m | heading={math.degrees(heading):+.0f}° | '
            f'{len(clusters)} clusters | {len(self._visited)} visited')

    def _is_stuck(self, now: float) -> bool:
        """Check if robot has moved significantly since last progress."""
        moved = math.hypot(self._robot_x - self._last_progress_x,
                           self._robot_y - self._last_progress_y)
        if moved > 0.3:
            self._last_progress_x = self._robot_x
            self._last_progress_y = self._robot_y
            self._last_progress_t = now
            return False
        return (now - self._last_progress_t) > self._stuck_timeout

    def _heading_to(self, gx, gy) -> float:
        """Heading from robot to (gx, gy) in robot-local frame."""
        dx = gx - self._robot_x
        dy = gy - self._robot_y
        world_angle = math.atan2(dy, dx)
        return math.atan2(
            math.sin(world_angle - self._robot_yaw),
            math.cos(world_angle - self._robot_yaw))

    def _is_visited(self, x, y) -> bool:
        for vx, vy in self._visited:
            if math.hypot(x - vx, y - vy) < self._visited_radius:
                return True
        return False

    def _find_frontiers(self):
        m = self._map
        w, h = m.info.width, m.info.height
        res = m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y
        data = m.data

        def get(r, c):
            if r < 0 or r >= h or c < 0 or c >= w:
                return UNKNOWN
            return data[r * w + c]

        def world(r, c):
            return ox + (c + 0.5) * res, oy + (r + 0.5) * res

        frontier = set()
        for row in range(h):
            for col in range(w):
                if get(row, col) != FREE:
                    continue
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if get(row + dr, col + dc) == UNKNOWN:
                        frontier.add(row * w + col)
                        break

        visited = set()
        clusters = []
        min_cells = max(1, int(0.25 / res))

        for start in frontier:
            if start in visited:
                continue
            pts, queue = [], [start]
            visited.add(start)
            while queue:
                cur = queue.pop()
                r, c = divmod(cur, w)
                pts.append(world(r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    ni = nr * w + nc
                    if 0 <= nr < h and 0 <= nc < w and ni in frontier and ni not in visited:
                        visited.add(ni)
                        queue.append(ni)
            if len(pts) >= min_cells:
                clusters.append(pts)

        return clusters

    def _pick_frontier(self, clusters):
        best_score = -float('inf')
        best_pos = None

        for cluster in clusters:
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            dist = math.hypot(cx - self._robot_x, cy - self._robot_y)

            if dist < 0.4 or dist > 12.0:
                continue

            # Skip visited frontiers
            if self._is_visited(cx, cy):
                continue

            size = len(cluster)
            heading = self._heading_to(cx, cy)
            clearance = self._lidar_clearance(heading)

            # Score: prefer large, close, forward-ish, unblocked
            score = size * 2.0 - dist * 0.5

            # Prefer frontiers roughly ahead (within ±90°)
            if abs(heading) < math.radians(90):
                score += 5.0
            # Don't penalize rear frontiers too much — they're still valid
            # but prefer forward ones if available

            if clearance < self._obs_dist:
                score -= 100.0
            else:
                score += min(clearance, 3.0) * 2.0

            if score > best_score:
                best_score = score
                best_pos = (cx, cy)

        return best_pos

    def _lidar_clearance(self, heading_local):
        scan = self._latest_scan
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

    # ═══════════════════════════════════════════════════════════════════════════
    # REACTIVE DRIVER (10 Hz)
    # ═══════════════════════════════════════════════════════════════════════════

    def _control_loop(self):
        now = time.monotonic()

        if not self._scan_received:
            self._cmd_pub.publish(Twist())
            self._log_tick += 1
            if self._log_tick % 10 == 0:
                elapsed = now - self._start_t
                self.get_logger().info(
                    f'Waiting for LiDAR ({elapsed:.1f}/{self._timeout:.0f} s)')
            return

        if self._last_scan_t and (now - self._last_scan_t) > 2.0:
            self._cmd_pub.publish(Twist())
            return

        self._update_tf()
        scan = self._latest_scan

        # ── Scan processing ───────────────────────────────────────────────
        sector_min = [float('inf')] * N_SECTORS
        sector_angle = 2.0 * math.pi / N_SECTORS
        closest_any = float('inf')
        strafe_force = 0.0

        for i, r in enumerate(scan.ranges):
            if not (scan.range_min <= r <= scan.range_max):
                continue
            if math.isnan(r) or math.isinf(r):
                continue

            angle = scan.angle_min + i * scan.angle_increment
            angle_pos = angle % (2.0 * math.pi)
            sector_idx = int(angle_pos / sector_angle) % N_SECTORS
            if r < sector_min[sector_idx]:
                sector_min[sector_idx] = r
            if r < closest_any:
                closest_any = r

            # Lateral repulsion
            repulse_zone = self._robot_rad + 0.10
            if r < repulse_zone:
                norm_angle = math.atan2(math.sin(angle), math.cos(angle))
                strafe_force -= math.sin(norm_angle) * ((repulse_zone - r) / repulse_zone)

        # ── Emergency stop ────────────────────────────────────────────────
        if closest_any < self._estop_dist:
            self._cmd_pub.publish(Twist())
            self._log_tick += 1
            if self._log_tick % 10 == 0:
                self.get_logger().warn(
                    f'EMERGENCY STOP — obstacle at {closest_any:.2f} m')
            return

        # ── Compute goal heading (live from TF) ──────────────────────────
        goal_heading = None
        if self._goal_world is not None and self._has_tf:
            goal_heading = self._heading_to(*self._goal_world)

        # ── TURN-THEN-DRIVE: if goal is behind us, turn in place first ───
        if goal_heading is not None and abs(goal_heading) > math.radians(75):
            # Goal is far to the side/behind — turn in place
            turn_dir = 1.0 if goal_heading > 0 else -1.0
            twist = Twist()
            twist.angular.z = self._turn_spd * turn_dir * 0.9

            # Allow slight strafe for safety
            max_strafe = self._move_spd * 0.6
            twist.linear.y = max(-max_strafe, min(max_strafe,
                                  strafe_force * self._move_spd))

            self._cmd_pub.publish(twist)

            self._log_tick += 1
            if self._log_tick % 20 == 0:
                self.get_logger().info(
                    f'TURNING to face goal | '
                    f'heading={math.degrees(goal_heading):+.0f}° | '
                    f'cmd=(0.00 fwd, {twist.linear.y:+.2f} str, '
                    f'{twist.angular.z:+.2f} trn)')
            return

        # ── Score sectors (goal is roughly ahead or no goal) ──────────────
        scores = [0.0] * N_SECTORS
        for s in range(N_SECTORS):
            clearance = min(sector_min[s], 3.0)
            prev_c = min(sector_min[(s - 1) % N_SECTORS], 3.0)
            next_c = min(sector_min[(s + 1) % N_SECTORS], 3.0)
            smooth = (clearance + prev_c + next_c) / 3.0

            sector_center = s * sector_angle
            sector_local = math.atan2(math.sin(sector_center),
                                       math.cos(sector_center))

            if goal_heading is not None:
                angle_diff = abs(math.atan2(
                    math.sin(sector_local - goal_heading),
                    math.cos(sector_local - goal_heading)))
                goal_bias = 1.0 - (angle_diff / math.pi)
            else:
                # No goal — bias forward
                goal_bias = 1.0 - (abs(sector_local) / math.pi)

            scores[s] = smooth + goal_bias * 3.0

            if clearance < self._obs_dist:
                scores[s] *= 0.1

        best_sector = max(range(N_SECTORS), key=lambda s: scores[s])
        best_angle = best_sector * sector_angle
        best_angle = math.atan2(math.sin(best_angle), math.cos(best_angle))

        # ── Forward speed ─────────────────────────────────────────────────
        front_sectors = [0, 1, N_SECTORS - 1]
        front_clear = min(sector_min[s] for s in front_sectors)
        effective_clear = min(front_clear, closest_any + self._robot_rad)

        if effective_clear < self._estop_dist:
            fwd = 0.0
        elif effective_clear < self._obs_dist:
            ratio = (effective_clear - self._estop_dist) / \
                    (self._obs_dist - self._estop_dist)
            fwd = self._move_spd * ratio * 0.5
        else:
            fwd = self._move_spd

        if closest_any < self._robot_rad:
            fwd = 0.0

        # Slow for moderate turns (but NOT for tiny corrections)
        turn_urgency = abs(best_angle) / math.pi
        if turn_urgency > 0.5:
            fwd *= max(0.2, 1.0 - turn_urgency)

        # ── Strafe ────────────────────────────────────────────────────────
        max_strafe = self._move_spd * 0.6
        strafe = max(-max_strafe, min(max_strafe,
                      strafe_force * self._move_spd))

        # ── Turn ──────────────────────────────────────────────────────────
        turn = self._turn_spd * (best_angle / math.pi)

        if front_clear < self._obs_dist and abs(best_angle) > 0.1:
            turn = self._turn_spd * (1.0 if best_angle > 0 else -1.0) * 0.8

        # ── Publish ───────────────────────────────────────────────────────
        twist = Twist()
        twist.linear.x = fwd
        twist.linear.y = strafe
        twist.angular.z = turn
        self._cmd_pub.publish(twist)

        # ── Log ───────────────────────────────────────────────────────────
        self._log_tick += 1
        if self._log_tick % 20 == 0:
            best_deg = math.degrees(best_angle)
            goal_str = (f'({self._goal_world[0]:.1f},{self._goal_world[1]:.1f})'
                        if self._goal_world else 'none')
            self.get_logger().info(
                f'front={front_clear:.2f} | '
                f'goal={goal_str} | '
                f'best={best_deg:+.0f}° | '
                f'cmd=({fwd:.2f} fwd, {strafe:+.2f} str, {turn:+.2f} trn) | '
                f'near={closest_any:.2f}')


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
