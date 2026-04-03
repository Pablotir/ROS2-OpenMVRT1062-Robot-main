#!/usr/bin/env python3
"""
exploration_controller.py — Map-aware smooth reactive exploration
==================================================================
Single node that:
  1. Reads /map (from slam_toolbox) to find FRONTIERS — unexplored areas
  2. Picks the best frontier as a goal (prefer large, unblocked openings)
  3. Drives there using direct LiDAR obstacle avoidance (no Nav2)

The robot actively seeks out doorways, hallways, and open areas because
those show up as frontiers on the SLAM map. It won't just circle inside
a room — it will head toward the boundary of known/unknown space.

Motion control
--------------
  Every 100 ms:
    - Build a sector clearance map from LiDAR (24 sectors, 15° each)
    - Score each sector: clearance * smoothness * goal_bias
    - goal_bias is HIGH for sectors pointing toward the frontier
    - Steer + drive toward the best-scoring sector
    - Strafe away from any close side obstacles (mecanum advantage)

Frontier detection
------------------
  Every 3 seconds:
    - Scan the /map occupancy grid for free cells adjacent to unknown cells
    - Cluster them into groups
    - Pick the largest reachable cluster as the goal
    - Compute heading from robot to goal centroid

Subscribes
----------
  /scan   sensor_msgs/LaserScan     360° LiDAR for obstacle avoidance
  /map    nav_msgs/OccupancyGrid    SLAM map for frontier detection

Publishes
---------
  /cmd_vel  geometry_msgs/Twist     linear.x, linear.y (strafe), angular.z
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

N_SECTORS    = 24    # 360° / 24 = 15° per sector
FREE         = 0
UNKNOWN      = -1


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
        self._scan_received = False
        self._latest_scan   = None
        self._map           = None
        self._start_t       = self.get_clock().now().nanoseconds * 1e-9
        self._last_scan_t   = None
        self._log_tick      = 0

        # Robot pose in map frame (from TF)
        self._robot_x   = 0.0
        self._robot_y   = 0.0
        self._robot_yaw = 0.0
        self._has_tf    = False

        # Frontier goal: heading in ROBOT-LOCAL frame (radians)
        # None = no frontier found yet, just wander
        self._goal_heading = None
        self._goal_world   = None   # (x, y) for logging

        # ── TF ────────────────────────────────────────────────────────────────
        self._tf_buf = tf2_ros.Buffer()
        self._tf_lis = tf2_ros.TransformListener(self._tf_buf, self)

        # ── Publisher ─────────────────────────────────────────────────────────
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── Subscribers ───────────────────────────────────────────────────────
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(LaserScan, '/scan', self._on_scan, scan_qos)

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(OccupancyGrid, '/map', self._on_map, map_qos)

        # ── Timers ────────────────────────────────────────────────────────────
        self.create_timer(0.1, self._control_loop)    # 10 Hz — driving
        self.create_timer(3.0, self._plan_frontier)   # 0.33 Hz — frontier search

        self.get_logger().info(
            f'Exploration controller (map-aware reactive) ready | '
            f'speed={self._move_spd} m/s | turn={self._turn_spd} rad/s | '
            f'obstacle={self._obs_dist} m | estop={self._estop_dist} m | '
            f'robot_radius={self._robot_rad} m')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_scan(self, msg):
        self._latest_scan = msg
        if not self._scan_received:
            self.get_logger().info(f'First LiDAR scan: {len(msg.ranges)} rays')
            self._scan_received = True
        self._last_scan_t = self.get_clock().now().nanoseconds * 1e-9

    def _on_map(self, msg):
        self._map = msg

    # ── TF lookup ─────────────────────────────────────────────────────────────

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

    # ── Frontier planner (runs every 3 s) ─────────────────────────────────────

    def _plan_frontier(self):
        if self._map is None or not self._has_tf:
            return

        self._update_tf()
        clusters = self._find_frontiers()

        if not clusters:
            self._goal_heading = None
            self._goal_world = None
            return

        # Pick the best frontier
        best = self._pick_frontier(clusters)
        if best is None:
            self._goal_heading = None
            self._goal_world = None
            return

        gx, gy = best
        dx = gx - self._robot_x
        dy = gy - self._robot_y
        dist = math.hypot(dx, dy)

        # Compute goal heading in robot-local frame
        world_angle = math.atan2(dy, dx)
        local_heading = math.atan2(
            math.sin(world_angle - self._robot_yaw),
            math.cos(world_angle - self._robot_yaw))

        # Only log if goal changed significantly
        if (self._goal_world is None or
                math.hypot(gx - self._goal_world[0], gy - self._goal_world[1]) > 0.5):
            self.get_logger().info(
                f'FRONTIER GOAL → ({gx:.2f}, {gy:.2f}) | '
                f'dist={dist:.2f} m | heading={math.degrees(local_heading):+.0f}° | '
                f'{len(clusters)} clusters')

        self._goal_heading = local_heading
        self._goal_world = (gx, gy)

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

        # Frontier: FREE cell adjacent to UNKNOWN
        frontier = [False] * (w * h)
        for row in range(h):
            for col in range(w):
                if get(row, col) != FREE:
                    continue
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if get(row + dr, col + dc) == UNKNOWN:
                        frontier[row * w + col] = True
                        break

        # BFS cluster
        visited = [False] * (w * h)
        clusters = []
        min_cells = max(1, int(0.25 / res))

        for start in range(w * h):
            if not frontier[start] or visited[start]:
                continue
            pts, queue = [], [start]
            visited[start] = True
            while queue:
                cur = queue.pop()
                r, c = divmod(cur, w)
                pts.append(world(r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    ni = nr * w + nc
                    if 0 <= nr < h and 0 <= nc < w and frontier[ni] and not visited[ni]:
                        visited[ni] = True
                        queue.append(ni)
            if len(pts) >= min_cells:
                clusters.append(pts)

        return clusters

    def _pick_frontier(self, clusters):
        """Pick the best frontier. Prefer large clusters in open directions."""
        best_score = -float('inf')
        best_pos = None

        for cluster in clusters:
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            dist = math.hypot(cx - self._robot_x, cy - self._robot_y)

            if dist < 0.3 or dist > 12.0:
                continue

            size = len(cluster)

            # Heading toward this frontier
            angle_world = math.atan2(cy - self._robot_y, cx - self._robot_x)
            heading_local = math.atan2(
                math.sin(angle_world - self._robot_yaw),
                math.cos(angle_world - self._robot_yaw))

            # LiDAR clearance in that direction
            clearance = self._lidar_clearance(heading_local)

            # Score: prefer large, reachable, unblocked frontiers
            score = size * 2.0 - dist * 1.0

            if clearance < self._obs_dist:
                score -= 200.0   # penalize blocked directions
            else:
                score += min(clearance, 3.0) * 3.0

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

    # ── 10 Hz control loop ────────────────────────────────────────────────────

    def _control_loop(self):
        now = self.get_clock().now().nanoseconds * 1e-9

        if not self._scan_received:
            self._cmd_pub.publish(Twist())
            self._log_tick += 1
            if self._log_tick % 10 == 0:
                elapsed = now - self._start_t
                if elapsed > self._timeout:
                    self.get_logger().warn(f'NO LIDAR after {elapsed:.0f} s')
                else:
                    self.get_logger().info(
                        f'Waiting for LiDAR ({elapsed:.1f}/{self._timeout:.0f} s)')
            return

        if self._last_scan_t and (now - self._last_scan_t) > 2.0:
            self._cmd_pub.publish(Twist())
            self._log_tick += 1
            if self._log_tick % 10 == 0:
                self.get_logger().warn('LiDAR stale — halted')
            return

        # Update pose for goal heading recalculation
        self._update_tf()

        scan = self._latest_scan

        # ── Build sector clearance map + lateral repulsion ────────────────
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

            # Lateral repulsion for close obstacles
            repulse_zone = self._robot_rad + 0.10
            if r < repulse_zone:
                norm_angle = math.atan2(math.sin(angle), math.cos(angle))
                lateral = math.sin(norm_angle)
                strength = (repulse_zone - r) / repulse_zone
                strafe_force -= lateral * strength

        # ── Emergency stop ────────────────────────────────────────────────
        if closest_any < self._estop_dist:
            self._cmd_pub.publish(Twist())
            self._log_tick += 1
            if self._log_tick % 10 == 0:
                self.get_logger().warn(
                    f'EMERGENCY STOP — obstacle at {closest_any:.2f} m')
            return

        # ── Score sectors ─────────────────────────────────────────────────
        # Recalculate goal heading from current pose (TF may have updated)
        goal_heading = self._goal_heading
        if self._has_tf and self._goal_world is not None:
            dx = self._goal_world[0] - self._robot_x
            dy = self._goal_world[1] - self._robot_y
            world_angle = math.atan2(dy, dx)
            goal_heading = math.atan2(
                math.sin(world_angle - self._robot_yaw),
                math.cos(world_angle - self._robot_yaw))

        scores = [0.0] * N_SECTORS
        for s in range(N_SECTORS):
            clearance = min(sector_min[s], 3.0)
            prev_c = min(sector_min[(s - 1) % N_SECTORS], 3.0)
            next_c = min(sector_min[(s + 1) % N_SECTORS], 3.0)
            smooth = (clearance + prev_c + next_c) / 3.0

            sector_center = s * sector_angle
            sector_local = math.atan2(math.sin(sector_center),
                                       math.cos(sector_center))

            # Goal bias: how close is this sector to the frontier goal?
            if goal_heading is not None:
                angle_diff = abs(math.atan2(
                    math.sin(sector_local - goal_heading),
                    math.cos(sector_local - goal_heading)))
                # 1.0 when aligned with goal, 0.0 when opposite
                goal_bias = 1.0 - (angle_diff / math.pi)
            else:
                # No goal — bias toward forward
                angle_from_fwd = abs(sector_local)
                goal_bias = 1.0 - (angle_from_fwd / math.pi)

            # Score = clearance * goal alignment
            # The goal bias is weighted heavily (3.0) so the robot
            # actively seeks the frontier instead of just following walls
            scores[s] = smooth + goal_bias * 3.0

            # Hard penalize blocked directions
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
            ratio = (effective_clear - self._estop_dist) / (self._obs_dist - self._estop_dist)
            fwd = self._move_spd * ratio * 0.5
        else:
            fwd = self._move_spd

        if closest_any < self._robot_rad:
            fwd = 0.0

        turn_urgency = abs(best_angle) / math.pi
        if turn_urgency > 0.4:
            fwd *= max(0.0, 1.0 - turn_urgency)

        # ── Strafe ────────────────────────────────────────────────────────
        max_strafe = self._move_spd * 0.6
        strafe = max(-max_strafe, min(max_strafe, strafe_force * self._move_spd))

        # ── Turn ──────────────────────────────────────────────────────────
        turn = self._turn_spd * (best_angle / math.pi)

        if front_clear < self._obs_dist:
            if abs(best_angle) > 0.1:
                turn = self._turn_spd * (1.0 if best_angle > 0 else -1.0) * 0.8
            else:
                left_clear = min(sector_min[s] for s in range(N_SECTORS // 6, N_SECTORS // 3))
                right_clear = min(sector_min[s] for s in range(2 * N_SECTORS // 3, 5 * N_SECTORS // 6))
                turn = self._turn_spd * 0.8 * (1.0 if left_clear > right_clear else -1.0)

        # ── Publish ───────────────────────────────────────────────────────
        twist = Twist()
        twist.linear.x = fwd
        twist.linear.y = strafe
        twist.angular.z = turn
        self._cmd_pub.publish(twist)

        # ── Log every 2 s ─────────────────────────────────────────────────
        self._log_tick += 1
        if self._log_tick % 20 == 0:
            best_deg = math.degrees(best_angle)
            goal_str = (f'({self._goal_world[0]:.1f},{self._goal_world[1]:.1f})'
                        if self._goal_world else 'none')
            self.get_logger().info(
                f'front={front_clear:.2f} m | '
                f'goal={goal_str} | '
                f'best={best_deg:+.0f}° | '
                f'cmd=({fwd:.2f} fwd, {strafe:+.2f} str, {turn:+.2f} trn) | '
                f'closest={closest_any:.2f} m')


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
