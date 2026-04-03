#!/usr/bin/env python3
"""
exploration_controller.py — Wall-follow + frontier bias exploration
====================================================================
Architecture (matches what slam_toolbox demos actually do):

  ALWAYS MOVING: the robot never stops to rotate in place.
  It continuously drives forward while blending two steering signals:

  1. WALL-FOLLOW BIAS  (LiDAR, 10 Hz)
     Keep a comfortable gap from the nearest wall.
     If a wall is too close on the right → steer left.
     If too close on the left → steer right.
     This naturally traces room boundaries.

  2. FRONTIER GOAL BIAS  (/map, every 5s)
     Find the largest unvisited frontier cluster.
     Add a yaw bias toward it that nudges the robot out of loops
     WITHOUT overriding the wall-follow safety.

  3. STRAFE REPULSION  (LiDAR, 10 Hz)
     If anything is inside robot_radius → strafe away from it
     (mecanum advantage — no stopping needed).

  STUCK DETECTION (translational only, 15s):
     If the robot hasn't moved 0.4 m in 15 s it marks current goal
     visited and picks the next frontier. Turning does NOT count as stuck.

Why no turn-then-drive?
  Turn-then-drive + short stuck timeout = infinite spin loop.
  The wall-follow + goal-bias approach means the robot rotates
  WHILE moving, exactly like the slam_toolbox gif.

Subscribes: /scan, /map
Publishes:  /cmd_vel  (linear.x + linear.y + angular.z)
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

# ── Tuning constants ──────────────────────────────────────────────────────────
N_SECTORS      = 24            # 360 / 24 = 15° per sector
FREE           = 0
UNKNOWN        = -1

# Wall-follow: target distance to keep from side walls
WALL_TARGET     = 0.55         # m — desired side clearance
WALL_DIST_GAIN  = 0.7          # distance correction gain
WALL_ALIGN_GAIN = 1.0          # alignment gain
WALL_SECTOR_L   = N_SECTORS // 4        # ~90° left
WALL_SECTOR_R   = 3 * N_SECTORS // 4   # ~270° = 90° right

# Noise filtering
ALIGN_EMA       = 0.20         # exponential moving average weight (lower = smoother)
ALIGN_DEADBAND  = 0.04         # m — ignore alignment differences smaller than this

# Goal bias weight — nudge only, wall-follow stays dominant
GOAL_BIAS_W     = 0.18         # rad, clamped

# Hallway mode: when both walls visible + front clear, center + run fast
HALLWAY_MAX_SIDE = 1.4         # m — both L+R within this = hallway
HALLWAY_MIN_FRONT = 0.80       # m — front must be clear
HALLWAY_SPEED    = 0.52        # m/s top speed in hallway
HALLWAY_CENTER_GAIN = 0.6      # centering gain (L-R error → turn correction)

# Normal speed (open room)
NORMAL_SPEED     = 0.25        # m/s


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
        self._latest_scan   = None
        self._scan_received = False
        self._last_scan_t   = None
        self._map           = None
        self._start_t       = time.monotonic()
        self._log_tick      = 0

        # Robot pose (from TF)
        self._robot_x   = 0.0
        self._robot_y   = 0.0
        self._robot_yaw = 0.0
        self._has_tf    = False

        # Frontier goal
        self._goal_world    = None    # (x, y) map frame
        self._goal_heading  = None    # robot-local radians, updated every control tick
        self._visited       = []      # blacklisted goals
        self._visited_rad   = 0.8     # m — how close = "reached"

        # Stuck detection — updated every 100ms in control loop (NOT here)
        self._last_pos_x    = 0.0
        self._last_pos_y    = 0.0
        self._last_move_t   = time.monotonic()
        self._stuck_timeout = 20.0    # s — generous; position updates at 10 Hz now

        # Noise filtering: EMA of alignment corrections
        self._smooth_align  = 0.0     # exponential moving average of align_error
        self._smooth_l      = None    # EMA of left clearance
        self._smooth_r      = None    # EMA of right clearance

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
        self.create_timer(0.1,  self._control_loop)    # 10 Hz driving
        self.create_timer(5.0,  self._plan_frontier)   # 0.2 Hz goal update

        self.get_logger().info(
            f'Exploration controller (wall-follow+frontier) ready | '
            f'speed={self._move_spd} | turn={self._turn_spd} | '
            f'obs={self._obs_dist} | estop={self._estop_dist} | '
            f'radius={self._robot_rad}')

    # ── Callbacks ─────────────────────────────────────────────────────────────

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
    # FRONTIER PLANNER (every 5 s) — just sets a goal direction
    # ═══════════════════════════════════════════════════════════════════════════

    def _plan_frontier(self):
        if self._map is None:
            return

        self._update_tf()
        now = time.monotonic()

        # Check goal reached
        if self._goal_world is not None and self._has_tf:
            gx, gy = self._goal_world
            if math.hypot(gx - self._robot_x, gy - self._robot_y) < 0.6:
                self.get_logger().info(
                    f'REACHED goal ({gx:.1f},{gy:.1f}) — marking visited')
                self._visited.append((gx, gy))
                self._goal_world = None
                self._goal_heading = None

        # Check stuck — position tracking now happens in control_loop (10 Hz)
        # Just read the result here
        if self._has_tf and self._goal_world:
            now = time.monotonic()
            if (now - self._last_move_t) > self._stuck_timeout:
                gx, gy = self._goal_world
                self.get_logger().warn(
                    f'STUCK (no translation for {self._stuck_timeout:.0f}s) — '
                    f'abandoning ({gx:.1f},{gy:.1f})')
                self._visited.append((gx, gy))
                self._goal_world = None
                self._goal_heading = None
                self._last_move_t = now

        # Pick new frontier if we don't have a goal
        if self._goal_world is None:
            clusters = self._find_frontiers()
            if clusters:
                best = self._pick_frontier(clusters)
                if best is not None:
                    gx, gy = best
                    self._goal_world = (gx, gy)
                    self._last_pos_x = self._robot_x
                    self._last_pos_y = self._robot_y
                    self._last_move_t = now
                    if self._has_tf:
                        h = self._heading_to(gx, gy)
                        self.get_logger().info(
                            f'NEW GOAL → ({gx:.2f},{gy:.2f}) | '
                            f'dist={math.hypot(gx-self._robot_x,gy-self._robot_y):.2f} m | '
                            f'heading={math.degrees(h):+.0f}° | '
                            f'{len(clusters)} clusters | {len(self._visited)} visited')
            else:
                if self._log_tick % 5 == 0:
                    self.get_logger().info('No frontiers — area fully mapped?')

    def _heading_to(self, gx, gy) -> float:
        dx = gx - self._robot_x
        dy = gy - self._robot_y
        world_a = math.atan2(dy, dx)
        return math.atan2(math.sin(world_a - self._robot_yaw),
                          math.cos(world_a - self._robot_yaw))

    def _is_visited(self, x, y) -> bool:
        return any(math.hypot(x - vx, y - vy) < self._visited_rad
                   for vx, vy in self._visited)

    def _find_frontiers(self):
        m = self._map
        w, h   = m.info.width, m.info.height
        res    = m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y
        data   = m.data

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

        visited_cells = set()
        clusters = []
        min_cells = max(1, int(0.20 / res))

        for start in frontier:
            if start in visited_cells:
                continue
            pts, queue = [], [start]
            visited_cells.add(start)
            while queue:
                cur = queue.pop()
                r, c = divmod(cur, w)
                pts.append(world(r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    ni = nr * w + nc
                    if (0 <= nr < h and 0 <= nc < w
                            and ni in frontier and ni not in visited_cells):
                        visited_cells.add(ni)
                        queue.append(ni)
            if len(pts) >= min_cells:
                clusters.append(pts)

        return clusters

    def _pick_frontier(self, clusters):
        """
        Pick the best frontier cluster.
        Scoring: large clusters, moderate distance, forward bias, open LiDAR path.
        Rear frontiers ARE allowed — the wall-follow will navigate around obstacles
        naturally. We just add a mild forward preference (+3 pts if within ±90°).
        """
        best_score = -float('inf')
        best_pos   = None

        for cluster in clusters:
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            dist = math.hypot(cx - self._robot_x, cy - self._robot_y)

            if dist < 0.4 or dist > 15.0:
                continue
            if self._is_visited(cx, cy):
                continue

            size    = len(cluster)
            heading = self._heading_to(cx, cy) if self._has_tf else 0.0
            clear   = self._lidar_clearance(heading)

            # Base score: large and not too far away
            score = size * 1.5 - dist * 0.3

            # Mild forward preference (doesn't hard-block rear frontiers)
            if abs(heading) < math.radians(90):
                score += 3.0

            # Prefer open LiDAR paths
            if clear < self._obs_dist:
                score -= 50.0   # blocked → very low priority
            else:
                score += min(clear, 4.0) * 1.5

            if score > best_score:
                best_score = score
                best_pos   = (cx, cy)

        return best_pos

    def _lidar_clearance(self, heading_local):
        scan = self._latest_scan
        if scan is None:
            return float('inf')
        cone   = math.radians(30.0)
        min_r  = float('inf')
        for i, r in enumerate(scan.ranges):
            if not (scan.range_min <= r <= scan.range_max) or math.isnan(r):
                continue
            ray  = scan.angle_min + i * scan.angle_increment
            diff = abs(math.atan2(math.sin(ray - heading_local),
                                   math.cos(ray - heading_local)))
            if diff <= cone and r < min_r:
                min_r = r
        return min_r

    # ═══════════════════════════════════════════════════════════════════════════
    # REACTIVE DRIVER (10 Hz) — always moving, never turns in place
    # ═══════════════════════════════════════════════════════════════════════════

    def _control_loop(self):
        now = time.monotonic()

        # ── Wait for LiDAR ────────────────────────────────────────────────
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

        # ── Build sector map ──────────────────────────────────────────────
        sector_min  = [float('inf')] * N_SECTORS
        sector_ang  = 2.0 * math.pi / N_SECTORS
        closest_any = float('inf')
        strafe_force = 0.0

        for i, r in enumerate(scan.ranges):
            if not (scan.range_min <= r <= scan.range_max):
                continue
            if math.isnan(r) or math.isinf(r):
                continue

            angle    = scan.angle_min + i * scan.angle_increment
            angle_p  = angle % (2.0 * math.pi)
            sidx     = int(angle_p / sector_ang) % N_SECTORS
            if r < sector_min[sidx]:
                sector_min[sidx] = r
            if r < closest_any:
                closest_any = r

            # Lateral repulsion when obstacle enters body zone
            repulse_zone = self._robot_rad + 0.10
            if r < repulse_zone:
                na = math.atan2(math.sin(angle), math.cos(angle))
                strafe_force -= math.sin(na) * ((repulse_zone - r) / repulse_zone)

        # ── Emergency stop ────────────────────────────────────────────────
        if closest_any < self._estop_dist:
            self._cmd_pub.publish(Twist())
            self._log_tick += 1
            if self._log_tick % 5 == 0:
                self.get_logger().warn(
                    f'EMERGENCY STOP — {closest_any:.2f} m')
            return

        # ── Wall-follow: noise-filtered alignment + hallway mode ────────────────
        # N_SECTORS=24 → sector_ang=15°
        # Right sectors: 18=270°(-90°), 20=300°(-60°), 16=240°(-120°)
        # Left sectors:   6=90°,         4=60°,         8=120°

        # Distance samples  (raw, perpendicular)
        left_sectors  = [(WALL_SECTOR_L + d) % N_SECTORS for d in range(-2, 3)]
        right_sectors = [(WALL_SECTOR_R + d) % N_SECTORS for d in range(-2, 3)]
        left_raw      = min(sector_min[s] for s in left_sectors)
        right_raw     = min(sector_min[s] for s in right_sectors)

        # EMA smooth the side distances to kill noise
        if self._smooth_l is None:
            self._smooth_l, self._smooth_r = left_raw, right_raw
        else:
            self._smooth_l = ALIGN_EMA * left_raw  + (1 - ALIGN_EMA) * self._smooth_l
            self._smooth_r = ALIGN_EMA * right_raw + (1 - ALIGN_EMA) * self._smooth_r
        left_clear  = self._smooth_l
        right_clear = self._smooth_r

        # Alignment via ±60° vs ±120° two-point projection
        SIN60 = 0.866
        r_front_right = sector_min[20 % N_SECTORS]
        r_rear_right  = sector_min[16 % N_SECTORS]
        r_front_left  = sector_min[ 4 % N_SECTORS]
        r_rear_left   = sector_min[ 8 % N_SECTORS]

        raw_align = 0.0
        ALIGN_MIN_RANGE = 1.4
        if (right_clear < ALIGN_MIN_RANGE
                and r_front_right < ALIGN_MIN_RANGE
                and r_rear_right  < ALIGN_MIN_RANGE):
            diff_r = r_rear_right - r_front_right
            if abs(diff_r) > ALIGN_DEADBAND / SIN60:
                raw_align += diff_r * SIN60 * WALL_ALIGN_GAIN

        if (left_clear < ALIGN_MIN_RANGE
                and r_front_left < ALIGN_MIN_RANGE
                and r_rear_left  < ALIGN_MIN_RANGE):
            diff_l = r_front_left - r_rear_left
            if abs(diff_l) > ALIGN_DEADBAND / SIN60:
                raw_align -= diff_l * SIN60 * WALL_ALIGN_GAIN

        # EMA smooth the alignment signal
        self._smooth_align = (ALIGN_EMA * raw_align
                              + (1 - ALIGN_EMA) * self._smooth_align)
        align_error = max(-0.25, min(0.25, self._smooth_align))

        # ── Detect hallway mode ────────────────────────────────────────────
        front_sectors = [0, 1, N_SECTORS - 1]
        front_clear   = min(sector_min[s] for s in front_sectors)

        in_hallway = (left_clear  < HALLWAY_MAX_SIDE
                      and right_clear < HALLWAY_MAX_SIDE
                      and front_clear > HALLWAY_MIN_FRONT)

        if in_hallway:
            # CENTER between the two walls: target L == R
            # Positive center_err = robot is right of center = turn left
            center_err  = (left_clear - right_clear) * HALLWAY_CENTER_GAIN
            # Combine centering + alignment (both smoothed)
            wall_error  = center_err + align_error
            target_speed = HALLWAY_SPEED
        else:
            # Normal wall-follow: independent distance correction on each wall
            dist_error = 0.0
            if right_clear < WALL_TARGET:
                dist_error += (WALL_TARGET - right_clear) * WALL_DIST_GAIN
            if left_clear < WALL_TARGET:
                dist_error -= (WALL_TARGET - left_clear) * WALL_DIST_GAIN
            wall_error   = dist_error + align_error
            target_speed = NORMAL_SPEED

        # ── Obstacle avoidance: pick best open sector ─────────────────────
        # For the forward hemisphere, find the sector with most clearance
        fwd_scores = {}
        for s in range(N_SECTORS):
            clearance = min(sector_min[s], 3.0)
            prev_c    = min(sector_min[(s-1) % N_SECTORS], 3.0)
            next_c    = min(sector_min[(s+1) % N_SECTORS], 3.0)
            smooth    = (clearance + prev_c + next_c) / 3.0
            sec_local = math.atan2(math.sin(s * sector_ang),
                                    math.cos(s * sector_ang))
            # Only consider roughly forward sectors (within ±120°)
            if abs(sec_local) > math.radians(120):
                continue
            # Forward preference
            fwd_bias  = 1.0 - (abs(sec_local) / math.radians(120))
            score     = smooth + fwd_bias * 0.5
            if clearance < self._obs_dist:
                score *= 0.05
            fwd_scores[s] = score

        if fwd_scores:
            best_s    = max(fwd_scores, key=lambda s: fwd_scores[s])
            best_ang  = math.atan2(math.sin(best_s * sector_ang),
                                    math.cos(best_s * sector_ang))
        else:
            best_ang  = 0.0

        # ── Goal bias ────────────────────────────────────────────────────
        goal_bias = 0.0
        goal_str  = 'none'
        if self._goal_world is not None and self._has_tf:
            goal_heading = self._heading_to(*self._goal_world)
            # Gentle bias — clamp to ±GOAL_BIAS_W so wall-follow stays dominant
            goal_bias = max(-GOAL_BIAS_W,
                            min(GOAL_BIAS_W, goal_heading * 0.3))
            goal_str = f'({self._goal_world[0]:.1f},{self._goal_world[1]:.1f})'

        # ── Combine steering signals ──────────────────────────────────────
        # Base turn from best open sector
        obstacle_turn = self._turn_spd * (best_ang / math.pi)
        # Wall correction
        wall_turn     = max(-self._turn_spd * 0.6,
                            min(self._turn_spd * 0.6, wall_error))
        # Blend: obstacle avoidance dominates when something is close
        front_sectors = [0, 1, N_SECTORS - 1]
        front_clear   = min(sector_min[s] for s in front_sectors)

        if front_clear < self._obs_dist:
            # Obstacle close → full avoidance steering
            turn = obstacle_turn
        else:
            # Clear path → blend wall-follow + goal bias
            turn = wall_turn + goal_bias

        turn = max(-self._turn_spd, min(self._turn_spd, turn))

        # ── Forward speed ─────────────────────────────────────────────────
        effective_clear = min(front_clear, closest_any + self._robot_rad)

        if effective_clear < self._estop_dist:
            fwd = 0.0
        elif effective_clear < self._obs_dist:
            ratio = ((effective_clear - self._estop_dist) /
                     (self._obs_dist - self._estop_dist))
            fwd = target_speed * max(0.2, ratio)
        else:
            fwd = target_speed

        if closest_any < self._robot_rad:
            fwd = 0.0

        # Log hallway mode
        mode_str = 'HALL' if in_hallway else 'room'

        # ── Update stuck detection at 10 Hz ──────────────────────────────
        if self._has_tf:
            moved = math.hypot(self._robot_x - self._last_pos_x,
                               self._robot_y - self._last_pos_y)
            if moved > 0.25:
                self._last_pos_x = self._robot_x
                self._last_pos_y = self._robot_y
                self._last_move_t = now

        # ── Strafe ────────────────────────────────────────────────────────
        max_str = self._move_spd * 0.6
        strafe  = max(-max_str, min(max_str, strafe_force * self._move_spd))

        # ── Publish ───────────────────────────────────────────────────────
        twist = Twist()
        twist.linear.x  = fwd
        twist.linear.y  = strafe
        twist.angular.z = turn
        self._cmd_pub.publish(twist)

        # ── Log every 2 s ─────────────────────────────────────────────────
        self._log_tick += 1
        if self._log_tick % 20 == 0:
            self.get_logger().info(
                f'[{mode_str}] fwd={fwd:.2f} trn={turn:+.2f} str={strafe:+.2f} | '
                f'front={front_clear:.2f} L={left_clear:.2f} R={right_clear:.2f} | '
                f'align={align_error:+.2f} | goal={goal_str} near={closest_any:.2f}')


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
