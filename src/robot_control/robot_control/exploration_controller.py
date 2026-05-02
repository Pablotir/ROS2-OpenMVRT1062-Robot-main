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
import heapq
import os
import subprocess

# ── Tuning constants ──────────────────────────────────────────────────────────
N_SECTORS      = 24            # 360 / 24 = 15° per sector
FREE           = 0
UNKNOWN        = -1

# Wall-follow: target distance to keep from side walls
WALL_TARGET     = 0.55         # m — desired side clearance
WALL_DIST_GAIN  = 0.35         # distance correction gain (was 0.7 — reduced to keep strafe gentle)
WALL_ALIGN_GAIN = 0.50         # alignment gain (was 1.0 — reduced)
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
HALLWAY_CENTER_GAIN = 0.30     # strafe centering gain (was 0.6 — now drives strafe not turn)
HALLWAY_ALIGN_GAIN  = 1.0      # yaw alignment gain in hallway — same strength as ROOM mode

# Normal speed (open room)
NORMAL_SPEED     = 0.25        # m/s

# Max strafe speed as fraction of move_speed
MAX_STRAFE_FRAC  = 0.50        # 50% of move_speed max
MAX_ALIGN_TURN   = 0.10        # rad/s — absolute cap on alignment yaw corrections

# Return-to-home
NO_FRONTIER_CONFIRM = 3        # consecutive empty frontier cycles before declaring done
HOME_ARRIVAL_DIST   = 0.6      # m — how close to home = "arrived"
MAP_SAVE_DIR        = '/root/ros2_ws/maps'

# Trajectory-based LiDAR coverage blacklisting
TRAJ_RECORD_DIST = 0.8    # m — record a waypoint every 0.8 m of travel
TRAJ_COVERED_RAD = 3.0    # m — suppress frontier clusters this close to any
                           #     recorded position (360° LiDAR covers 10m, 3.0m
                           #     is the conservative occlusion-safe radius)


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
        self._visited_rad   = 1.5     # m — suppress clusters this close to any visited goal
        self._trajectory    = []      # robot positions recorded every TRAJ_RECORD_DIST
        self._last_traj_x   = 0.0
        self._last_traj_y   = 0.0

        # Stuck detection — updated every 100ms in control loop (NOT here)
        self._last_pos_x    = 0.0
        self._last_pos_y    = 0.0
        self._last_move_t   = time.monotonic()
        self._stuck_timeout = 20.0    # s — generous; position updates at 10 Hz now

        # Noise filtering: EMA of alignment corrections
        self._smooth_align  = 0.0     # exponential moving average of align_error
        self._smooth_l      = None    # EMA of left clearance
        self._smooth_r      = None    # EMA of right clearance

        # G14: FSM States Definition
        self.STATE_CROSSING = 'CROSSING'
        self.STATE_HALLWAY = 'HALLWAY'
        self.STATE_ROOM_PERIMETER = 'ROOM'
        self._current_state = self.STATE_CROSSING
        self._prev_state   = self.STATE_CROSSING   # track transitions to reset EMA
        self._hugging_side = 'RIGHT'

        # Corner escape: count consecutive front-blocked cycles
        self._corner_count   = 0          # cycles with front < obs_dist
        self._corner_turning = False      # in forced hard-turn phase
        self._corner_dir     = 1.0        # +1 = left, -1 = right
        # Odometry-based corner turn exit: record yaw at start, exit when
        # robot has rotated >= corner_target_rad (instead of ticking time)
        self._corner_start_yaw   = None   # robot yaw when hard-turn began
        self._corner_target_rad  = math.radians(40.0)  # rotate 40° then stop

        # Return-to-home
        self._home_x = 0.0
        self._home_y = 0.0
        self._home_recorded = False
        self._no_frontier_count = 0
        self._returning_home = False
        self._exploration_complete = False
        self._home_path   = []        # A* waypoints [(x,y), ...]
        self._home_wp_idx = 0         # current waypoint index

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
        self.create_timer(2.0,  self._plan_frontier)   # 0.5 Hz goal update (was 5s — now reacts faster)

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
        first_map = (self._map is None)
        self._map = msg
        # Trigger frontier planning immediately on the first map so the robot
        # has a goal before it can reach any T-junction and corner-turn into
        # already-scanned areas like cubby alcoves.
        if first_map and self._has_tf:
            self._plan_frontier()

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
        if self._exploration_complete:
            return

        self._update_tf()
        now = time.monotonic()

        # ── Returning home: check arrival, skip frontier replanning ───────
        if self._returning_home:
            if self._has_tf:
                dist_home = math.hypot(
                    self._home_x - self._robot_x,
                    self._home_y - self._robot_y)
                if dist_home < HOME_ARRIVAL_DIST:
                    self.get_logger().info(
                        f'HOME REACHED ({self._robot_x:.1f},{self._robot_y:.1f})'
                        f' — saving map and stopping')
                    self._save_map()
                    self._exploration_complete = True
                else:
                    self.get_logger().info(
                        f'RETURNING HOME | dist={dist_home:.1f} m | '
                        f'pos=({self._robot_x:.1f},{self._robot_y:.1f}) | '
                        f'home=({self._home_x:.1f},{self._home_y:.1f})')
            return

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
                self._no_frontier_count = 0   # reset — frontiers still exist
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
                # No frontiers — confirm over multiple cycles before declaring done
                self._no_frontier_count += 1
                if (self._no_frontier_count >= NO_FRONTIER_CONFIRM
                        and self._home_recorded):
                    self._returning_home = True
                    self._goal_world = (self._home_x, self._home_y)
                    self._plan_path_home()
                    self.get_logger().info(
                        f'EXPLORATION COMPLETE — returning home '
                        f'({self._home_x:.1f},{self._home_y:.1f}) | '
                        f'{len(self._home_path)} waypoints')
                else:
                    self.get_logger().info(
                        f'No frontiers ({self._no_frontier_count}/'
                        f'{NO_FRONTIER_CONFIRM}) — confirming...')

    def _heading_to(self, gx, gy) -> float:
        dx = gx - self._robot_x
        dy = gy - self._robot_y
        world_a = math.atan2(dy, dx)
        return math.atan2(math.sin(world_a - self._robot_yaw),
                          math.cos(world_a - self._robot_yaw))

    def _is_visited(self, x, y) -> bool:
        # Goal blacklist: explicit reached-goal suppression
        if any(math.hypot(x - vx, y - vy) < self._visited_rad
               for vx, vy in self._visited):
            return True
        # Trajectory coverage: every recorded robot position represents a full
        # 360° LiDAR scan at 10m range.  Frontier clusters within 2.5m of any
        # trajectory point were definitively scanned and need not be revisited.
        if any(math.hypot(x - tx, y - ty) < TRAJ_COVERED_RAD
               for tx, ty in self._trajectory):
            return True
        return False

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
        # 0.35m (7 cells at 5cm res): filters single-cell noise while still
        # qualifying doorways (~0.8m wide = 16 cells) and the main hallway.
        # The cubby won't generate a large cluster because its walls are fully
        # visible from the junction — all cells become FREE, not frontier.
        min_cells = max(1, int(0.35 / res))

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
            score = size * 1.5 - dist * 1.5

            # Bonus for moving forward, heavy penalty for going backward
            local_ang = heading
            fwd_score = 5.0 * math.cos(local_ang)
            if abs(local_ang) > math.pi / 2:
                fwd_score -= 25.0

            # Right-hand rule: proportional bonus for rightward frontiers,
            # proportional penalty for leftward frontiers.
            # local_ang: negative = right, positive = left (ROS convention)
            right_bonus = 20.0 * max(0.0, -local_ang / math.pi)  # 0..+20 for right
            left_penalty = 10.0 * max(0.0,  local_ang / math.pi)  # 0..+10 for left

            score = (size * 1.5) - (dist * 1.5) + fwd_score + right_bonus - left_penalty

            # Prefer open LiDAR paths — penalty kept modest so perpendicular
            # hallways (LiDAR sees corner wall, not the hallway beyond) don't
            # score worse than backward targets with open corridors behind them.
            if clear < self._obs_dist:
                score -= 20.0   # blocked: lower priority but not disqualifying
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

        # ── Exploration complete: full stop ────────────────────────────────
        if self._exploration_complete:
            self._cmd_pub.publish(Twist())
            return

        self._update_tf()

        # Record home position from first valid TF
        if self._has_tf and not self._home_recorded:
            self._home_x = self._robot_x
            self._home_y = self._robot_y
            self._home_recorded = True
            # Seed trajectory with home so starting-room clusters are immediately
            # within the LiDAR coverage blacklist radius.
            self._trajectory.append((self._home_x, self._home_y))
            self._last_traj_x = self._home_x
            self._last_traj_y = self._home_y
            self.get_logger().info(
                f'Home position recorded: ({self._home_x:.2f}, {self._home_y:.2f})')

        # Record trajectory waypoint every TRAJ_RECORD_DIST metres
        if self._has_tf:
            if math.hypot(self._robot_x - self._last_traj_x,
                          self._robot_y - self._last_traj_y) >= TRAJ_RECORD_DIST:
                self._trajectory.append((self._robot_x, self._robot_y))
                self._last_traj_x = self._robot_x
                self._last_traj_y = self._robot_y

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

        # ── Hard e-stop: only publish zero and warn — NO escape sequence ──
        if closest_any < self._estop_dist:
            self.get_logger().warn(
                f'E-STOP: obstacle at {closest_any:.2f} m — holding')
            self._cmd_pub.publish(Twist())
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

        # G14: FSM STATE MACHINE TRIGGERS
        HALLWAY_THRESH = 0.75  # both walls within this → hallway
        WALL_THRESH    = 2.5   # one wall within this → room perimeter
        front_sectors  = [0, 1, N_SECTORS - 1]
        front_clear    = min(sector_min[s] for s in front_sectors)

        # HALLWAY only if both walls close AND front is clear.
        # HALLWAY_MIN_FRONT guard prevents entering high-speed hallway mode
        # when a dead-end or turn is ahead — robot drops to ROOM instead.
        if (left_clear < HALLWAY_THRESH and right_clear < HALLWAY_THRESH
                and front_clear >= HALLWAY_MIN_FRONT):
            self._current_state = self.STATE_HALLWAY
        elif right_clear < WALL_THRESH or left_clear < WALL_THRESH:
            self._current_state = self.STATE_ROOM_PERIMETER
            if right_clear < WALL_THRESH + 0.5:
                self._hugging_side = 'RIGHT'
            elif left_clear < WALL_THRESH:
                self._hugging_side = 'LEFT'
        else:
            self._current_state = self.STATE_CROSSING

        # Reset alignment EMA on any FSM state transition to avoid stale
        # corrections bleeding into the new state's controller
        if self._current_state != self._prev_state:
            self._smooth_align = 0.0
            self._prev_state   = self._current_state

        mode_str = self._current_state
        if self._current_state == self.STATE_ROOM_PERIMETER:
            mode_str += f"({getattr(self, '_hugging_side', 'R')[0]})"
        fwd = 0.0
        turn = 0.0
        strafe_cmd = 0.0
        align_error = 0.0
        goal_str = 'none'

        # ══ RETURN-TO-HOME: A* path-following bypasses the FSM ════════════
        if self._returning_home and self._home_path and self._has_tf:
            wp = self._home_path[self._home_wp_idx]
            dx = wp[0] - self._robot_x
            dy = wp[1] - self._robot_y
            dist_wp = math.hypot(dx, dy)

            # Advance to next waypoint when close
            if dist_wp < 0.35 and self._home_wp_idx < len(self._home_path) - 1:
                self._home_wp_idx += 1
                wp = self._home_path[self._home_wp_idx]
                dx = wp[0] - self._robot_x
                dy = wp[1] - self._robot_y
                dist_wp = math.hypot(dx, dy)

            # Heading to waypoint
            wp_heading = math.atan2(math.sin(math.atan2(dy, dx) - self._robot_yaw),
                                   math.cos(math.atan2(dy, dx) - self._robot_yaw))

            # Proportional steering — stronger than exploration nudges
            turn = max(-self._turn_spd * 0.7, min(self._turn_spd * 0.7, wp_heading * 1.2))
            target_speed = NORMAL_SPEED
            mode_str = 'HOME'
            goal_str = f'wp {self._home_wp_idx+1}/{len(self._home_path)}'

        # ══ NORMAL EXPLORATION FSM ════════════════════════════════════════
        # HALLWAY: strafe to center, tiny turn for parallel alignment only.
        elif self._current_state == self.STATE_HALLWAY:
            center_err = (left_clear - right_clear) * HALLWAY_CENTER_GAIN
            # Deadband: skip micro-strafe when walls are within 5 cm of equal
            if abs(left_clear - right_clear) < 0.05:
                strafe_cmd = 0.0
            else:
                strafe_cmd = max(-self._move_spd * MAX_STRAFE_FRAC,
                                 min(self._move_spd * MAX_STRAFE_FRAC, center_err))

            SIN60 = 0.866
            r_front_right = sector_min[20 % N_SECTORS]
            r_rear_right  = sector_min[16 % N_SECTORS]
            r_front_left  = sector_min[ 4 % N_SECTORS]
            r_rear_left   = sector_min[ 8 % N_SECTORS]

            align_err = 0.0
            align_weight = 0.0
            if right_clear < WALL_THRESH:
                diff_r = r_rear_right - r_front_right
                if abs(diff_r) < 0.5:
                    align_err += diff_r * SIN60 * WALL_ALIGN_GAIN
                    align_weight += 1.0
            if left_clear < WALL_THRESH:
                diff_l = r_rear_left - r_front_left
                if abs(diff_l) < 0.5:
                    align_err -= diff_l * SIN60 * WALL_ALIGN_GAIN
                    align_weight += 1.0

            if align_weight > 0:
                align_err /= align_weight

            self._smooth_align = (ALIGN_EMA * align_err + (1 - ALIGN_EMA) * self._smooth_align)
            align_error = self._smooth_align

            turn = max(-MAX_ALIGN_TURN, min(MAX_ALIGN_TURN, align_error * HALLWAY_ALIGN_GAIN))

            # ±2° deadband: when alignment turn is within 2°/s the robot is
            # parallel to the walls — zero the correction so it drives
            # perfectly straight.  The goal nudge below still applies.
            if abs(turn) < math.radians(2.0):
                turn = 0.0

            # Mild goal nudge in hallway: steer toward frontier goal so the
            # robot naturally drifts toward doorways rather than charging into
            # dead ends.  Capped at 0.08 rad/s — won't override wall centering.
            if self._goal_world is not None and self._has_tf:
                gh = self._heading_to(*self._goal_world)
                turn += max(-0.08, min(0.08, gh * 0.12))
                goal_str = f'({self._goal_world[0]:.1f},{self._goal_world[1]:.1f})'

            target_speed = HALLWAY_SPEED

        elif self._current_state == self.STATE_ROOM_PERIMETER:
            SIN60 = 0.866
            r_front_right = sector_min[20 % N_SECTORS]
            r_rear_right  = sector_min[16 % N_SECTORS]
            r_front_left  = sector_min[ 4 % N_SECTORS]
            r_rear_left   = sector_min[ 8 % N_SECTORS]

            if getattr(self, '_hugging_side', 'RIGHT') == 'RIGHT':
                if right_clear < 3.5:
                    dist_err   = (WALL_TARGET - right_clear) * WALL_DIST_GAIN
                    strafe_cmd = max(-self._move_spd * MAX_STRAFE_FRAC,
                                    min(self._move_spd * MAX_STRAFE_FRAC, dist_err))
                    diff_r = r_rear_right - r_front_right
                    align_err = diff_r * SIN60 * WALL_ALIGN_GAIN if abs(diff_r) < 0.5 else 0.0
                    self._smooth_align = (ALIGN_EMA * align_err + (1 - ALIGN_EMA) * self._smooth_align)
                    turn = max(-MAX_ALIGN_TURN, min(MAX_ALIGN_TURN, self._smooth_align))
                    align_error = self._smooth_align
                else:
                    strafe_cmd = -self._move_spd * 0.30
                    turn = 0.0
            else:
                if left_clear < 3.5:
                    dist_err   = -(WALL_TARGET - left_clear) * WALL_DIST_GAIN
                    strafe_cmd = max(-self._move_spd * MAX_STRAFE_FRAC,
                                    min(self._move_spd * MAX_STRAFE_FRAC, dist_err))
                    diff_l = r_rear_left - r_front_left
                    align_err = -diff_l * SIN60 * WALL_ALIGN_GAIN if abs(diff_l) < 0.5 else 0.0
                    self._smooth_align = (ALIGN_EMA * align_err + (1 - ALIGN_EMA) * self._smooth_align)
                    turn = max(-MAX_ALIGN_TURN, min(MAX_ALIGN_TURN, self._smooth_align))
                    align_error = self._smooth_align
                else:
                    strafe_cmd = self._move_spd * 0.30
                    turn = 0.0

            target_speed = NORMAL_SPEED

        elif self._current_state == self.STATE_CROSSING:
            target_speed = NORMAL_SPEED
            if self._goal_world is not None and self._has_tf:
                goal_heading = self._heading_to(*self._goal_world)
                turn = max(-GOAL_BIAS_W * 2.0, min(GOAL_BIAS_W * 2.0, goal_heading * 0.8))
                goal_str = f'({self._goal_world[0]:.1f},{self._goal_world[1]:.1f})'
            else:
                turn = -0.35

        # ── Global Obstacle Avoidance (Overrides State) ───────────────────
        # Find best open sector in the forward hemisphere
        fwd_scores = {}
        sector_ang = 2.0 * math.pi / N_SECTORS
        for s in range(N_SECTORS):
            clearance = min(sector_min[s], 3.0)
            sec_local = math.atan2(math.sin(s * sector_ang),
                                   math.cos(s * sector_ang))
            if abs(sec_local) > math.radians(120): continue
            fwd_scores[s] = clearance
        
        best_ang = 0.0
        if fwd_scores:
            best_s    = max(fwd_scores, key=lambda s: fwd_scores[s])
            best_ang  = math.atan2(math.sin(best_s * sector_ang),
                                   math.cos(best_s * sector_ang))

        # ── Obstacle avoidance: gentle correction, strafe-first for minor offsets
        # HALLWAY dead-end: the HALLWAY_MIN_FRONT guard means if we arrive here
        # with front_clear < obs_dist in HALLWAY mode, the robot genuinely hit
        # something unexpected (e.g. a person / obstacle mid-corridor).  Treat
        # it like ROOM — allow full corner escape so the robot isn't stranded.
        # In HALLWAY mode, brief front-sensor dips are transient (sensor noise
        # or a passing person).  A 40° corner escape would corrupt the robot's
        # yaw and break directional scoring for the next frontier pick.
        # HALLWAY_MIN_FRONT already drops the robot to ROOM mode for real blockages.
        in_hallway = (self._current_state == self.STATE_HALLWAY)

        if front_clear < self._obs_dist and not in_hallway:
            # Always increment corner count — including in hallway.
            # HALLWAY_MIN_FRONT prevents reaching this during normal wall-follow;
            # if we're here anyway something is genuinely blocking the path.
            self._corner_count += 1

            # Small lateral offset → prefer strafe (mecanum advantage)
            STRAFE_THRESHOLD = math.radians(20.0)
            if abs(best_ang) <= STRAFE_THRESHOLD and not self._corner_turning:
                strafe_cmd += math.sin(best_ang) * self._move_spd * 0.6
                turn = max(-math.radians(5) * (self._turn_spd / math.radians(45)),
                           min(math.radians(5) * (self._turn_spd / math.radians(45)),
                               best_ang * 0.15))
            elif not self._corner_turning:
                if self._corner_count >= 3:
                    self._corner_dir       = 1.0 if left_clear > right_clear else -1.0
                    self._corner_turning   = True
                    self._corner_start_yaw = self._robot_yaw
                else:
                    gentle_gain = 0.15
                    turn = max(-self._turn_spd * 0.3,
                               min(self._turn_spd * 0.3,
                                   self._turn_spd * gentle_gain * (best_ang / (math.pi / 2.0))))
            target_speed *= 0.35
        else:
            self._corner_count = 0

        # ── Committed hard-turn phase — exit when odometry shows ≥40° rotation
        if self._corner_turning:
            yaw_now = self._robot_yaw
            if self._corner_start_yaw is not None:
                # Angular distance travelled since hard-turn start
                rotated = abs(math.atan2(
                    math.sin(yaw_now - self._corner_start_yaw),
                    math.cos(yaw_now - self._corner_start_yaw)))
                if rotated >= self._corner_target_rad:
                    self._corner_turning = False
                    self._corner_start_yaw = None
                else:
                    turn = self._corner_dir * self._turn_spd
                    target_speed *= 0.2
            else:
                # Safety: no yaw reference yet — use current as start
                self._corner_start_yaw = self._robot_yaw

        turn = max(-self._turn_spd, min(self._turn_spd, turn))

        # ── Forward Speed Dynamics ────────────────────────────────────────
        effective_clear = min(front_clear, closest_any + self._robot_rad)
        if effective_clear < self._estop_dist:
            fwd = 0.0
        elif effective_clear < self._obs_dist:
            ratio = ((effective_clear - self._estop_dist) / (self._obs_dist - self._estop_dist))
            fwd = target_speed * max(0.2, ratio)
        else:
            fwd = target_speed

        turn_ratio   = abs(turn) / self._turn_spd
        turn_penalty = max(0.2, 1.0 - (turn_ratio * 1.5))
        fwd *= turn_penalty

        if closest_any < self._robot_rad:
            fwd = 0.0

        # ── Update stuck detection at 10 Hz ──────────────────────────────
        if self._has_tf:
            moved = math.hypot(self._robot_x - self._last_pos_x,
                               self._robot_y - self._last_pos_y)
            if moved > 0.25:
                self._last_pos_x = self._robot_x
                self._last_pos_y = self._robot_y
                self._last_move_t = now

        # ── Strafe ────────────────────────────────────────────────────────
        # In HALLWAY at high speed (0.52 m/s), large strafe causes severe
        # FR/RL wheel asymmetry. Cap hallway strafe tightly.
        # For ROOM/CROSSING, allow up to move_spd * 0.8 as before.
        if self._current_state == self.STATE_HALLWAY:
            max_str = 0.06   # ~6 cm/s max lateral push in hallway
        else:
            max_str = self._move_spd * 0.8
        total_strafe = (strafe_force * self._move_spd) + strafe_cmd
        strafe  = max(-max_str, min(max_str, total_strafe))

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

    # ═══════════════════════════════════════════════════════════════════════════
    # A* PATH PLANNER — shortest obstacle-free path from current pos to home
    # ═══════════════════════════════════════════════════════════════════════════

    def _plan_path_home(self):
        """Compute A* path on the OccupancyGrid from robot position to home."""
        m = self._map
        if m is None:
            self.get_logger().warn('No map for path planning — falling back to direct')
            self._home_path = [(self._home_x, self._home_y)]
            return

        w, h  = m.info.width, m.info.height
        res   = m.info.resolution
        ox    = m.info.origin.position.x
        oy    = m.info.origin.position.y
        data  = m.data

        # ── World ↔ grid conversions ──────────────────────────────────────
        def to_grid(wx, wy):
            c = int((wx - ox) / res)
            r = int((wy - oy) / res)
            return (max(0, min(r, h - 1)), max(0, min(c, w - 1)))

        def to_world(r, c):
            return ox + (c + 0.5) * res, oy + (r + 0.5) * res

        # ── Inflate obstacles by robot radius ─────────────────────────────
        inflate = int(math.ceil(self._robot_rad / res)) + 1
        passable = bytearray(w * h)
        for i in range(w * h):
            passable[i] = 1 if data[i] == 0 else 0  # only FREE cells

        inflated = bytearray(passable)
        for r in range(h):
            for c in range(w):
                if data[r * w + c] > 0:  # occupied
                    for dr in range(-inflate, inflate + 1):
                        for dc in range(-inflate, inflate + 1):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                inflated[nr * w + nc] = 0

        # ── A* search ─────────────────────────────────────────────────────
        start = to_grid(self._robot_x, self._robot_y)
        goal  = to_grid(self._home_x, self._home_y)

        # Ensure start and goal are passable (robot is there, so it must be)
        inflated[start[0] * w + start[1]] = 1
        inflated[goal[0] * w + goal[1]]   = 1

        SQRT2 = 1.414
        open_set = [(0.0, start)]
        g_score  = {start: 0.0}
        came_from = {}

        while open_set:
            _, cur = heapq.heappop(open_set)
            if cur == goal:
                break

            cr, cc = cur
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if not inflated[nr * w + nc]:
                    continue
                step = SQRT2 if (dr != 0 and dc != 0) else 1.0
                ng = g_score[cur] + step
                nkey = (nr, nc)
                if ng < g_score.get(nkey, float('inf')):
                    g_score[nkey] = ng
                    f = ng + math.hypot(nr - goal[0], nc - goal[1])
                    heapq.heappush(open_set, (f, nkey))
                    came_from[nkey] = cur

        # ── Reconstruct path ──────────────────────────────────────────────
        if goal not in came_from and start != goal:
            self.get_logger().warn('A* found no path — falling back to direct')
            self._home_path = [(self._home_x, self._home_y)]
            return

        path_cells = []
        cur = goal
        while cur in came_from:
            path_cells.append(cur)
            cur = came_from[cur]
        path_cells.append(start)
        path_cells.reverse()

        # Simplify: keep every 10th cell (~0.5m spacing at 0.05m res)
        simplified = [path_cells[i] for i in range(0, len(path_cells), 10)]
        if simplified[-1] != path_cells[-1]:
            simplified.append(path_cells[-1])

        self._home_path = [to_world(r, c) for r, c in simplified]
        self._home_wp_idx = 0
        self.get_logger().info(
            f'A* path: {len(path_cells)} cells → {len(self._home_path)} waypoints')

    # ═══════════════════════════════════════════════════════════════════════════
    # MAP SAVING — called once when exploration is complete and robot is home
    # ═══════════════════════════════════════════════════════════════════════════

    def _save_map(self):
        """Save the current map as PGM + YAML and request pose graph serialization."""
        os.makedirs(MAP_SAVE_DIR, exist_ok=True)

        m = self._map
        if m is None:
            self.get_logger().warn('No map data available to save')
            return

        w, h = m.info.width, m.info.height
        res   = m.info.resolution
        ox    = m.info.origin.position.x
        oy    = m.info.origin.position.y

        # ── Write PGM (P5 binary grayscale) ───────────────────────────────
        pgm_path = os.path.join(MAP_SAVE_DIR, 'exploration_map.pgm')
        try:
            with open(pgm_path, 'wb') as f:
                f.write(f'P5\n{w} {h}\n255\n'.encode())
                # OccupancyGrid row 0 = bottom of map; PGM row 0 = top → flip
                for row in range(h - 1, -1, -1):
                    row_data = bytearray(w)
                    for col in range(w):
                        val = m.data[row * w + col]
                        if val == -1:       # unknown
                            row_data[col] = 205
                        elif val == 0:      # free
                            row_data[col] = 254
                        else:               # occupied (typically 100)
                            row_data[col] = 0
                    f.write(row_data)
            self.get_logger().info(f'Map image saved: {pgm_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save PGM: {e}')

        # ── Write YAML metadata ───────────────────────────────────────────
        yaml_path = os.path.join(MAP_SAVE_DIR, 'exploration_map.yaml')
        try:
            with open(yaml_path, 'w') as f:
                f.write(f'image: exploration_map.pgm\n')
                f.write(f'resolution: {res}\n')
                f.write(f'origin: [{ox}, {oy}, 0.0]\n')
                f.write(f'negate: 0\n')
                f.write(f'occupied_thresh: 0.65\n')
                f.write(f'free_thresh: 0.196\n')
            self.get_logger().info(f'Map metadata saved: {yaml_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save YAML: {e}')

        # ── Request slam_toolbox pose graph serialization ─────────────────
        posegraph_path = os.path.join(MAP_SAVE_DIR, 'exploration_posegraph')
        try:
            subprocess.Popen([
                'ros2', 'service', 'call',
                '/slam_toolbox/serialize_map',
                'slam_toolbox/srv/SerializePoseGraph',
                f"{{filename: '{posegraph_path}'}}"
            ])
            self.get_logger().info(
                f'Pose graph serialization requested: {posegraph_path}')
        except Exception as e:
            self.get_logger().warn(f'Pose graph serialization failed: {e}')

        self.get_logger().info(
            '══════════════════════════════════════════════════\n'
            '  EXPLORATION COMPLETE — MAP SAVED — MOTORS STOPPED\n'
            '══════════════════════════════════════════════════')


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