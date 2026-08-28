#!/usr/bin/env python3
"""
arm_lidar_mask_node.py — FK-based dynamic LiDAR arm masking
=============================================================
Filters the raw /scan topic to remove LiDAR returns caused by the robot's
own arm and chassis in the rear hemisphere.  Publishes /scan_filtered.

Two-layer filtering (rear 180° only — front 180° is NEVER touched):

  Layer 1 — REAR BODY BUBBLE
    Any return in the rear hemisphere closer than `rear_bubble_radius` is
    the robot seeing its own chassis+arm envelope.  Replaced with NaN.

  Layer 2 — FK ARM SHADOW
    Uses forward kinematics to compute 3D positions of each arm joint.
    For joints/links that cross the LiDAR scan plane, computes the angular
    shadow as seen from the LiDAR origin and masks those specific rays.

Decoupling
----------
This node depends ONLY on:
  - /arm/joint_states (sensor_msgs/JointState) — current joint angles
  - Arm physical geometry (link lengths, mount position) from parameters
  - Forward kinematics equations (determined by physical arm construction)

It does NOT depend on IK solvers, grab logic, control algorithms, or any
arm_manipulation package code.  The arm control pipeline can be freely
changed without affecting this node.

Subscribes
----------
  /scan               sensor_msgs/LaserScan   raw LiDAR data
  /arm/joint_states   sensor_msgs/JointState  arm joint positions (20 Hz)

Publishes
---------
  /scan_filtered      sensor_msgs/LaserScan   masked LiDAR data

Services
--------
  /arm_mask/enable    std_srvs/Trigger   resume masking (after grab phase)
  /arm_mask/disable   std_srvs/Trigger   passthrough mode (during grab phase)
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan, JointState
from std_srvs.srv import Trigger


class ArmLidarMaskNode(Node):
    """Dynamic LiDAR masking based on arm FK + rear body bubble."""

    # Joint name → index mapping for JointState messages
    JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                   'wrist_flex', 'wrist_roll']

    def __init__(self):
        super().__init__('arm_lidar_mask')

        # ── Parameters ────────────────────────────────────────────────────
        # Robot geometry
        self.declare_parameter('rear_bubble_radius',    0.584)   # m — 23 in total robot+arm length
        self.declare_parameter('lidar_scan_height',     0.376)   # m — scan plane above ground
        self.declare_parameter('lidar_x_offset',        0.0)     # m — LiDAR X offset from base_link
        self.declare_parameter('lidar_y_offset',        0.0)     # m — LiDAR Y offset from base_link
        self.declare_parameter('base_z_above_ground',   0.0908)  # m — base_link height above ground

        # Arm mounting (relative to base_link)
        self.declare_parameter('arm_mount_x',          -0.1715)  # m — rear edge of chassis
        self.declare_parameter('arm_mount_y',           0.0)     # m — centered widthwise
        self.declare_parameter('arm_mount_z',           0.040)   # m — chassis top surface (in base_link frame)
        self.declare_parameter('arm_faces_rear',        True)    # arm +x points toward robot rear

        # Arm link lengths (from arm_params.yaml, converted to meters)
        self.declare_parameter('shoulder_height',       0.170)   # m — shoulder pivot above arm base
        self.declare_parameter('l1',                    0.115)   # m — shoulder to elbow
        self.declare_parameter('l2',                    0.1375)  # m — elbow to wrist
        self.declare_parameter('l3',                    0.090)   # m — wrist to gripper/camera tip

        # Arm zero offsets (from arm_params.yaml)
        self.declare_parameter('pan_zero_offset_deg',  -4.6)
        self.declare_parameter('elbow_zero_offset_deg', 81.0)
        self.declare_parameter('wrist_zero_offset_deg', 5.0)

        # Masking tuning
        self.declare_parameter('scan_height_tolerance', 0.06)    # m — link within ±6cm of scan plane is "visible"
        self.declare_parameter('shadow_padding_deg',    4.0)     # ° — extra padding per side of each shadow
        self.declare_parameter('link_radius',           0.025)   # m — effective cylindrical radius of arm links
        self.declare_parameter('camera_radius',         0.035)   # m — camera + cable effective radius

        # Read parameters
        self._bubble_r     = self.get_parameter('rear_bubble_radius').value
        self._scan_height  = self.get_parameter('lidar_scan_height').value
        self._lidar_x      = self.get_parameter('lidar_x_offset').value
        self._lidar_y      = self.get_parameter('lidar_y_offset').value
        self._base_z       = self.get_parameter('base_z_above_ground').value

        self._arm_x        = self.get_parameter('arm_mount_x').value
        self._arm_y        = self.get_parameter('arm_mount_y').value
        self._arm_z        = self.get_parameter('arm_mount_z').value
        self._arm_rear     = self.get_parameter('arm_faces_rear').value

        self._sh_h         = self.get_parameter('shoulder_height').value
        self._l1           = self.get_parameter('l1').value
        self._l2           = self.get_parameter('l2').value
        self._l3           = self.get_parameter('l3').value

        self._pan_off      = math.radians(self.get_parameter('pan_zero_offset_deg').value)
        self._elbow_off    = math.radians(self.get_parameter('elbow_zero_offset_deg').value)
        self._wrist_off    = math.radians(self.get_parameter('wrist_zero_offset_deg').value)

        self._height_tol   = self.get_parameter('scan_height_tolerance').value
        self._pad          = math.radians(self.get_parameter('shadow_padding_deg').value)
        self._link_r       = self.get_parameter('link_radius').value
        self._cam_r        = self.get_parameter('camera_radius').value

        # Precompute: LiDAR scan plane height in base_link frame
        self._scan_z_base  = self._scan_height - self._base_z  # ≈ 0.2855 m

        # ── State ─────────────────────────────────────────────────────────
        self._masking_enabled = True
        self._latest_joints = {}           # joint_name → angle (degrees)
        self._has_joints = False
        self._shadow_regions = []          # [(min_angle, max_angle), ...]

        # ── Pub/Sub ───────────────────────────────────────────────────────
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(
            LaserScan, '/scan', self._on_scan, scan_qos)
        self.create_subscription(
            JointState, '/arm/joint_states', self._on_joint_states, 10)

        self._filtered_pub = self.create_publisher(
            LaserScan, '/scan_filtered', scan_qos)

        # ── Services ──────────────────────────────────────────────────────
        self.create_service(Trigger, '/arm_mask/enable',  self._srv_enable)
        self.create_service(Trigger, '/arm_mask/disable', self._srv_disable)

        self.get_logger().info(
            f'Arm LiDAR mask ready | bubble={self._bubble_r:.3f}m | '
            f'scan_plane={self._scan_height:.3f}m | '
            f'arm_mount=({self._arm_x:.3f}, {self._arm_y:.3f}, {self._arm_z:.3f}) | '
            f'arm_faces_rear={self._arm_rear}')

    # ── Service handlers ──────────────────────────────────────────────────

    def _srv_enable(self, request, response):
        self._masking_enabled = True
        self.get_logger().info('Arm LiDAR masking ENABLED')
        response.success = True
        response.message = 'Masking enabled'
        return response

    def _srv_disable(self, request, response):
        self._masking_enabled = False
        self.get_logger().info('Arm LiDAR masking DISABLED (passthrough)')
        response.success = True
        response.message = 'Masking disabled — passthrough mode'
        return response

    # ── Joint state callback ──────────────────────────────────────────────

    def _on_joint_states(self, msg: JointState):
        """Cache latest arm joint angles (degrees) from the arm driver."""
        for i, name in enumerate(msg.name):
            if name in self.JOINT_NAMES and i < len(msg.position):
                self._latest_joints[name] = msg.position[i]
        self._has_joints = bool(self._latest_joints)

        # Recompute shadow regions whenever joints update
        if self._has_joints:
            self._shadow_regions = self._compute_arm_shadow()

    # ── Forward Kinematics (positions only) ───────────────────────────────

    def _arm_joint_positions(self):
        """Compute 3D positions of each arm joint in base_link frame.

        Returns list of (x, y, z) tuples for:
          [0] shoulder_pivot
          [1] elbow
          [2] wrist
          [3] gripper_tip / camera_tip
        """
        pan_deg   = self._latest_joints.get('shoulder_pan',   0.0)
        lift_deg  = self._latest_joints.get('shoulder_lift',  0.0)
        elbow_deg = self._latest_joints.get('elbow_flex',     0.0)
        wrist_deg = self._latest_joints.get('wrist_flex',     0.0)

        # FK angles (same equations as arm_kinematics.py)
        pan_rad = math.radians(pan_deg) + self._pan_off
        t1 = math.radians(90.0 - lift_deg)
        t2 = t1 - (math.radians(elbow_deg) + self._elbow_off)
        t3 = t2 - (math.radians(wrist_deg) + self._wrist_off)

        # Joint positions in arm frame (meters)
        # Shoulder pivot — at top of pedestal
        positions_arm = [(0.0, 0.0, self._sh_h)]

        # Elbow
        ex_planar = self._l1 * math.cos(t1)
        ez_planar = self._l1 * math.sin(t1)
        ex = ex_planar * math.cos(pan_rad)
        ey = ex_planar * math.sin(pan_rad)
        ez = ez_planar + self._sh_h
        positions_arm.append((ex, ey, ez))

        # Wrist
        wx_planar = ex_planar + self._l2 * math.cos(t2)
        wz_planar = ez_planar + self._l2 * math.sin(t2)
        wx = wx_planar * math.cos(pan_rad)
        wy = wx_planar * math.sin(pan_rad)
        wz = wz_planar + self._sh_h
        positions_arm.append((wx, wy, wz))

        # Gripper / camera tip
        gx_planar = wx_planar + self._l3 * math.cos(t3)
        gz_planar = wz_planar + self._l3 * math.sin(t3)
        gx = gx_planar * math.cos(pan_rad)
        gy = gx_planar * math.sin(pan_rad)
        gz = gz_planar + self._sh_h
        positions_arm.append((gx, gy, gz))

        # Transform arm frame → base_link frame
        # If arm faces rear: arm +x = robot -x, arm +y = robot -y
        sign = -1.0 if self._arm_rear else 1.0
        positions_base = []
        for ax, ay, az in positions_arm:
            bx = self._arm_x + sign * ax
            by = self._arm_y + sign * ay
            bz = self._arm_z + az
            positions_base.append((bx, by, bz))

        return positions_base

    # ── Shadow computation ────────────────────────────────────────────────

    def _compute_arm_shadow(self):
        """Compute angular shadow regions of the arm in the LiDAR scan plane.

        Returns a list of (min_angle, max_angle) tuples in radians,
        where angles are measured from the LiDAR origin using the
        standard LaserScan convention (0 = forward, positive = CCW).
        """
        positions = self._arm_joint_positions()
        if not positions:
            return []

        shadow_angles = []

        # For each link segment, check if it crosses or is near the scan plane
        link_segments = [
            (positions[0], positions[1], self._link_r),   # shoulder → elbow
            (positions[1], positions[2], self._link_r),   # elbow → wrist
            (positions[2], positions[3], self._cam_r),    # wrist → camera/gripper (wider for cable)
        ]

        for (p1, p2, radius) in link_segments:
            z1 = p1[2]  # in base_link frame
            z2 = p2[2]

            # Check if either endpoint or anything in between is near the scan plane
            z_min = min(z1, z2) - radius
            z_max = max(z1, z2) + radius

            if z_max < (self._scan_z_base - self._height_tol):
                continue  # Entire link is below scan plane
            if z_min > (self._scan_z_base + self._height_tol):
                # Link is above scan plane — but it still blocks rays
                # (LiDAR can't see through it to read the wall behind)
                # For now, include it since the beam would hit the link
                # on the way up if the link is close enough vertically.
                # Only skip if very far above.
                if z_min > (self._scan_z_base + 0.15):
                    continue

            # This link is in or near the scan plane — compute angular span
            # Find the 2D (x, y) positions of both endpoints relative to LiDAR
            for px, py, pz in [p1, p2]:
                dx = px - self._lidar_x
                dy = py - self._lidar_y
                dist = math.hypot(dx, dy)
                if dist < 0.01:
                    continue  # Point is at the LiDAR origin (shouldn't happen)

                angle = math.atan2(dy, dx)

                # Angular padding = geometry padding + config padding
                # Angular extent of the link's radius at this distance
                geom_pad = math.atan2(radius, dist) if dist > radius else math.pi / 4
                total_pad = geom_pad + self._pad

                shadow_angles.append((angle - total_pad, angle + total_pad))

        if not shadow_angles:
            return []

        # Merge overlapping shadow regions
        return self._merge_angle_ranges(shadow_angles)

    @staticmethod
    def _merge_angle_ranges(ranges):
        """Merge overlapping angular ranges.  Input: [(min, max), ...].
        Returns merged list of non-overlapping ranges.
        """
        if not ranges:
            return []

        # Sort by start angle
        sorted_r = sorted(ranges, key=lambda r: r[0])
        merged = [sorted_r[0]]

        for lo, hi in sorted_r[1:]:
            prev_lo, prev_hi = merged[-1]
            if lo <= prev_hi:
                # Overlapping — extend
                merged[-1] = (prev_lo, max(prev_hi, hi))
            else:
                merged.append((lo, hi))

        return merged

    # ── Scan filtering ────────────────────────────────────────────────────

    def _on_scan(self, msg: LaserScan):
        """Filter raw scan and publish /scan_filtered."""
        if not self._masking_enabled:
            # Passthrough mode — republish unchanged
            self._filtered_pub.publish(msg)
            return

        # Work with a numpy array for performance
        ranges = np.array(msg.ranges, dtype=np.float32)
        n_rays = len(ranges)
        angles = np.arange(n_rays) * msg.angle_increment + msg.angle_min

        # ── Identify rear hemisphere (|angle| > π/2) ──────────────────
        # Standard convention: 0 = forward, ±π = rear
        rear_mask = np.abs(angles) > (math.pi / 2.0)

        # ── Layer 1: Rear body bubble ─────────────────────────────────
        # Any reading in the rear hemisphere closer than the bubble radius
        # is the robot seeing itself.
        bubble_mask = rear_mask & (ranges < self._bubble_r) & (ranges > 0.0)
        ranges[bubble_mask] = float('nan')

        # ── Layer 2: FK arm shadow ────────────────────────────────────
        if self._has_joints and self._shadow_regions:
            for shadow_lo, shadow_hi in self._shadow_regions:
                # Find rays within this shadow angular range
                # Handle angle wrapping: normalize both shadow bounds and ray angles
                # to the same range for comparison
                in_shadow = self._angles_in_range(angles, shadow_lo, shadow_hi)
                shadow_filter = in_shadow & rear_mask
                ranges[shadow_filter] = float('nan')

        # ── Publish filtered scan ─────────────────────────────────────
        filtered = LaserScan()
        filtered.header = msg.header
        filtered.angle_min = msg.angle_min
        filtered.angle_max = msg.angle_max
        filtered.angle_increment = msg.angle_increment
        filtered.time_increment = msg.time_increment
        filtered.scan_time = msg.scan_time
        filtered.range_min = msg.range_min
        filtered.range_max = msg.range_max
        filtered.ranges = ranges.tolist()
        filtered.intensities = list(msg.intensities) if msg.intensities else []

        self._filtered_pub.publish(filtered)

    @staticmethod
    def _angles_in_range(angles, lo, hi):
        """Check which angles fall within [lo, hi], handling wraparound.

        All angles in radians.  Handles the case where the range wraps
        around ±π (e.g., shadow from 170° to 190° = 2.967 to 3.316 rad,
        which might wrap in the LaserScan convention of [-π, π]).
        """
        # Normalize lo and hi to [-π, π]
        lo_n = math.atan2(math.sin(lo), math.cos(lo))
        hi_n = math.atan2(math.sin(hi), math.cos(hi))

        if lo_n <= hi_n:
            # Normal range (no wraparound)
            return (angles >= lo_n) & (angles <= hi_n)
        else:
            # Wraps around ±π (e.g., shadow spans from 170° to -170°)
            return (angles >= lo_n) | (angles <= hi_n)


def main(args=None):
    rclpy.init(args=args)
    node = ArmLidarMaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
