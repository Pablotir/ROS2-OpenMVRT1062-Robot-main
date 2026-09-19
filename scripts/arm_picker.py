#!/usr/bin/env python3
"""
arm_picker.py — YOLO-World + Moondream + RealSense D405 + IK Pick-and-Place
State machine: SEARCHING → VERIFYING → ALIGNING → GRABBING → RETURNING

# ── VERSION HISTORY ──────────────────────────────────────────────────────────
# V1 (2026-06-06)  WiFi camera + visual-servoing multipliers
# V2 (2026-06-10)  YOLO-World, active sweep, resistance-sensing grip
# V3 (2026-07-12)  Intel RealSense D405 (USB, wired) + Analytical IK
#   • CameraStream (WiFi/HTTP) → RealSenseStream (pyrealsense2)
#   • Multiplier calibration removed — IK computes exact joint angles from depth
#   • Wrist roll now commanded (unlocked)
#   • D405 70 mm minimum-range blind zone accounted for in approach planning
#   • Eye-in-hand FK+IK: arm already aligned → reach computed from live depth
# V4 (2026-07-25)  Proper homogeneous-matrix eye-in-hand calibration
#   • T_cam_wrist: explicit 4x4 calibration matrix (measure offsets once)
#   • FK now returns full 4x4 T_wrist_base instead of xyz tuple
#   • Coordinate chain: P_cam → T_cam_wrist → T_wrist_base → P_base
#   • Switched to YOLO segmentation model for mask-based depth sampling
#   • get_xyz_from_mask(): depth median over object pixels only (no background)
# V5 (2026-08-02)  R3 Workspace Boundary Calibration
#   • Option 4 interactive menu tests physical calibrated bounds for the robot in R3 space
#   • Fixed Lift joint angle sign inversion in analytical IK / FK
#   • Applied exact pan offset and pan limits from empirical testing
# V6 (2026-08-03)  Weighted IK, Level Approach, Surface Scanning
#   • Weighted joint preference: lift=3×, elbow=2×, wrist=1× (calibration-derived)
#   • Auto-pitch from geometry (clamped to calibrated -60°..−5° range)
#   • level_approach(): all servos at same speed, wrist compensates to keep
#     gripper parallel to floor throughout motion
#   • Post-lunge re-alignment pass and mask-size guard (reject >15% frame area)
#   • surface_proximity_depth(): depth ring-scan around ball detects floor/table
#     and clamps gripper approach to avoid pushing past the surface
#
#   NOTE — Wrist Roll (servo 5):
#   Currently held at 0° for all floor grabs (horizontal fingers, top-down).
#   Future work: angled grabs for cups, mugs, or objects with handles / shapes
#   that exceed the claw's grip width can be more optimally grabbed at an angle
#   if they have a handle or if the object's longest axis would benefit from
#   rotating the roll to align with the claw opening.
# ─────────────────────────────────────────────────────────────────────────────
"""
import cv2, time, signal, base64, math, threading, atexit
import numpy as np
import pyrealsense2 as rs
import requests

from ultralytics import YOLO

try:
    from lerobot.robots.so101_follower.so101_follower import SO101Follower as SOFollower
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig as SOFollowerRobotConfig
except (ImportError, ModuleNotFoundError):
    try:
        from lerobot.robots.so_follower.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    except (ImportError, ModuleNotFoundError):
        from lerobot.common.robot_devices.robots.feetech import SO100Follower as SOFollower
        from lerobot.common.robot_devices.robots.configs import SO100FollowerConfig as SOFollowerRobotConfig

# ── Hardware ──────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
PORT       = "/dev/arm_controller"
ARM_ID     = "jetson_arm"

# ── Detection ─────────────────────────────────────────────────────────────────
TARGET_DESC      = "red ball"
YOLO_CLASS_ID    = 32     # 32 = sports ball in standard COCO segmentation models
SKIP_MOONDREAM   = True   # Set False to re-enable Moondream semantic verification

# ── Arm Geometry (SO-ARM101 — User Measured Constants) ───────────────────────
IK_L1 = 115.0   # shoulder pivot  → elbow pivot  (11.5 cm)
IK_L2 = 137.5   # elbow pivot     → wrist pivot   (13.75 cm)
IK_L3 = 90.0    # wrist pivot     → gripper tip   (9.0 cm)
# Note: Maximum total physical reach of arm from base = 115 + 137.5 + 90 = 342.5 mm

# Pan alignment calibration (measured: -4.6° corresponds to straight ahead X-axis)
PAN_ZERO_OFFSET_DEG = -4.6
PAN_MIN_DEG         = -113.8   # Far left user-preferred limit
PAN_MAX_DEG         =  113.8   # Far right user-preferred limit

# ── Empirical R3 Workspace Bounds (SO-ARM101 — calibrated from physical images) ──
# These are the outer convex envelope limits measured in the ARM BASE frame
# (origin = shoulder pivot, +X forward, +Y left, +Z up).
# Any target from the depth camera outside these bounds will never be reachable.
#
#   X: min = arm folds back to robot body edge (~-115mm)
#      max = furthest forward horizontal reach (~+260mm floor-level)
#   Y: min = far right limit (mirrored from left, ~-211mm)
#      max = far left limit  (~+211mm)
#   Z: min = below shoulder pivot at floor reach (~-202mm)
#      max = straight up overhead (~+251mm)
#   Horizontal rho (sqrt(X^2+Y^2)): max at floor grab level (~260mm)
WS_X_MIN_MM  = -140.0   # behind the robot (folded-in elbow can go slightly negative, added 20mm tolerance)
WS_X_MAX_MM  =  285.0   # max forward reach (floor-level, added 20mm tolerance)
WS_Y_MAX_MM  =  235.0   # max lateral left  (added 24mm tolerance)
WS_Y_MIN_MM  = -235.0   # max lateral right (mirrored)
WS_Z_MIN_MM  = -225.0   # lowest reachable height below shoulder pivot (added 20mm tolerance for floor grab)
WS_Z_MAX_MM  =  275.0   # highest point straight up (added 20mm tolerance)
WS_RHO_MAX_MM =  285.0  # max horizontal extension from shoulder pivot

# ── RealSense D405 — Eye-in-Hand Calibration (T_cam_wrist) ──────────────────
# Static 4×4 homogeneous transform: how the D405 is physically bolted to the wrist.
# STEP 1: Measure these once with a ruler after you have mounted the camera.
#
#   X_offset: lateral offset  (+ve = camera is to the LEFT  of wrist centre)
#   Y_offset: vertical offset (+ve = camera is ABOVE wrist centre)
#   Z_offset: forward offset  (+ve = camera lens is FURTHER FORWARD than wrist centre)
#
# If the camera is tilted (pitched down), uncomment and fill in CAM_PITCH_DEG.
# Positive pitch = camera looks downward.
CAM_X_OFFSET_MM =  0.0   # lateral  (measure and tune)
CAM_Y_OFFSET_MM =  50.0   # vertical (measure and tune)
CAM_Z_OFFSET_MM = 0.0   # forward  (D405 lens protrudes ~30 mm past wrist pivot)
CAM_PITCH_DEG   =  45.0   # tilt of camera relative to wrist axis (0 = parallel)

# D405 minimum usable range. Objects closer than this have no valid depth.
D405_MIN_RANGE_MM = 70.0

# Maximum realistic table grab depth (mm). Readings larger than this (e.g. 1200mm)
# mean the depth sensor sampled floor/background noise, so we reject them.
MAX_GRAB_DEPTH_MM = 600.0

# How far PAST the object surface the gripper tip should be at grab time.
# 0 = gripper tip exactly at object surface. Increase to reach deeper.
GRASP_PENETRATION_MM = 20.0

def workspace_in_bounds(x_mm: float, y_mm: float, z_mm: float) -> bool:
    """
    Fast R3 bounding-box pre-check using empirically measured workspace limits.
    Returns True if (x, y, z) in the arm-base frame could possibly be reachable,
    False if it is definitively outside the physical envelope — IK is pointless.
    Called BEFORE solve_ik() to give a cleaner, more informative rejection message.
    """
    rho = math.sqrt(x_mm**2 + y_mm**2)
    if x_mm  < WS_X_MIN_MM:  return False
    if x_mm  > WS_X_MAX_MM:  return False
    if y_mm  < WS_Y_MIN_MM:  return False
    if y_mm  > WS_Y_MAX_MM:  return False
    if z_mm  < WS_Z_MIN_MM:  return False
    if z_mm  > WS_Z_MAX_MM:  return False
    if rho   > WS_RHO_MAX_MM: return False
    return True


def _build_T_cam_wrist() -> np.ndarray:
    """
    Build the 4x4 homogeneous calibration matrix T_cam_wrist.
    Loads the multi-pose optimization result from hand_eye_calibration.yaml
    if available, otherwise falls back to the CAD mount matrix.
    """
    import os, yaml
    calib_paths = [
        "/root/ros2_ws/calibration/hand_eye_calibration.yaml",
        os.path.join(os.path.dirname(__file__), "..", "calibration", "hand_eye_calibration.yaml"),
        os.path.join(os.path.dirname(__file__), "calibration", "hand_eye_calibration.yaml"),
        "calibration/hand_eye_calibration.yaml",
        "hand_eye_calibration.yaml"
    ]
    for p in calib_paths:
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    calib = yaml.safe_load(f)
                R = np.array(calib['rotation_matrix'], dtype=np.float64)
                t = np.array(calib['translation_mm'], dtype=np.float64).flatten()
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t
                method = calib.get('method', 'CALIBRATED')
                print(f"✅ Loaded calibrated T_cam_wrist from: {p} ({method})")
                print(f"   Translation (mm): X={t[0]:.1f}, Y={t[1]:.1f}, Z={t[2]:.1f}")
                return T
            except Exception as e:
                print(f"⚠️ Failed reading {p}: {e}")

    print("⚠️  hand_eye_calibration.yaml not found — using CAD mount matrix fallback.")
    alpha = math.radians(CAM_PITCH_DEG)
    sin_a, cos_a = math.sin(alpha), math.cos(alpha)
    R = np.array([
        [ 0.0, -sin_a,  cos_a],
        [ 0.0,  cos_a,  sin_a],
        [-1.0,   0.0,    0.0 ],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = [CAM_Z_OFFSET_MM, -CAM_Y_OFFSET_MM, CAM_X_OFFSET_MM]
    return T

# Precompute at import time
T_CAM_WRIST: np.ndarray = None   # set in main() after math module is loaded

# ── Searching Behaviour ───────────────────────────────────────────────────────
SEARCH_SWEEP_RANGE = 80.0   # ° pan left/right from START_POS centre
SEARCH_SWEEP_SPEED = 0.5    # ° per YOLO throttle tick (~5 fps = 2.5°/s)

# ── Alignment — closed-loop visual servoing ───────────────────────────────────
ALIGN_THRESHOLD    = 35    # px   centred when dot within this many px of crosshair (approx 10% margin)
ALIGN_CENTRED_NEED = 3     # consecutive centred frames to confirm
ALIGN_MAX_FRAMES   = 150   # give up after N frames (~7 s at 20 fps)
ALIGN_LOST_GRACE   = 25    # consecutive not-found frames before abort

ALIGN_PAN_OFFSET   = 35    # px: rightward crosshair offset (target ends up on the right, meaning the arm/claw is positioned to the LEFT of the object)

ALIGN_PAN_K   = 0.04
ALIGN_LIFT_K  = 0.05
ALIGN_MAX_PAN_DEG  = 3.0
ALIGN_MAX_LIFT_DEG = 1.5

ALIGN_INIT_PAN_K    = 0.20
ALIGN_INIT_LIFT_K   = 0.15
ALIGN_INIT_MAX_PAN  = 25.0
ALIGN_INIT_MAX_LIFT = 12.0

# ── Arm Positions ─────────────────────────────────────────────────────────────
# Calibrated scan posture (from calibration/arm_reference_poses.yaml)
_BASE = {
    "shoulder_pan.pos":   -4.48,
    "shoulder_lift.pos": -106.02,
    "elbow_flex.pos":     99.91,
    "wrist_flex.pos":     33.41,
    "wrist_roll.pos":   -155.96,
    "gripper.pos":        73.84,
}
# Calibrated stow posture (from calibration/arm_reference_poses.yaml)
_STOW_BASE = {
    "shoulder_pan.pos":   -4.48,
    "shoulder_lift.pos": -106.11,
    "elbow_flex.pos":    100.00,
    "wrist_flex.pos":     75.96,
    "wrist_roll.pos":   -156.75,
    "gripper.pos":        73.77,
}

_ALIGN_READY = {
    "shoulder_pan.pos":   -4.48,   # overwritten in pipeline to match target pan
    "shoulder_lift.pos":  86.6,   # arm extended forward horizontally
    "elbow_flex.pos":    -73.5,
    "wrist_flex.pos":     -8.9,
    "wrist_roll.pos":   -155.96,
    "gripper.pos":        73.84,
}

# ═══════════════════════════════════════════════════════════════════════════════
# RealSense D405 camera stream
# ═══════════════════════════════════════════════════════════════════════════════
class RealSenseStream:
    """
    Background thread that continuously pulls aligned color+depth frames
    from the Intel RealSense D405 over USB 3.0.

    Usage:
        cap = RealSenseStream()
        color_bgr, depth_frame = cap.read()
        xyz = cap.get_xyz(px, py)   # → (x_mm, y_mm, z_mm) or None
        cap.stop()
    """

    def __init__(self, width=848, height=480, fps=30):
        self._pipeline   = rs.pipeline()
        self._profile    = None
        self._bgr_convert = False

        # Try formats in order of preference for D405 on Linux/Jetson.
        # Each attempt is explicit so we can see exactly what fails.
        # The D405 on Jetson Linux MUST use YUYV. If we request BGR8,
        # the driver accepts it but silently delivers completely black frames.
        # We use 15fps to keep the CPU decode cost low.
        candidates = [
            (width, height, rs.format.yuyv, 15,  "YUYV 848x480 15fps"),
            (640,   480,    rs.format.yuyv, 15,  "YUYV 640x480 15fps"),
            (width, height, rs.format.rgb8, 15,  "RGB8 848x480 15fps"),
            (640,   480,    rs.format.rgb8, 15,  "RGB8 640x480 15fps"),
            (width, height, rs.format.bgr8, 15,  "BGR8 848x480 15fps"),
            (640,   480,    rs.format.bgr8, 15,  "BGR8 640x480 15fps"),
        ]
        for (w, h, fmt, f, label) in candidates:
            try:
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, w, h, fmt, f)
                cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, f)
                self._profile = self._pipeline.start(cfg)
                self._bgr_convert = (fmt == rs.format.rgb8 or fmt == rs.format.yuyv)
                print(f"📷 RealSense D405 started: {label}  (bgr_convert={self._bgr_convert})")
                width, height = w, h
                break
            except Exception as e:
                print(f"   ⚠️  {label} failed: {e}")
                self._pipeline.stop() if self._profile else None
                self._pipeline = rs.pipeline()  # reset pipeline

        if self._profile is None:
            # Last resort: auto-detect
            try:
                print("   🔄 Trying pipeline auto-detect (no explicit format)...")
                self._profile = self._pipeline.start()
                self._bgr_convert = True
                print("📷 RealSense D405 started: auto-detect")
            except Exception as e:
                raise RuntimeError(f"❌ Could not open RealSense in any format: {e}")

        # Query actual color format the hardware selected
        color_stream = self._profile.get_stream(rs.stream.color)
        self._color_format = color_stream.format()
        print(f"   🔍 Actual color format: {self._color_format}")

        self._colorizer  = rs.colorizer()
        self._colorizer.set_option(rs.option.color_scheme, 0) # Jet colormap
        self._colorizer.set_option(rs.option.histogram_equalization_enabled, 1)
        
        # Get depth scale for manual calculations
        depth_sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()
        
        self._lock       = threading.Lock()
        self._color      = np.zeros((height, width, 3), dtype=np.uint8)
        self._colorized  = np.zeros((height, width, 3), dtype=np.uint8)
        self._depth_img  = None
        self._intrinsics = None
        self._running    = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._running:
            try:
                frames      = self._pipeline.wait_for_frames(timeout_ms=5000)
                # D405 RGB and Depth share the same ISP sensor, so they are perfectly aligned natively.
                # Do NOT use rs.align, as it can corrupt D405 depth frames.
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    print("⚠️  RealSense: got frames but color/depth missing — check USB cable")
                    time.sleep(0.2)
                    continue
                color = np.asanyarray(color_frame.get_data())
                # Convert to BGR uint8 using the actual hardware format
                fmt = self._color_format
                if fmt == rs.format.bgr8:
                    pass  # already BGR uint8
                elif fmt == rs.format.rgb8:
                    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                elif fmt == rs.format.rgba8:
                    color = cv2.cvtColor(color, cv2.COLOR_RGBA2BGR)
                elif fmt == rs.format.bgra8:
                    color = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
                elif fmt == rs.format.yuyv:
                    # YUYV arrives as uint16 (H,W) on Jetson — reinterpret as uint8 bytes
                    # and reshape to (H, W, 2) which is what COLOR_YUV2BGR_YUYV requires
                    raw = np.asanyarray(color_frame.get_data()).view(np.uint8)
                    color = raw.reshape(color_frame.height, color_frame.width, 2)
                    color = cv2.cvtColor(color, cv2.COLOR_YUV2BGR_YUYV)
                elif fmt == rs.format.uyvy:
                    if color.ndim == 2 and color.shape[1] != color_frame.width:
                        color = color.reshape(color_frame.height, color_frame.width, 2)
                    color = cv2.cvtColor(color, cv2.COLOR_YUV2BGR_UYVY)
                elif fmt == rs.format.z16:
                    # Depth-as-color fallback: map 16-bit to 8-bit grey then BGR
                    color = (color >> 8).astype(np.uint8)
                    color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
                else:
                    # Unknown format — try to squeeze to BGR best-effort
                    print(f"   ⚠️  Unknown fmt {fmt}, shape {color.shape}, dtype {color.dtype}")
                    if color.ndim == 2:
                        color = cv2.cvtColor(color.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    elif color.shape[2] == 4:
                        color = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
                    elif color.shape[2] == 3 and self._bgr_convert:
                        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                # ALWAYS enforce uint8 BGR — Canny and YOLO require it
                color = np.ascontiguousarray(color, dtype=np.uint8)
                
                # Extract raw depth array
                depth_img = np.asanyarray(depth_frame.get_data()).copy()
                
                # Use official Intel RealSense colorizer (gives RGB, OpenCV needs BGR)
                colorized = np.asanyarray(self._colorizer.colorize(depth_frame).get_data())
                colorized = cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR)
                
                with self._lock:
                    self._color = color
                    self._colorized = colorized
                    self._depth_img = depth_img
                    self._intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            except Exception as e:
                print(f"⚠️  RealSense frame error: {e}")
                import traceback; traceback.print_exc()
                time.sleep(0.3)

    def read(self):
        """Return (color_bgr_copy, has_depth, colorized_depth)."""
        with self._lock:
            if self._depth_img is None:
                return self._color.copy(), False, None
            return self._color.copy(), True, self._colorized.copy()

    def get_xyz(self, px: int, py: int, search_w=40, search_h=40):
        """
        Deprojects pixel (px, py) into 3-D camera-space coordinates (mm).
        Uses a dynamic median patch (search_w x search_h) for noise rejection.
        Optimized with NumPy slicing.
        """
        with self._lock:
            depth_img = self._depth_img
            intr      = self._intrinsics
            scale     = self._depth_scale
        if depth_img is None or intr is None:
            return None
        
        # Calculate bounding box for the patch, clipped to image bounds
        x1 = max(0, px - search_w // 2)
        x2 = min(intr.width, px + search_w // 2 + 1)
        y1 = max(0, py - search_h // 2)
        y2 = min(intr.height, py + search_h // 2 + 1)
        
        patch = depth_img[y1:y2, x1:x2]
        if patch.size == 0:
            return None
            
        distances = patch.astype(float) * scale
        
        # Filter out 0 (invalid) and far distances (> 2.0 meters)
        valid = distances[(distances > 0.001) & (distances < 2.0)]
        
        if valid.size == 0:
            return None
            
        z_m = float(np.median(valid))
        point = rs.rs2_deproject_pixel_to_point(intr, [float(px), float(py)], z_m)
        return point[0] * 1000.0, point[1] * 1000.0, point[2] * 1000.0  # → mm

    def get_xyz_from_mask(self, mask: np.ndarray) -> tuple[float, float, float] | None:
        """
        Compute (cx_px, cy_px, z_mm) for the object described by a binary
        segmentation mask (same HxW as the color frame, dtype uint8, 255=object).

        Uses ONLY the depth pixels that belong to the mask so background
        depth values never contaminate the object distance estimate.
        This is the recommended method when a segmentation model is available.

        Returns (cx_px, cy_px, median_depth_mm) or None if no valid depth pixels.
        """
        with self._lock:
            depth_img = self._depth_img
            intr      = self._intrinsics
            scale     = self._depth_scale
        if depth_img is None or intr is None:
            return None

        # Centroid of the mask (pixel coordinates)
        M = cv2.moments(mask)
        if M["m00"] < 1:
            return None
        cx_px = int(M["m10"] / M["m00"])
        cy_px = int(M["m01"] / M["m00"])

        # Collect depth readings for all mask pixels
        h, w  = depth_img.shape[:2]
        mh, mw = mask.shape[:2]
        # Resize mask to depth resolution if they differ
        if (mh, mw) != (h, w):
            mask_rs = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_rs = mask

        raw_depths = depth_img[mask_rs == 255].astype(float) * scale
        valid = raw_depths[(raw_depths > 0.001) & (raw_depths < 2.0)]
        if valid.size == 0:
            return None

        z_m = float(np.median(valid))
        z_mm = z_m * 1000.0

        # Deproject centroid pixel using the median mask depth
        point = rs.rs2_deproject_pixel_to_point(
            intr, [float(cx_px), float(cy_px)], z_m)
        return float(cx_px), float(cy_px), z_mm

    def stop(self):
        self._running = False
        try:
            self._pipeline.stop()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Arm helpers
# ═══════════════════════════════════════════════════════════════════════════════
def get_pos(robot) -> dict:
    obs    = robot.get_observation()
    joints = {"shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
              "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"}
    return {k: v for k, v in obs.items() if k in joints}


def smooth_move(robot, target: dict, step_size=2.0, step_delay=0.02,
                hold_joints=None):
    if hold_joints is None:
        hold_joints = []
    cur = get_pos(robot)
    # Freeze hold_joints at their target immediately
    for j in hold_joints:
        if j in target:
            cur[j] = target[j]
    max_delta = max(abs(target[j] - cur.get(j, 0.0)) for j in target)
    if max_delta < 0.5:
        return
    n = max(1, int(max_delta / step_size))
    for s in range(1, n + 1):
        t      = s / n
        interp = {j: cur.get(j, 0.0) + t * (target[j] - cur.get(j, 0.0))
                  for j in target}
        robot.send_action(interp)
        time.sleep(step_delay)


def level_approach(robot, target: dict, step_size=2.0, step_delay=0.03):
    """
    Move all joints simultaneously toward *target* at uniform speed while
    continuously adjusting wrist_flex so the gripper stays parallel to the
    floor (end-effector pitch ≈ 0° relative to horizontal).

    At every interpolation step the wrist angle is computed from the
    current lift+elbow so that:
        pitch = t1 + t2 + wrist_motor = 0   (horizontal)
    where t1, t2 are the FK sagittal angles.

    All servos move at the same rate — no staging.
    """
    cur = get_pos(robot)
    max_delta = max(abs(target[j] - cur.get(j, 0.0)) for j in target)
    if max_delta < 0.5:
        return
    n = max(1, int(max_delta / step_size))

    for s in range(1, n + 1):
        t = s / n
        interp = {j: cur.get(j, 0.0) + t * (target[j] - cur.get(j, 0.0))
                  for j in target}

        # Compute wrist_flex to keep gripper level (pitch = desired pitch)
        # FK convention: t1_abs = 90 - lift, t2_abs = t1_abs - (elbow + 81.0)
        # pitch = t2_abs - (wrist + 5.0). For pitch = 0 (horizontal):
        # wrist = math.degrees(t2_abs) - 5.0
        # Target may have a non-zero desired pitch, so interpolate to final wrist.
        lift_now = interp.get("shoulder_lift.pos", 0.0)
        elb_now  = interp.get("elbow_flex.pos", 0.0)
        t1_rad = math.radians(90.0 - lift_now)
        t2_rad = t1_rad - math.radians(elb_now + 81.0)
        
        # level_wrist keeps pitch = 0 (horizontal)
        level_wrist = math.degrees(t2_rad) - 5.0
        
        # Blend: early steps → level;  final step → target wrist value
        final_wrist = target.get("wrist_flex.pos", level_wrist)
        interp["wrist_flex.pos"] = level_wrist + t * (final_wrist - level_wrist)

        robot.send_action(interp)
        time.sleep(step_delay)


def _set_torque(robot, enable: bool):
    action_str = "enable" if enable else "disable"
    val = 1 if enable else 0
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll", "gripper"]
    # Try multiple API patterns (LeRobot version differences)
    try:
        if enable:
            robot.bus.enable_torque()
        else:
            robot.bus.disable_torque()
        return True
    except Exception:
        pass
    try:
        robot.bus.write("Torque_Enable", [val] * len(motor_names), motor_names)
        return True
    except Exception:
        pass
    try:
        robot.bus.write("Torque_Enable", val, motor_names)
        return True
    except Exception:
        pass
    print(f"   ⚠️  Could not {action_str} torque (no matching API found).")
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Kinematics — SO-ARM101
# ═══════════════════════════════════════════════════════════════════════════════
def forward_kinematics(q: dict) -> np.ndarray:
    """
    Full 4×4 homogeneous Forward Kinematics: returns T_wrist_base.
    Matches calibrate_hand_eye.py and validate_calibration.py.
    """
    pan  = math.radians(-q.get("shoulder_pan.pos",  0.0) - PAN_ZERO_OFFSET_DEG)
    lift = q.get("shoulder_lift.pos", 0.0)
    elb  = q.get("elbow_flex.pos",    0.0)
    wst  = q.get("wrist_flex.pos",    0.0)
    roll = math.radians(-q.get("wrist_roll.pos",   0.0))

    t1 = math.radians(90.0 - lift)
    t2 = t1 - math.radians(elb + 81.0)
    t3 = t2 - math.radians(wst + 5.0)

    rho_w = IK_L1 * math.cos(t1) + IK_L2 * math.cos(t2)
    z_w   = IK_L1 * math.sin(t1) + IK_L2 * math.sin(t2)

    wx = rho_w * math.cos(pan)
    wy = rho_w * math.sin(pan)
    wz = z_w

    # Approach direction (Wrist X)
    ax = math.cos(t3) * math.cos(pan)
    ay = math.cos(t3) * math.sin(pan)
    az = math.sin(t3)

    # Perpendicular direction (Wrist Z)
    zx = -math.sin(pan)
    zy =  math.cos(pan)
    zz = 0.0

    # Wrist Y = Z cross X
    yx = zy * az - zz * ay
    yy = zz * ax - zx * az
    yz = zx * ay - zy * ax

    R_base = np.array([
        [ax, yx, zx],
        [ay, yy, zy],
        [az, yz, zz]
    ])

    R_roll = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(roll), -math.sin(roll)],
        [0.0, math.sin(roll),  math.cos(roll)]
    ])

    R_final = R_base @ R_roll

    T = np.array([
        [R_final[0,0], R_final[0,1], R_final[0,2], wx],
        [R_final[1,0], R_final[1,1], R_final[1,2], wy],
        [R_final[2,0], R_final[2,1], R_final[2,2], wz],
        [0., 0., 0., 1.],
    ])
    return T


def solve_ik(x_mm: float, y_mm: float, z_mm: float,
             end_pitch_deg: float | None = None,
             current_joints: dict | None = None,
             wrist_roll_deg: float = -155.96) -> dict | None:
    """
    Closed-form analytical IK for SO-ARM101.
    Target (x_mm, y_mm, z_mm) is in the ARM BASE frame.
    """
    # ── J1 (base pan) ────────────────────────────────────────────────────────
    pan_rad = math.atan2(y_mm, x_mm)
    pan_deg = -(math.degrees(pan_rad) + PAN_ZERO_OFFSET_DEG)

    if pan_deg < PAN_MIN_DEG or pan_deg > PAN_MAX_DEG:
        print(f"   ⚠️  IK reject: pan target {pan_deg:.1f}° out of bounds "
              f"({PAN_MIN_DEG:.1f}° to {PAN_MAX_DEG:.1f}°)")
        return None

    # ── Horizontal reach magnitude ───────────────────────────────────────────
    rho = math.sqrt(x_mm**2 + y_mm**2)

    # ── Auto-compute approach pitch if not specified ─────────────────────────
    if end_pitch_deg is None:
        natural_pitch = math.degrees(math.atan2(z_mm, rho))
        preferred_pitch = float(np.clip(natural_pitch, -80.0, -5.0))
    else:
        preferred_pitch = end_pitch_deg

    # ── Search for a reachable pitch ─────────────────────────────────────────
    pitch_candidates = [preferred_pitch]
    if end_pitch_deg is None:
        step = -5.0
        p = preferred_pitch + step
        while p >= -90.0:
            pitch_candidates.append(p)
            p += step

    best_solution = None
    best_cost = float('inf')

    for test_pitch in pitch_candidates:
        pitch_rad = math.radians(test_pitch)

        # ── Back out the wrist contribution ──────────────────────────────────
        wrist_x = rho   - IK_L3 * math.cos(pitch_rad)
        wrist_z = z_mm  - IK_L3 * math.sin(pitch_rad)

        # ── 2-link IK for J2 + J3 ────────────────────────────────────────────
        D = math.sqrt(wrist_x**2 + wrist_z**2)
        D_max = IK_L1 + IK_L2 - 1.0
        D_min = abs(IK_L1 - IK_L2) + 1.0
        
        if D > D_max or D < D_min:
            continue  # Try next pitch

        cos_theta2 = (D**2 - IK_L1**2 - IK_L2**2) / (2.0 * IK_L1 * IK_L2)
        cos_theta2 = float(np.clip(cos_theta2, -1.0, 1.0))
        alpha = math.atan2(wrist_z, wrist_x)

        valid_candidates = []
        for sign in (-1.0, +1.0):   # elbow-down then elbow-up
            t2 = sign * math.acos(cos_theta2)
            beta = math.atan2(IK_L2 * math.sin(t2),
                              IK_L1 + IK_L2 * math.cos(t2))
            t1 = alpha - beta
            
            # Reverse the FK equations to get servo commands
            m_lift  = 90.0 - math.degrees(t1)
            m_elbow = -math.degrees(t2) - 81.0
            m_wrist = math.degrees(t1 + t2 - pitch_rad) - 5.0
            
            # Hardware limits (widened slightly to allow valid IK before hardware self-limits)
            if not (-110 <= m_lift <= 150): continue
            if not (-120 <= m_elbow <= 120): continue
            if not (-120 <= m_wrist <= 120): continue
            
            # Prevent extreme backward leaning that hits the base.
            # Lift = -90° means pointing horizontally backwards.
            if m_lift < -90.0: continue

            valid_candidates.append({
                "shoulder_pan.pos":  pan_deg,
                "shoulder_lift.pos": m_lift,
                "elbow_flex.pos":    m_elbow,
                "wrist_flex.pos":    m_wrist,
                "wrist_roll.pos":    wrist_roll_deg,
                "gripper.pos":       60.0,
            })

        for c in valid_candidates:
            if current_joints is None:
                # If no current joints provided, return first valid
                print(f"   ✅ IK matched at pitch {test_pitch:.1f}°")
                return c
                
            # Otherwise find the lowest cost move
            cost = (3.0 * abs(c["shoulder_lift.pos"] - current_joints.get("shoulder_lift.pos", 0)) +
                    2.0 * abs(c["elbow_flex.pos"] - current_joints.get("elbow_flex.pos", 0)) +
                    1.0 * abs(c["wrist_flex.pos"] - current_joints.get("wrist_flex.pos", 0)))
            if cost < best_cost:
                best_cost = cost
                best_solution = c

        if best_solution is not None:
            # Found a valid configuration for this pitch
            print(f"   ✅ IK matched at pitch {test_pitch:.1f}°")
            return best_solution

    print(f"   ⚠️  IK reject: Target geometrically unreachable at any pitch.")
    return None


def surface_proximity_depth(cap, obj_px: int, obj_py: int,
                             inner_r: int = 35, outer_r: int = 80) -> float | None:
    """
    Sample depth in an annular ring around the ball centroid pixel to detect
    the surface (floor/table) that the ball is resting on.

    Returns the median depth (mm) of the surrounding surface, or None if not
    enough valid pixels are available.

    inner_r / outer_r: pixel radii defining the ring (excludes the ball itself).
    """
    with cap._lock:
        depth_img = cap._depth_img
        intr      = cap._intrinsics
        scale     = cap._depth_scale
    if depth_img is None or intr is None:
        return None

    h, w = depth_img.shape[:2]
    # Build coordinate grids relative to centroid
    ys, xs = np.ogrid[-outer_r:outer_r+1, -outer_r:outer_r+1]
    r2 = xs**2 + ys**2
    ring_mask = (r2 >= inner_r**2) & (r2 <= outer_r**2)

    # Absolute pixel positions
    row0 = max(0, obj_py - outer_r)
    col0 = max(0, obj_px - outer_r)
    row1 = min(h, obj_py + outer_r + 1)
    col1 = min(w, obj_px + outer_r + 1)

    rm_crop = ring_mask[
        (row0 - (obj_py - outer_r)):(row1 - (obj_py - outer_r)),
        (col0 - (obj_px - outer_r)):(col1 - (obj_px - outer_r))
    ]
    patch = depth_img[row0:row1, col0:col1]
    if patch.shape != rm_crop.shape:
        return None

    depths = patch[rm_crop].astype(float) * scale * 1000.0   # → mm
    valid = depths[(depths > D405_MIN_RANGE_MM) & (depths < MAX_GRAB_DEPTH_MM)]
    if valid.size < 20:
        return None
    return float(np.median(valid))


def depth_to_arm_target(xyz_cam: tuple[float, float, float],
                         robot) -> tuple[float, float, float] | None:
    """
    STEP 3 — Base Coordinate Transform (Camera → Wrist → Base).

    Implements the standard eye-in-hand matrix chain:

        P_base = T_wrist_base  ×  T_cam_wrist  ×  P_cam

    where:
      P_cam        = 3-D point in camera coordinates (mm, homogeneous)
      T_cam_wrist  = static calibration matrix (CAM_*_OFFSET_MM, measured once)
      T_wrist_base = live FK matrix from current servo angles
      P_base       = target position in arm base frame → fed to IK solver

    The approach depth is baked into P_cam by subtracting GRASP_PENETRATION_MM
    from the measured z so the gripper tip lands on/in the object surface.

    Returns (x_mm, y_mm, z_mm) in base frame (origin = shoulder pivot), or
    None if the object is inside the D405 70 mm minimum-range blind zone.
    """
    x_cam, y_cam, z_cam = xyz_cam

    # Guard: D405 cannot see objects closer than 70 mm
    if z_cam < D405_MIN_RANGE_MM + 10.0:
        print(f"   ⚠️  Object inside D405 blind zone ({z_cam:.0f} mm < "
              f"{D405_MIN_RANGE_MM} mm) — skipping")
        return None

    # Bake grasp approach penetration into the camera-space z coordinate.
    approach_z = z_cam + GRASP_PENETRATION_MM
    if approach_z < D405_MIN_RANGE_MM:
        approach_z = D405_MIN_RANGE_MM
        print(f"   ⚠️  Approach clamped to D405 min range")

    # Build P_cam as a homogeneous 4-vector (mm) in camera optical coordinates
    P_cam = np.array([x_cam, y_cam, approach_z, 1.0])

    # ── STEP 3a: Camera frame → Wrist frame ──────────────────────────────────
    # T_CAM_WRIST was built from your physical offset measurements at startup.
    P_wrist = T_CAM_WRIST @ P_cam

    # ── STEP 3b: Wrist frame → Base frame via live FK ─────────────────────────
    cur         = get_pos(robot)
    T_wrist_base = forward_kinematics(cur)   # 4×4 matrix from servo readings
    P_base      = T_wrist_base @ P_wrist

    x_base, y_base, z_base = P_base[0], P_base[1], P_base[2]
    print(f"   🔗 P_cam=({x_cam:+.0f},{y_cam:+.0f},{z_cam:.0f})mm → "
          f"P_wrist=({P_wrist[0]:+.0f},{P_wrist[1]:+.0f},{P_wrist[2]:.0f})mm → "
          f"P_base=({x_base:+.0f},{y_base:+.0f},{z_base:.0f})mm")
    return x_base, y_base, z_base


# ═══════════════════════════════════════════════════════════════════════════════
# Attribute Taxonomy for Moondream Semantic Verification
#
# Structure: category_name -> {
#     "words":    set of trigger words that activate this category,
#     "template": question template — use {syn} for the synonym string,
#     "synonyms": word -> [list of synonyms Moondream might accept],
# }
#
# Add new entries here to teach the system new attribute types.
# Any word in TARGET_DESC that matches a category's "words" set will trigger
# a separate yes/no question to Moondream for that attribute.
# ═══════════════════════════════════════════════════════════════════════════════
_ATTRIBUTE_TAXONOMY = {

    # ── Object type ────────────────────────────────────────────────────────────
    "object_type": {
        "words": {
            "ball", "sphere", "orb",
            "bottle", "flask", "jug", "container",
            "cup", "mug", "glass", "tumbler",
            "box", "carton", "cube", "block",
            "bowl", "plate", "dish",
            "book", "notebook", "folder", "binder",
            "phone", "remote", "controller",
            "toy", "figure", "doll",
            "shoe", "boot", "sneaker",
            "bag", "backpack", "purse", "wallet",
            "pillow", "cushion",
            "can", "tin",
            "pen", "pencil", "marker",
            "key", "keys",
            "cloth", "towel", "shirt", "sock", "glove",
            "plant", "pot",
            "apple", "orange", "banana",
            "stress",   # "stress ball" — treated as compound with "ball"
        },
        "template": "Is the main object in this image a {syn}? Answer with only the single word 'yes' or 'no'.",
        "synonyms": {
            "ball":       ["ball", "sphere", "orb", "round object", "globe"],
            "bottle":     ["bottle", "flask", "jug", "container"],
            "cup":        ["cup", "mug", "drinking glass", "tumbler"],
            "box":        ["box", "carton", "cube", "rectangular container"],
            "bowl":       ["bowl", "dish", "basin"],
            "book":       ["book", "notebook", "binder", "folder"],
            "phone":      ["phone", "smartphone", "mobile device", "cell phone"],
            "remote":     ["remote", "controller", "remote control"],
            "toy":        ["toy", "figurine", "doll", "action figure"],
            "shoe":       ["shoe", "boot", "sneaker", "footwear"],
            "bag":        ["bag", "backpack", "purse", "tote"],
            "pillow":     ["pillow", "cushion", "throw pillow"],
            "can":        ["can", "tin", "cylinder"],
            "pen":        ["pen", "pencil", "marker", "writing instrument"],
            "key":        ["key", "keys", "keychain"],
            "cloth":      ["cloth", "fabric", "textile", "piece of fabric"],
            "towel":      ["towel", "cloth", "fabric"],
            "shirt":      ["shirt", "top", "clothing item"],
            "sock":       ["sock", "stocking"],
            "glove":      ["glove", "hand covering"],
            "plant":      ["plant", "houseplant", "potted plant"],
            "stress":     ["stress ball", "foam ball", "squishy ball"],
        },
    },

    # ── Color ──────────────────────────────────────────────────────────────────
    "color": {
        "words": {
            "red", "orange", "yellow", "green", "blue", "purple", "violet",
            "pink", "black", "white", "grey", "gray", "brown", "cyan",
            "magenta", "gold", "silver", "beige", "teal", "navy", "olive",
            "neon", "multicolor", "colorful",
        },
        "template": "Is the main object in this image {syn}? Answer with only the single word 'yes' or 'no'.",
        "synonyms": {
            "grey":       ["grey", "gray"],
            "gray":       ["gray", "grey"],
            "multicolor": ["multicolored", "colorful", "has multiple colors"],
            "neon":       ["neon", "bright", "fluorescent"],
            "gold":       ["gold", "golden", "yellow-gold"],
            "silver":     ["silver", "metallic", "chrome"],
        },
    },

    # ── Size ───────────────────────────────────────────────────────────────────
    "size": {
        "words": {
            "tiny", "small", "little", "mini", "miniature",
            "medium", "mid",
            "large", "big", "huge", "giant", "enormous",
        },
        "template": "Is the main object in this image {syn}? Answer with only the single word 'yes' or 'no'.",
        "synonyms": {
            "tiny":       ["tiny", "very small", "miniature"],
            "small":      ["small", "little", "compact"],
            "little":     ["little", "small", "compact"],
            "mini":       ["mini", "miniature", "very small"],
            "miniature":  ["miniature", "tiny", "very small"],
            "medium":     ["medium-sized", "average size"],
            "mid":        ["medium-sized", "average size"],
            "large":      ["large", "big", "sizable"],
            "big":        ["big", "large", "sizable"],
            "huge":       ["huge", "very large", "oversized"],
            "giant":      ["giant", "very large", "enormous"],
            "enormous":   ["enormous", "very large", "gigantic"],
        },
    },

    # ── Texture / Surface Finish ───────────────────────────────────────────────
    "texture": {
        "words": {
            "shiny", "glossy", "reflective", "metallic", "polished",
            "matte", "dull", "flat",
            "rough", "textured", "bumpy", "coarse",
            "smooth",
            "soft", "fluffy", "fuzzy", "furry", "hairy",
            "hard", "rigid", "stiff",
            "squishy", "spongy", "foam", "foamy",
            "transparent", "translucent", "opaque", "clear",
            "wet", "dry",
        },
        "template": "Is the surface of the main object in this image {syn}? Answer with only the single word 'yes' or 'no'.",
        "synonyms": {
            "shiny":        ["shiny", "glossy", "reflective"],
            "glossy":       ["glossy", "shiny", "polished"],
            "reflective":   ["reflective", "shiny", "mirror-like"],
            "metallic":     ["metallic", "metal-looking", "shiny and metal"],
            "polished":     ["polished", "smooth and shiny"],
            "matte":        ["matte", "non-shiny", "flat finish"],
            "dull":         ["dull", "matte", "non-reflective"],
            "flat":         ["flat finish", "matte", "non-glossy"],
            "rough":        ["rough", "textured", "coarse"],
            "textured":     ["textured", "not smooth", "has texture"],
            "bumpy":        ["bumpy", "uneven", "has bumps"],
            "coarse":       ["coarse", "rough", "gritty"],
            "smooth":       ["smooth", "even", "flat surface"],
            "soft":         ["soft", "gentle", "compressible"],
            "fluffy":       ["fluffy", "soft and airy", "plush"],
            "fuzzy":        ["fuzzy", "soft", "covered in fuzz"],
            "furry":        ["furry", "has fur", "fluffy"],
            "hairy":        ["hairy", "covered in hair"],
            "hard":         ["hard", "rigid", "solid"],
            "rigid":        ["rigid", "stiff", "does not bend"],
            "stiff":        ["stiff", "rigid", "inflexible"],
            "squishy":      ["squishy", "soft and deformable", "compressible"],
            "spongy":       ["spongy", "foam-like", "compressible"],
            "foam":         ["foam", "spongy", "soft and light"],
            "foamy":        ["foamy", "foam-like", "spongy"],
            "transparent":  ["transparent", "see-through", "clear"],
            "translucent":  ["translucent", "semi-transparent", "partially see-through"],
            "opaque":       ["opaque", "not see-through", "solid color"],
            "clear":        ["clear", "transparent", "see-through"],
            "wet":          ["wet", "damp", "moist"],
            "dry":          ["dry", "not wet"],
        },
    },

    # ── Material ───────────────────────────────────────────────────────────────
    "material": {
        "words": {
            "plastic", "rubber", "silicone",
            "metal", "steel", "iron", "aluminum", "aluminium",
            "wooden", "wood",
            "glass",
            "ceramic", "porcelain",
            "paper", "cardboard",
            "fabric", "cloth", "cotton", "wool", "denim", "leather",
            "foam",
        },
        "template": "Is the main object in this image made of {syn}? Answer with only the single word 'yes' or 'no'.",
        "synonyms": {
            "plastic":    ["plastic", "synthetic material"],
            "rubber":     ["rubber", "latex", "elastic material"],
            "silicone":   ["silicone", "rubber-like material"],
            "metal":      ["metal", "metallic material"],
            "steel":      ["steel", "metal"],
            "iron":       ["iron", "metal"],
            "aluminum":   ["aluminum", "aluminium", "light metal"],
            "aluminium":  ["aluminium", "aluminum", "light metal"],
            "wooden":     ["wood", "wooden material"],
            "wood":       ["wood", "wooden material"],
            "glass":      ["glass", "transparent solid"],
            "ceramic":    ["ceramic", "pottery", "fired clay"],
            "porcelain":  ["porcelain", "ceramic", "fine china"],
            "paper":      ["paper"],
            "cardboard":  ["cardboard", "thick paper", "corrugated material"],
            "fabric":     ["fabric", "cloth", "textile"],
            "cloth":      ["cloth", "fabric", "textile"],
            "cotton":     ["cotton", "soft fabric"],
            "wool":       ["wool", "knitted material"],
            "denim":      ["denim", "jeans material", "jean fabric"],
            "leather":    ["leather", "animal hide"],
            "foam":       ["foam", "sponge-like material", "expanded polymer"],
        },
    },

    # ── State / Orientation ────────────────────────────────────────────────────
    "state": {
        "words": {
            "folded", "unfolded", "flat",
            "open", "closed", "shut",
            "full", "empty",
            "crumpled", "wrinkled", "creased",
            "rolled", "coiled",
            "stacked", "piled",
            "upright", "standing", "lying", "horizontal", "vertical",
            "on", "off",
            "broken", "cracked", "intact", "whole",
        },
        "template": "Is the main object in this image {syn}? Answer with only the single word 'yes' or 'no'.",
        "synonyms": {
            "folded":     ["folded", "has a fold", "creased over itself"],
            "unfolded":   ["unfolded", "flat", "opened out"],
            "flat":       ["flat", "lying flat", "not three-dimensional"],
            "open":       ["open", "opened up", "not closed"],
            "closed":     ["closed", "shut", "not open"],
            "shut":       ["shut", "closed"],
            "full":       ["full", "filled up", "not empty"],
            "empty":      ["empty", "hollow", "nothing inside"],
            "crumpled":   ["crumpled", "scrunched", "not flat"],
            "wrinkled":   ["wrinkled", "has wrinkles", "not smooth"],
            "creased":    ["creased", "has a crease", "has fold marks"],
            "rolled":     ["rolled up", "coiled", "tubular"],
            "coiled":     ["coiled", "wound up", "spiral"],
            "stacked":    ["stacked", "piled on top of each other"],
            "upright":    ["upright", "standing up", "vertical"],
            "standing":   ["standing", "upright", "vertical"],
            "lying":      ["lying down", "horizontal", "on its side"],
            "horizontal": ["horizontal", "lying flat", "on its side"],
            "vertical":   ["vertical", "upright", "standing"],
            "broken":     ["broken", "damaged", "cracked"],
            "cracked":    ["cracked", "has a crack", "damaged"],
            "intact":     ["intact", "undamaged", "whole"],
            "whole":      ["whole", "complete", "intact"],
        },
    },
}

# ── Fallback: words not matched by any category → treated as object noun ───────
def _word_to_attr(word: str) -> tuple[str, list[str], str] | None:
    """
    Look up a single word across all taxonomy categories.
    Returns (category_name, synonyms, question_template) or None if not found.
    """
    for cat_name, cat in _ATTRIBUTE_TAXONOMY.items():
        if word in cat["words"]:
            syns     = cat["synonyms"].get(word, [word])
            template = cat["template"]
            return cat_name, syns, template
    return None


def _parse_target_desc(desc: str) -> list[tuple[str, str, list[str], str]]:
    """
    Parse TARGET_DESC into a list of attribute checks.

    Returns a list of (category, word, synonyms, question_template) tuples,
    one per word. Words that don't match any category are treated as generic
    object nouns with a fallback question.

    Example — "small shiny red ball":
        [("size",        "small", ["small","little","compact"],      "Is ... {syn}?"),
         ("texture",     "shiny", ["shiny","glossy","reflective"],   "Is the surface ... {syn}?"),
         ("color",       "red",   ["red"],                           "Is ... {syn}?"),
         ("object_type", "ball",  ["ball","sphere","orb",...],        "Is ... a {syn}?")]
    """
    checks = []
    words  = desc.lower().split()
    for word in words:
        result = _word_to_attr(word)
        if result is not None:
            cat_name, syns, template = result
            checks.append((cat_name, word, syns, template))
        else:
            # Unknown word — ask as a generic noun
            checks.append((
                "object_type", word, [word],
                "Is the main object in this image a {syn}? Answer with only the single word 'yes' or 'no'.",
            ))
    return checks


def _moondream_ask(b64: str, url: str, question: str) -> str:
    """Fire one yes/no question at Moondream via /api/generate, print Q&A, return answer."""
    print(f"      Q: {question[:80]}..." if len(question) > 80 else f"      Q: {question}")
    # Use /api/generate — more compatible than /api/chat across Ollama versions.
    gen_url = OLLAMA_URL  # already points to /api/generate
    try:
        r    = requests.post(gen_url, json={
            "model":  "moondream",
            "prompt": question,
            "images": [b64],
            "stream": False,
        }, timeout=15)
        data = r.json()
    except Exception as e:
        print(f"      [HTTP error: {e}]")
        return ""

    if "response" not in data:
        print(f"      [Unexpected keys: {list(data.keys())}  raw: {str(data)[:120]}]")
        # Fallback: try chat-style key
        if "message" in data:
            return data["message"].get("content", "").strip().lower()
        return ""

    answer = data["response"].strip().lower()
    if not answer:
        print(f"      [Empty answer — full data: {str(data)[:200]}]")
    return answer


def _is_yes(answer: str) -> bool:
    """Return True if Moondream's answer clearly means yes."""
    return "yes" in answer and "no" not in answer.split("yes")[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Moondream verification — semantic attribute decomposition
# ═══════════════════════════════════════════════════════════════════════════════
def verify_moondream(crop: np.ndarray) -> bool:
    """
    Verifies a YOLO detection by asking Moondream one yes/no question per
    semantic attribute in TARGET_DESC.

    Supported attribute categories (auto-detected from TARGET_DESC words):
        color        -- red, blue, green, gold, neon...
        size         -- small, large, huge, tiny...
        texture      -- shiny, matte, rough, smooth, squishy, fluffy...
        material     -- plastic, rubber, wooden, metal, foam...
        state        -- folded, open, empty, crumpled, standing...
        object_type  -- ball, bottle, cup, box, book, shoe... (with synonyms)

    ALL checks must pass for the object to be accepted.

    Example -- TARGET_DESC = 'small squishy red ball':
        Q1 (size):        'Is the main object small or little or compact?'  -> yes
        Q2 (texture):     'Is the surface squishy or soft and deformable?'  -> yes
        Q3 (color):       'Is the main object red?'                         -> yes
        Q4 (object_type): 'Is the main object a ball or sphere or orb...?'  -> yes
        => PASS only if all four confirmed.
    """
    print(f"\n🧠 Moondream semantic check for '{TARGET_DESC}'...")
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64    = base64.b64encode(buf).decode()
    url    = OLLAMA_URL.replace("/api/generate", "/api/chat")

    checks = _parse_target_desc(TARGET_DESC)

    _CAT_ICON = {
        "color":       "🎨",
        "size":        "📐",
        "texture":     "✋",
        "material":    "🧱",
        "state":       "📂",
        "object_type": "🔍",
    }

    try:
        for cat_name, word, synonyms, template in checks:
            syn_str  = " or ".join(synonyms)
            question = template.format(syn=syn_str)
            answer   = _moondream_ask(b64, url, question)
            icon     = _CAT_ICON.get(cat_name, "?")
            print(f"   {icon} [{cat_name}:{word}] -> '{answer}'")
            if not _is_yes(answer):
                print(f"   x Rejected on '{word}' ({cat_name})")
                return False

        print(f"   OK All {len(checks)} attribute(s) confirmed -- it is a {TARGET_DESC}")
        return True

    except Exception as e:
        print(f"Moondream error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Canny-refined object centre
# ═══════════════════════════════════════════════════════════════════════════════
def canny_centre(frame: np.ndarray, x1, y1, x2, y2):
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return (x1 + x2) // 2, (y1 + y2) // 2
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    M     = cv2.moments(edges)
    if M["m00"] > 0:
        return x1 + int(M["m10"] / M["m00"]), y1 + int(M["m01"] / M["m00"])
    return (x1 + x2) // 2, (y1 + y2) // 2


# ═══════════════════════════════════════════════════════════════════════════════
# ObjectTracker: YOLO-primary, colour-mask fallback
# ═══════════════════════════════════════════════════════════════════════════════
class ObjectTracker:
    HSV_SLACK      = np.array([15, 60, 60], dtype=np.uint8)
    COLOR_MIN_AREA = 80
    SEARCH_PAD     = 40

    def __init__(self):
        self.lower       = None
        self.upper       = None
        self.last_box    = None
        self.last_center = None
        self.last_area   = None
        self.locked      = False

    def lock_on(self, frame, x1, y1, x2, y2):
        h_box      = y2 - y1
        sample_y2  = y1 + max(4, int(h_box * 0.40))
        crop       = frame[y1:sample_y2, x1:x2]
        if crop.size == 0:
            crop = frame[y1:y2, x1:x2]
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean = hsv.mean(axis=(0, 1)).astype(np.float32)
        self.lower = np.clip(mean - self.HSV_SLACK,
                             [0, 20, 20], [180, 255, 255]).astype(np.uint8)
        self.upper = np.clip(mean + self.HSV_SLACK,
                             [0, 20, 20], [180, 255, 255]).astype(np.uint8)
        fh, fw     = frame.shape[:2]
        self.last_box    = (x1, y1, x2, y2)
        self.last_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.last_area   = ((x2 - x1) / fw) * ((y2 - y1) / fh)
        self.locked      = True
        print(f"   🔒 Tracker locked: HSV [{self.lower}]–[{self.upper}]")

    def detect(self, frame, yolo_results, model_class_id):
        fh, fw = frame.shape[:2]
        # YOLO first
        for box in yolo_results[0].boxes:
            if int(box.cls[0].item()) != model_class_id:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw, x2), min(fh, y2)
            cx, cy  = canny_centre(frame, x1, y1, x2, y2)
            area    = ((x2 - x1) / fw) * ((y2 - y1) / fh)
            self.last_box = (x1, y1, x2, y2)
            self.last_center = (cx, cy)
            self.last_area   = area
            return True, (cx, cy), area, "yolo"
        # Colour fallback
        if not (self.locked and self.lower is not None and self.last_box):
            return False, self.last_center, self.last_area, "lost"
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        lx1, ly1, lx2, ly2 = self.last_box
        p    = self.SEARCH_PAD
        rx1  = max(0, lx1 - p);  ry1 = max(0, ly1 - p)
        rx2  = min(fw, lx2 + p); ry2 = min(fh, ly2 + p)
        roi  = np.zeros_like(mask)
        roi[ry1:ry2, rx1:rx2] = mask[ry1:ry2, rx1:rx2]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, self.last_center, self.last_area, "lost"
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.COLOR_MIN_AREA:
            return False, self.last_center, self.last_area, "lost"
        M = cv2.moments(largest)
        if M["m00"] <= 0:
            return False, self.last_center, self.last_area, "lost"
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        bx, by, bw, bh = cv2.boundingRect(largest)
        area = (bw / fw) * (bh / fh)
        overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        frame[:] = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        self.last_center = (cx, cy)
        self.last_area   = area
        return True, (cx, cy), area, "color"


# ═══════════════════════════════════════════════════════════════════════════════
# Visual servoing alignment loop
# ═══════════════════════════════════════════════════════════════════════════════
def align_arm(robot, cap: RealSenseStream, model,
              tracker: ObjectTracker,
              max_frames: int | None = None) -> tuple[bool, tuple[int, int] | None]:
    """
    Closed-loop visual servoing. Returns (success, (obj_px, obj_py)).
    obj_px/py is the final pixel coordinate of the object centre.

    max_frames: override for ALIGN_MAX_FRAMES (useful for quick re-centre passes).
    """
    centred_streak = 0
    lost_streak    = 0
    sign_flip_pan  = 0
    last_pan_sign  = 0
    last_obj_pixel = None
    effective_max  = max_frames if max_frames is not None else ALIGN_MAX_FRAMES

    # ── Record initial posture ───────────────────────────────────────────────
    # We freeze elbow, wrist, and gripper at their initial values for the duration
    # of the alignment step. This prevents them from drifting more negative (stretching out)
    # due to gravity sag being fed back into the position command.
    initial_pos = get_pos(robot)
    start_lift  = initial_pos.get("shoulder_lift.pos", 0.0)
    start_elb   = initial_pos.get("elbow_flex.pos", 0.0)
    start_wst   = initial_pos.get("wrist_flex.pos", 0.0)
    start_grp   = initial_pos.get("gripper.pos", 60.0)

    for frame_idx in range(effective_max):
        color, has_depth, depth_colormap = cap.read()
        h, w     = color.shape[:2]
        frame_cx = (w // 2) + ALIGN_PAN_OFFSET
        frame_cy =  h // 2

        results = model(color, verbose=False)
        found, center, area, src = tracker.detect(color, results, YOLO_CLASS_ID)

        display = color.copy()
        cv2.line(display, (frame_cx-20, frame_cy), (frame_cx+20, frame_cy), (0,0,255), 1)
        cv2.line(display, (frame_cx, frame_cy-20), (frame_cx, frame_cy+20), (0,0,255), 1)
        cv2.circle(display, (frame_cx, frame_cy), 6, (0,0,255), -1)

        if not found:
            lost_streak += 1
            cv2.putText(display,
                f"🎯 ALIGNING — lost ({lost_streak}/{ALIGN_LOST_GRACE})",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,100,255), 2)
            cv2.imshow("Picker Vision", display)
            cv2.waitKey(1)
            if lost_streak >= ALIGN_LOST_GRACE:
                if last_obj_pixel is not None:
                    last_pan_err = last_obj_pixel[0] - frame_cx
                    last_lift_err = last_obj_pixel[1] - frame_cy
                    if abs(last_pan_err) <= 60 and abs(last_lift_err) <= 60:
                        print(f"   🟡 Object lost, but last pos was close enough (err={last_pan_err:+d},{last_lift_err:+d}) — proceeding to grab")
                        return True, last_obj_pixel
                print(f"   ❌ Lost too long ({frame_idx+1} frames) — restarting search")
                return False, None
            time.sleep(0.03)
            continue

        lost_streak = 0
        obj_px, obj_py = center
        last_obj_pixel = (obj_px, obj_py)

        pan_err  = obj_px - frame_cx
        lift_err = obj_py - frame_cy

        cur_pan_sign = 1 if pan_err > 0 else (-1 if pan_err < 0 else 0)
        if last_pan_sign != 0 and cur_pan_sign != 0 and cur_pan_sign != last_pan_sign:
            sign_flip_pan += 1
        last_pan_sign = cur_pan_sign

        # Non-linear exponential decay as we approach the target to prevent overshoot
        err_mag = math.hypot(pan_err, lift_err)
        decay = max(0.1, 1.0 - math.exp(-err_mag / 30.0))
        
        # NOTE: pan_err sign was originally correct (it oscillated around 0, proving it was a stable closed loop).
        pan_cmd  = float(np.clip(pan_err * ALIGN_PAN_K * decay,  -ALIGN_MAX_PAN_DEG,  ALIGN_MAX_PAN_DEG))
        lift_cmd = float(np.clip(lift_err * ALIGN_LIFT_K * decay, -ALIGN_MAX_LIFT_DEG, ALIGN_MAX_LIFT_DEG))

        cur = get_pos(robot)
        
        # ── Prevent backward tilt (more negative lift) ───────────────────────
        # User constraint: Lift cannot become more negative than its starting position,
        # otherwise the arm tilts back.
        next_lift = cur["shoulder_lift.pos"] + lift_cmd
        hit_lift_constraint = False
        if next_lift < start_lift:
            next_lift = start_lift
            lift_cmd = start_lift - cur["shoulder_lift.pos"]
            hit_lift_constraint = True

        centred_pan  = abs(pan_err) < ALIGN_THRESHOLD
        centred_lift = (abs(lift_err) < ALIGN_THRESHOLD) or hit_lift_constraint
        centred = centred_pan and centred_lift

        if centred:
            centred_streak += 1
            if centred_streak >= ALIGN_CENTRED_NEED:
                print(f"   ✅ Aligned in {frame_idx+1} frames (pan={pan_err}px, lift={lift_err}px)")
                return True, last_obj_pixel
        else:
            centred_streak = 0

        robot.send_action({
            "shoulder_pan.pos":  cur["shoulder_pan.pos"] + pan_cmd,
            "shoulder_lift.pos": next_lift,
            "elbow_flex.pos":    start_elb,
            "wrist_flex.pos":    start_wst,
            "gripper.pos":       start_grp,
        })
        
        # Format the command printout to reflect actual movement (clamped)
        print(f"   [{src.upper():5s}] frame {frame_idx+1:03d}: "
              f"err=({pan_err:+4d},{lift_err:+4d})px  "
              f"cmd=({pan_cmd:+5.2f},{lift_cmd:+5.2f})°")

        src_color = (0,255,0) if src == "yolo" else (0,165,255)
        cv2.circle(display, (obj_px, obj_py), 7, src_color, -1)
        cv2.line(display, (frame_cx, frame_cy), (obj_px, obj_py), (0,255,255), 1)
        status = "✅ CENTRED" if centred else "🎯 ALIGNING"
        cv2.putText(display,
            f"{status} [{src.upper()}]  err=({pan_err:+d},{lift_err:+d})px",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        cv2.imshow("Picker Vision", display)
        cv2.waitKey(1)

    if last_obj_pixel is not None:
        print(f"   ⏳ Alignment hardcap reached ({effective_max} frames). Proceeding with best alignment.")
        return True, last_obj_pixel
    else:
        print(f"   ❌ Alignment timed out and no object found — restarting")
        return False, None


def _print_tuning_hints(sign_flip_count: int):
    print("\n📐 ALIGNMENT TUNING HINTS:")
    if sign_flip_count >= 3:
        print(f"   ⚡ {sign_flip_count} oscillations → decrease ALIGN_PAN_K ({ALIGN_PAN_K})")
    else:
        print(f"   🐢 No oscillation → increase ALIGN_PAN_K ({ALIGN_PAN_K})")


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration & Workspace Limit Utilities
# ═══════════════════════════════════════════════════════════════════════════════
def show_workspace_limits():
    """Calculates and prints theoretical arm workspace limits based on IK constants."""
    print("\n" + "═"*65)
    print(" 📐 THEORETICAL ARM WORKSPACE LIMITS & EDGE BOUNDARIES")
    print("═"*65)
    L1, L2, L3 = IK_L1, IK_L2, IK_L3
    d_max = L1 + L2
    d_min = abs(L1 - L2)
    ee_max = L1 + L2 + L3
    print(f" Arm Link Lengths:")
    print(f"   L1 (Shoulder pivot → Elbow pivot) : {L1:5.1f} mm ({L1/10:4.1f} cm)")
    print(f"   L2 (Elbow pivot → Wrist pivot)    : {L2:5.1f} mm ({L2/10:4.1f} cm)")
    print(f"   L3 (Wrist pivot → Gripper tip)    : {L3:5.1f} mm ({L3/10:4.1f} cm)")
    print(f"\n Extension Distance Envelope (Wrist Pivot D = √(X² + Z²)):")
    print(f"   Minimum Wrist Distance (D_min)   : {d_min:5.1f} mm ({d_min/10:4.1f} cm)")
    print(f"   Maximum Wrist Distance (D_max)   : {d_max:5.1f} mm ({d_max/10:4.1f} cm)")
    print(f"   Maximum Total Reach (to Tip)     : {ee_max:5.1f} mm ({ee_max/10:4.1f} cm)")
    print(f"   Shoulder Height above Table      : {IK_SHOULDER_HEIGHT_MM:5.1f} mm")

    print("\n Edge Reach Boundaries at sample pitch angles:")
    for pitch in [0, -30, -45, -90]:
        rad = math.radians(pitch)
        w_max_x = (d_max - 1.0) - L3 * math.cos(rad)
        w_max_z = - L3 * math.sin(rad)
        tip_x   = w_max_x + L3 * math.cos(rad)
        tip_z   = w_max_z + L3 * math.sin(rad)
        print(f"   Pitch {pitch:3d}° → Max Wrist Target: X={w_max_x:5.1f}mm | Total Reach Tip: X={tip_x:5.1f}mm, Z={tip_z:5.1f}mm")
    print("═"*65 + "\n")


def demonstrate_workspace_boundaries(robot):
    """
    Physically moves the arm to key workspace boundaries/edges
    so the user can visually observe the physical reach limits in action.
    """
    print("\n" + "═"*65)
    print(" 🦾 PHYSICAL WORKSPACE BOUNDARY DEMONSTRATION MODE")
    print("═"*65)
    print(" The arm will smoothly move to each physical boundary edge.")
    print(" Pauses at each edge for 2 seconds so you can visually inspect reach.")
    print(" Press Ctrl+C at any time to abort and return to start position.")
    print("═"*65 + "\n")

    boundaries = [
        ("Straight Out (Image 1)",
         "Arm extended forward (Pitch -61°)",
         (204.9, 0.0, -201.4, -61.2)),

        ("Max Forward Floor Reach (Image 2)",
         "Arm extended far forward, touching the floor (Pitch -43°)",
         (260.2, 0.0, -199.4, -43.4)),

        ("Far Left - Floor Reach (Image 4)",
         "Arm reaching far left across 180° profile (Pitch -101°)",
         (48.0, -152.5, -232.2, -101.1)),

        ("Max Vertical Reach (Image 6)",
         "Arm extended straight up overhead (Pitch 3.6°)",
         (67.8, 0.0, 256.3, 3.6)),

        ("Minimum Retracted Reach",
         "Arm folded in close to base",
         (100.0, 0.0, 0.0, -30.0)),
    ]

    START_POS = dict(_BASE)

    try:
        print("▶ Moving to Start Position...")
        smooth_move(robot, START_POS, step_size=2.0, step_delay=0.03)
        time.sleep(1.0)

        for idx, (title, desc, spec) in enumerate(boundaries, 1):
            print(f"\n📍 Boundary [{idx}/{len(boundaries)}]: {title}")
            print(f"   Description: {desc}")

            x, y, z, pitch = spec
            target_pos = solve_ik(x, y, z, end_pitch_deg=pitch)

            if target_pos is None:
                print(f"   ⚠️  Target (X={x:.0f}, Y={y:.0f}, Z={z:.0f}) unreachable — skipping")
                continue

            print(f"   🦾 Moving to: X={x:+.1f}mm, Y={y:+.1f}mm, Z={z:+.1f}mm | Pitch={pitch}°")
            print(f"   Joints: Pan={target_pos['shoulder_pan.pos']:+.1f}°, Lift={target_pos['shoulder_lift.pos']:+.1f}°, "
                  f"Elb={target_pos['elbow_flex.pos']:+.1f}°, Wst={target_pos['wrist_flex.pos']:+.1f}°")

            smooth_move(robot, target_pos, step_size=1.5, step_delay=0.03)
            time.sleep(2.0)

        print("\n✅ Boundary demonstration complete! Returning to Start Position...")
        smooth_move(robot, START_POS, step_size=2.0, step_delay=0.03)
        time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n⏹️  Demonstration aborted by user. Returning to start...")
        smooth_move(robot, START_POS, step_size=3.0, step_delay=0.02)


def demonstrate_raw_image_positions(robot):
    """
    Moves the arm to the EXACT joint angles captured in the user's calibration images,
    bypassing IK completely.
    """
    positions = {
        "1": ("Image 1 (Straight Out)", {
            "shoulder_pan.pos": -4.6, "shoulder_lift.pos": 86.6,
            "elbow_flex.pos": -80.5, "wrist_flex.pos": -8.9, "gripper.pos": 60.0
        }),
        "2": ("Image 2 (Touching floor, straight out)", {
            "shoulder_pan.pos": -6.4, "shoulder_lift.pos": 104.2,
            "elbow_flex.pos": -38.5, "wrist_flex.pos": -9.3, "gripper.pos": 60.0
        }),
        "3": ("Image 3 (Edge of main robot's frame)", {
            "shoulder_pan.pos": -3.8, "shoulder_lift.pos": 81.4,
            "elbow_flex.pos": 67.6, "wrist_flex.pos": -61.3, "gripper.pos": 60.0
        }),
        "4": ("Image 4 (Far left across 180° - floor)", {
            "shoulder_pan.pos": -72.5, "shoulder_lift.pos": 86.6,
            "elbow_flex.pos": -50.7, "wrist_flex.pos": 39.1, "gripper.pos": 60.0
        }),
        "5": ("Image 5 (Far left - level with frame)", {
            "shoulder_pan.pos": -118.4, "shoulder_lift.pos": 86.6,
            "elbow_flex.pos": -34.0, "wrist_flex.pos": -63.0, "gripper.pos": 60.0
        }),
        "6": ("Image 6 (Straight up)", {
            "shoulder_pan.pos": -7.1, "shoulder_lift.pos": -3.0,
            "elbow_flex.pos": -82.0, "wrist_flex.pos": -1.1, "gripper.pos": 60.0
        }),
        "7": ("Default Scan Posture (_BASE)", _BASE),
        "8": ("Image 8 (Far right across 180° - floor)  [mirror of Image 4]", {
            "shoulder_pan.pos": +72.5, "shoulder_lift.pos": 86.6,
            "elbow_flex.pos": -50.7, "wrist_flex.pos": 39.1, "gripper.pos": 60.0
        }),
        "9": ("Image 9 (Far right - level with frame)  [mirror of Image 5]", {
            "shoulder_pan.pos": +113.8, "shoulder_lift.pos": 86.6,
            "elbow_flex.pos": -34.0, "wrist_flex.pos": -63.0, "gripper.pos": 60.0
        }),
    }

    print("\n" + "═"*65)
    print(" 📸 EXACT IMAGE CALIBRATION POSTURES (RAW JOINTS)")
    print("═"*65)
    for key, (name, _) in positions.items():
        print(f"   [{key}] {name}")
    print("   [0] Return to Main Menu")
    print("═"*65)
    
    while True:
        choice = input("\nSelect a posture to move to (0-9): ").strip()
        if choice == "0":
            break
        if choice in positions:
            name, target_pos = positions[choice]
            print(f"▶ Moving to {name}...")
            print(f"   Joints: Pan={target_pos.get('shoulder_pan.pos',0):.1f}°, Lift={target_pos.get('shoulder_lift.pos',0):.1f}°, "
                  f"Elb={target_pos.get('elbow_flex.pos',0):.1f}°, Wst={target_pos.get('wrist_flex.pos',0):.1f}°")
            smooth_move(robot, target_pos, step_size=2.0, step_delay=0.03)
        else:
            print("Invalid choice. Enter 0-9.")


def run_manual_calibration(robot):
    """
    Cuts motor torque, allowing the user to move the arm manually by hand.
    Continuously measures joint angles, calculates FK (X, Y, Z), tracks
    minimum/maximum joint angles and maximum extension reached.
    """
    print("\n" + "═"*65)
    print(" 🛠️  MANUAL MOVEMENT & RANGE-OF-MOTION CALIBRATION MODE")
    print("═"*65)
    print(" ▶ First moving to the Default Scan Posture...")
    smooth_move(robot, _BASE, step_size=2.0, step_delay=0.03)
    time.sleep(1.0)
    
    print("\n Disabling motor torque... You can now move the arm manually by hand.")
    print(" Practice the perfect lunge from this starting position!")
    print(" Hold the arm exactly where you want the lunge to end, then press Ctrl+C in terminal.")
    print("═"*65 + "\n")

    _set_torque(robot, False)

    stats = {
        "shoulder_pan":  {"min": 999.0, "max": -999.0},
        "shoulder_lift": {"min": 999.0, "max": -999.0},
        "elbow_flex":    {"min": 999.0, "max": -999.0},
        "wrist_flex":    {"min": 999.0, "max": -999.0},
        "wrist_roll":    {"min": 999.0, "max": -999.0},
        "gripper":       {"min": 999.0, "max": -999.0},
    }
    max_reach_base = 0.0
    max_reach_pos  = None

    try:
        while True:
            pos = get_pos(robot)
            for j_key, val in pos.items():
                j_name = j_key.split(".")[0]
                if j_name in stats:
                    stats[j_name]["min"] = min(stats[j_name]["min"], val)
                    stats[j_name]["max"] = max(stats[j_name]["max"], val)

            T_wb = forward_kinematics(pos)
            wx, wy, wz = T_wb[0, 3], T_wb[1, 3], T_wb[2, 3]
            d_dist = math.sqrt(wx**2 + wy**2 + wz**2)

            if d_dist > max_reach_base:
                max_reach_base = d_dist
                max_reach_pos  = (wx, wy, wz, dict(pos))

            print(f"\r📍 Live Pos: X={wx:+6.1f}mm Y={wy:+6.1f}mm Z={wz:+6.1f}mm | Reach D={d_dist:5.1f}mm | "
                  f"Pan={pos.get('shoulder_pan.pos',0):+5.1f}° Lift={pos.get('shoulder_lift.pos',0):+5.1f}° "
                  f"Elb={pos.get('elbow_flex.pos',0):+5.1f}° Wst={pos.get('wrist_flex.pos',0):+5.1f}°",
                  end="", flush=True)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\n" + "═"*65)
        print(" 📊 MANUAL CALIBRATION RESULTS SUMMARY")
        print("═"*65)
        print(" Recorded Joint Angle Limits:")
        for j_name, s in stats.items():
            if s["min"] != 999.0:
                print(f"   {j_name:15s}: Min = {s['min']:+6.1f}°  |  Max = {s['max']:+6.1f}°  | Range = {s['max']-s['min']:6.1f}°")

        print(f"\n Maximum Extension Reached (Wrist Pivot in Arm Base Frame):")
        print(f"   Max Distance D = {max_reach_base:.1f} mm ({max_reach_base/10:.1f} cm)")
        if max_reach_pos:
            mx, my, mz, mpos = max_reach_pos
            print(f"   At Coordinates : X={mx:+.1f}mm, Y={my:+.1f}mm, Z={mz:+.1f}mm")
            print(f"   At Joint Angles: Pan={mpos.get('shoulder_pan.pos',0):.1f}°, Lift={mpos.get('shoulder_lift.pos',0):.1f}°, "
                  f"Elb={mpos.get('elbow_flex.pos',0):.1f}°, Wst={mpos.get('wrist_flex.pos',0):.1f}°")
        print("═"*65 + "\n")
    finally:
        print("🔌 Restoring motor torque...")
        _set_torque(robot, True)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "═"*65)
    print(" 🤖 SO-ARM101 CONTROLLER & WORKSPACE CALIBRATION TOOL")
    print("═"*65)
    print(" Select Mode:")
    print("   [1] Vision Alignment + Manual Lunge Demonstration")
    print("   [2] View Theoretical Arm Workspace Limits & Edge Boundaries")
    print("   [3] Manual Arm Movement & Range-of-Motion Calibration (Torque OFF)")
    print("   [4] Move to Exact Calibrated Image Postures (Raw Joints)")
    print("   [5] Test Full Automatic Pick-and-Place (Vision + IK Lunge)")
    print("   [6] Exit")
    print("═"*65)
    choice = input("Enter choice (1-5): ").strip()

    if choice == "2":
        show_workspace_limits()
        do_move = input("Do you want to physically move the arm to these boundary edges now? (y/n): ").strip().lower()
        if do_move.startswith("y"):
            print("🔌 Connecting to SO-ARM101...")
            config = SOFollowerRobotConfig(port=PORT, id=ARM_ID, use_degrees=True)
            robot  = SOFollower(config)
            robot.connect()
            try:
                demonstrate_workspace_boundaries(robot)
            finally:
                robot.disconnect()
        return
    elif choice == "3":
        print("🔌 Connecting to SO-ARM101...")
        config = SOFollowerRobotConfig(port=PORT, id=ARM_ID, use_degrees=True)
        robot  = SOFollower(config)
        robot.connect()
        try:
            run_manual_calibration(robot)
        finally:
            robot.disconnect()
        return
    elif choice == "4":
        print("🔌 Connecting to SO-ARM101...")
        config = SOFollowerRobotConfig(port=PORT, id=ARM_ID, use_degrees=True)
        robot  = SOFollower(config)
        robot.connect()
        try:
            demonstrate_raw_image_positions(robot)
        finally:
            robot.disconnect()
        return
    elif choice == "6":
        print("Exiting.")
        return
    elif choice not in ["1", "5"]:
        print("Invalid selection. Exiting.")
        return
        
    do_manual_lunge = (choice == "1")

    print("🚀 Initialising RealSense D405 + IK Pick-and-Place Pipeline...")

    global T_CAM_WRIST
    T_CAM_WRIST = _build_T_cam_wrist()
    print(f"📐 T_cam_wrist built:")
    print(f"   offsets X={CAM_X_OFFSET_MM}mm  Y={CAM_Y_OFFSET_MM}mm  Z={CAM_Z_OFFSET_MM}mm  pitch={CAM_PITCH_DEG}°")

    global TARGET_DESC, YOLO_CLASS_ID
    target_in = input(f"\n🎯 Enter object to grab (e.g. 'bottle', 'cup', 'red ball') [default '{TARGET_DESC}']: ").strip()
    if target_in:
        TARGET_DESC = target_in

    # Resolve model path (prefer yolov8s-worldv2.pt)
    model_paths = [
        "/root/ros2_ws/models/yolov8s-worldv2.pt",
        os.path.join(os.path.dirname(__file__), "..", "models", "yolov8s-worldv2.pt"),
        os.path.join(os.path.dirname(__file__), "models", "yolov8s-worldv2.pt"),
        "models/yolov8s-worldv2.pt",
        "yolov8s-worldv2.pt",
        "yolov8n-seg.pt",
        "yolo11n-seg.pt"
    ]
    chosen_path = None
    for mp in model_paths:
        if os.path.exists(mp):
            chosen_path = mp
            break
    if chosen_path is None:
        chosen_path = "/root/ros2_ws/models/yolov8s-worldv2.pt"

    print(f"Loading YOLO model: {chosen_path}...")
    model = YOLO(chosen_path)
    if "world" in chosen_path.lower() or hasattr(model, "set_classes"):
        model.set_classes([TARGET_DESC])
        YOLO_CLASS_ID = 0  # In YOLO-World, the single custom class is index 0
        print(f"   🌍 YOLO-World loaded and targeting: ['{TARGET_DESC}'] (Class ID: {YOLO_CLASS_ID})")
    else:
        print(f"   🎯 Standard YOLO targeting class {YOLO_CLASS_ID} ('{TARGET_DESC}')")

    # ── Robot arm ─────────────────────────────────────────────────────────────
    print("🔌 Connecting to SO-ARM101...")
    config = SOFollowerRobotConfig(port=PORT, id=ARM_ID, use_degrees=True)
    robot  = SOFollower(config)
    robot.connect()
    print("   ✅ Arm connected")

    START_POS = dict(_BASE)
    STOW      = dict(_STOW_BASE)

    def handle_exit(sig, frame):
        print("\n📍 Stowing arm...")
        smooth_move(robot, STOW)
        time.sleep(1)
        robot.disconnect()
        exit(0)
    signal.signal(signal.SIGINT, handle_exit)

    # ── RealSense D405 ────────────────────────────────────────────────────────
    print("📷 Connecting to RealSense D405...")
    cap = RealSenseStream()
    time.sleep(2.0)   # let first frames arrive and intrinsics populate

    # ── Move to start ─────────────────────────────────────────────────────────
    print("\n▶ Moving to Start Position...")
    smooth_move(robot, START_POS)
    time.sleep(1.0)

    last_yolo_t = 0.0
    STATE       = "SEARCHING"
    sweep_dir   = 1.0
    sweep_pan   = START_POS["shoulder_pan.pos"]

    # ── Emergency stow ────────────────────────────────────────────────────────
    def _emergency_stow():
        print("\n⚠️  Emergency stow triggered...")
        try:
            smooth_move(robot, STOW, step_size=3.0)
            time.sleep(0.5)
            robot.disconnect()
        except Exception as e:
            print(f"   Stow error: {e}")
        try:
            cap.stop()
            cv2.destroyAllWindows()
        except Exception:
            pass
    atexit.register(_emergency_stow)

    try:
        while True:
            color, has_depth, depth_colormap = cap.read()
            if color is None or not has_depth:
                time.sleep(0.01)
                continue
            h, w = color.shape[:2]
            frame_cx, frame_cy = w // 2, h // 2

            # ── HUD base ──────────────────────────────────────────────────────
            display = color.copy()
            cv2.line(display, (frame_cx-20, frame_cy), (frame_cx+20, frame_cy), (0,0,255), 1)
            cv2.line(display, (frame_cx, frame_cy-20), (frame_cx, frame_cy+20), (0,0,255), 1)
            cv2.circle(display, (frame_cx, frame_cy), 6, (0,0,255), -1)

            cv2.putText(display, f"[{STATE}]",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,140,255), 2)

            # ── Depth overlay: show centre-pixel depth ─────────────────────────
            xyz_centre = cap.get_xyz(frame_cx, frame_cy)
            if xyz_centre is not None:
                d_mm = xyz_centre[2]
                depth_color = (0, 200, 0) if d_mm > D405_MIN_RANGE_MM else (0, 0, 255)
                cv2.putText(display, f"depth: {d_mm:.0f}mm",
                            (w - 160, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, depth_color, 2)

            if STATE == "SEARCHING":
                cv2.putText(display, f"STATE: {STATE} | YOLO: {TARGET_DESC}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # ── Throttle YOLO to ~5 fps during search ─────────────────────────
            if time.time() - last_yolo_t < 0.2:
                cv2.imshow("Picker Vision", np.hstack((display, depth_colormap)))
                cv2.waitKey(1)
                continue
            last_yolo_t = time.time()

            results = model(color, verbose=False, conf=0.5)
            
            # ── Draw YOLO detections ──────────────────────────────────────────
            target_box     = None
            target_box_idx = -1
            for i, box in enumerate(results[0].boxes):
                if int(box.cls[0].item()) == YOLO_CLASS_ID:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                    
                    # Check if the bounding box has ANY valid depth inside it
                    cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
                    box_w, box_h = max(10, bx2 - bx1), max(10, by2 - by1)
                    
                    if cap.get_xyz(cx, cy, search_w=box_w//2, search_h=box_h//2) is None:
                        cv2.rectangle(display, (bx1, by1), (bx2, by2), (0,0,255), 2)
                        cv2.putText(display, "OUT OF DEPTH FOV", (bx1, by1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                        continue

                    target_box     = box
                    target_box_idx = i
                    cv2.rectangle(display, (bx1, by1), (bx2, by2), (0,255,0), 2)
                    cv2.putText(display, f"YOLO: {TARGET_DESC}", (bx1, by1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    break

            # ── No detection — sweep ──────────────────────────────────────────
            if target_box is None:
                if STATE == "SEARCHING":
                    sweep_pan += sweep_dir * SEARCH_SWEEP_SPEED
                    pan_limit_right = START_POS["shoulder_pan.pos"] + SEARCH_SWEEP_RANGE
                    pan_limit_left  = START_POS["shoulder_pan.pos"] - SEARCH_SWEEP_RANGE
                    if sweep_pan >= pan_limit_right:
                        sweep_pan = pan_limit_right;  sweep_dir = -1.0
                    elif sweep_pan <= pan_limit_left:
                        sweep_pan = pan_limit_left;   sweep_dir =  1.0
                    sweep_cmd = dict(START_POS)
                    sweep_cmd["shoulder_pan.pos"] = sweep_pan
                    robot.send_action(sweep_cmd)
                
                cv2.imshow("Picker Vision", np.hstack((display, depth_colormap)))
                cv2.waitKey(1)
                continue

            # ── YOLO candidate found ──────────────────────────────────────────
            x1, y1, x2, y2 = map(int, target_box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Use segmentation mask centroid if available (more accurate than bbox centre)
            seg_mask = None
            if (results[0].masks is not None and
                    target_box_idx >= 0 and
                    target_box_idx < len(results[0].masks.xy)):
                polygon  = results[0].masks.xy[target_box_idx].astype(np.int32)
                seg_mask = np.zeros(color.shape[:2], dtype=np.uint8)
                cv2.fillPoly(seg_mask, [polygon], 255)
                # Centroid from mask moments
                Mm = cv2.moments(seg_mask)
                if Mm["m00"] > 0:
                    obj_px = int(Mm["m10"] / Mm["m00"])
                    obj_py = int(Mm["m01"] / Mm["m00"])
                else:
                    obj_px, obj_py = canny_centre(color, x1, y1, x2, y2)
            else:
                obj_px, obj_py = canny_centre(color, x1, y1, x2, y2)

            cv2.rectangle(display, (x1, y1), (x2, y2), (0,255,255), 2)
            cv2.circle(display, (obj_px, obj_py), 6, (0,255,0), -1)
            cv2.putText(display, f"YOLO: {TARGET_DESC}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            
            cv2.imshow("Picker Vision", np.hstack((display, depth_colormap)))
            cv2.waitKey(1)

            if SKIP_MOONDREAM:
                print("⏭️  Moondream skipped — grabbing on YOLO detection")
            else:
                # Give Moondream more context so it doesn't hallucinate
                cy1 = max(0, y1-50);  cy2 = min(h, y2+50)
                cx1 = max(0, x1-50);  cx2 = min(w, x2+50)
                crop = color[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                STATE = "VERIFYING"
                if not verify_moondream(crop):
                    print("❌ Moondream rejected — resuming search")
                    STATE = "SEARCHING"
                    time.sleep(0.5)
                    continue

                print("✅ Moondream confirmed! Initiating grab...")

            # ── Step 1: Align in scan pose (pan/lift to centre ball) ──────────
            # Run visual servoing FIRST while the arm is still in the retracted
            # scan pose.  This centres the ball in the camera frame and gives us
            # the exact confirmed pan angle before the arm lunges forward.
            print("🎯 Aligning arm to ball centre (scan pose)...")
            tracker = ObjectTracker()
            tracker.lock_on(color, x1, y1, x2, y2)
            aligned, final_pixel = align_arm(robot, cap, model, tracker)
            if not aligned or final_pixel is None:
                print("❌ Alignment failed — restarting search")
                smooth_move(robot, START_POS, step_size=2.0, step_delay=0.03)
                STATE = "SEARCHING"
                continue

            obj_px, obj_py = final_pixel

            print("✅ Ball centred — calculating direct trajectory...")
            STATE = "GRABBING"
            print("\n🎯 Reading depth for IK from scan posture...")

            # ── IK-based grab ─────────────────────────────────────────────────
            STATE = "GRABBING"
            print("\n🎯 Reading depth for IK...")


            # Take a fresh depth snapshot to ensure we have the latest frame
            time.sleep(0.2)

            # ── Refresh seg mask with size guard ──────────────────────────────
            # We must recalculate the mask because the camera has moved during alignment.
            color_aligned, _, _ = cap.read()
            h_a, w_a = color_aligned.shape[:2]
            results_aligned = model(color_aligned, verbose=False, conf=0.5)
            seg_mask = None
            
            best_idx = -1
            min_dist = float('inf')
            
            if results_aligned[0].boxes is not None:
                for idx, box in enumerate(results_aligned[0].boxes):
                    if int(box.cls[0]) == YOLO_CLASS_ID:
                        x1_a, y1_a, x2_a, y2_a = map(int, box.xyxy[0].tolist())
                        cx_a = (x1_a + x2_a) / 2
                        cy_a = (y1_a + y2_a) / 2
                        dist = math.hypot(cx_a - obj_px, cy_a - obj_py)
                        if dist < min_dist:
                            min_dist = dist
                            best_idx = idx
            
            if best_idx >= 0 and results_aligned[0].masks is not None:
                polygon  = results_aligned[0].masks.xy[best_idx].astype(np.int32)
                cand_mask = np.zeros(color_aligned.shape[:2], dtype=np.uint8)
                cv2.fillPoly(cand_mask, [polygon], 255)
                n_px = int(np.sum(cand_mask == 255))
                MAX_MASK_PIXELS = int(h_a * w_a * 0.15)
                if n_px > MAX_MASK_PIXELS:
                    print(f"   ⚠️  Mask too large ({n_px}px > {MAX_MASK_PIXELS}px limit)"
                          f" — likely floor/BG detection, using aligned centroid only")
                else:
                    seg_mask = cand_mask
                    Mm = cv2.moments(seg_mask)
                    if Mm["m00"] > 0:
                        obj_px = int(Mm["m10"] / Mm["m00"])
                        obj_py = int(Mm["m01"] / Mm["m00"])
                        print(f"   🎭 Valid fresh mask ({n_px}px) centroid=({obj_px},{obj_py})")

            # ── STEP 2: Get 3-D object position in camera space ────────────────
            # Prefer mask-based depth (object pixels only) over point sampling.
            # Falls back to bbox-patch median if no segmentation mask is available.
            cur_pose = get_pos(robot)
            pan_deg = cur_pose.get("shoulder_pan.pos", 0.0)
            lift_deg = cur_pose.get("shoulder_lift.pos", 0.0)
            elb_deg = cur_pose.get("elbow_flex.pos", 0.0)
            wst_deg = cur_pose.get("wrist_flex.pos", 0.0)
            t1_deg = 90.0 - lift_deg
            t2_deg = t1_deg - (elb_deg + 81.0)
            t3_deg = t2_deg - (wst_deg + 5.0)
            print(f"   🤖 RAW JOINTS: pan={pan_deg:.1f} lift={lift_deg:.1f} elb={elb_deg:.1f} wst={wst_deg:.1f}")
            print(f"   📐 FK ANGLES: t1={t1_deg:.1f}° t2={t2_deg:.1f}° t3={t3_deg:.1f}° (wrist absolute)")
            
            xyz = None
            if seg_mask is not None:
                result = cap.get_xyz_from_mask(seg_mask)
                if result is not None:
                    cx_px, cy_px, z_mm = result
                    # Full deproject using the mask centroid
                    s = cap.get_xyz(int(cx_px), int(cy_px), search_w=5, search_h=5)
                    if s is not None:
                        xyz = s
                        print(f"   🎭 Mask depth: centroid=({int(cx_px)},{int(cy_px)})  "
                              f"z={z_mm:.0f}mm  (from {np.sum(seg_mask==255)} mask pixels)")

            if xyz is None:
                # Fallback: average bbox-patch samples
                box_w, box_h = max(10, x2 - x1), max(10, y2 - y1)
                xyz_samples  = []
                for _ in range(5):
                    s = cap.get_xyz(obj_px, obj_py, search_w=box_w//2, search_h=box_h//2)
                    if s is not None:
                        xyz_samples.append(s)
                    time.sleep(0.02)
                if not xyz_samples:
                    print("⚠️  No depth data — object may be in D405 blind zone")
                    smooth_move(robot, START_POS, step_size=2.0, step_delay=0.03)
                    STATE = "SEARCHING"
                    continue
                xs = [s[0] for s in xyz_samples]
                ys = [s[1] for s in xyz_samples]
                zs = [s[2] for s in xyz_samples]
                xyz = (float(np.median(xs)), float(np.median(ys)), float(np.median(zs)))
            print(f"   📍 Camera-space: x={xyz[0]:+.0f}mm  y={xyz[1]:+.0f}mm  "
                  f"depth={xyz[2]:.0f}mm")
            # Margin of error / Offset debug
            print(f"   📊 Target margin of error from Camera Center: X_err={xyz[0]:+.0f}mm, Y_err={xyz[1]:+.0f}mm")

            # ── Guard: reject background floor/wall depth noise ────────────────
            if xyz[2] > MAX_GRAB_DEPTH_MM:
                print(f"   ⚠️  Depth reading {xyz[2]:.0f}mm exceeds max table grab depth ({MAX_GRAB_DEPTH_MM:.0f}mm) — background depth detected, ignoring!")
                smooth_move(robot, START_POS, step_size=2.0, step_delay=0.03)
                STATE = "SEARCHING"
                continue

            # ── Convert to arm base frame ──────────────────────────────────────
            target = depth_to_arm_target(xyz, robot)
            if target is None:
                print("⚠️  Cannot reach target — restarting search")
                smooth_move(robot, START_POS, step_size=2.0, step_delay=0.03)
                STATE = "SEARCHING"
                continue

            arm_x, arm_y, arm_z = target
            print(f"   🦾 Arm-base target: x={arm_x:+.0f}mm  y={arm_y:+.0f}mm  z={arm_z:+.0f}mm")

            # ── R3 workspace bounds pre-check (empirical calibrated limits) ──────
            if not workspace_in_bounds(arm_x, arm_y, arm_z):
                rho = math.sqrt(arm_x**2 + arm_y**2)
                print(f"   ⚠️  Target outside calibrated R3 workspace envelope — skipping")
                print(f"         X={arm_x:+.0f}mm (limit {WS_X_MIN_MM:.0f}–{WS_X_MAX_MM:.0f})  "
                      f"Y={arm_y:+.0f}mm (limit {WS_Y_MIN_MM:.0f}–{WS_Y_MAX_MM:.0f})  "
                      f"Z={arm_z:+.0f}mm (limit {WS_Z_MIN_MM:.0f}–{WS_Z_MAX_MM:.0f})  "
                      f"Rho={rho:.0f}mm (limit ≤{WS_RHO_MAX_MM:.0f})")
                smooth_move(robot, START_POS, step_size=2.0, step_delay=0.03)
                STATE = "SEARCHING"
                continue

            # ── Solve IK — auto-pitch, pick closest elbow config ──────────────
            # end_pitch_deg=None → auto-compute from geometry (arm approaches
            # along shoulder→target axis, no rigid straightening).
            # current_joints → picks elbow-up or elbow-down whichever
            # minimises weighted joint travel from current pose.
            #
            # Pitch limits from calibration data:
            #   Image 2 (floor touch):  lift=104.2, elb=-38.5, wst=-9.3
            #     → t1=rad(90-104.2)=-14.2° t2=t1+(-38.5)=-52.7° → pitch≈-52.7°
            #   Image 3 (edge of frame): lift=81.4, elb=67.6, wst=-61.3
            #     → t1=8.6° t2=76.2° → pitch+wrist≈14.9°
            # Clamp between -60° (steep from calibration) and -5° (near level).
            current_j = get_pos(robot)
            rho_t = math.sqrt(arm_x**2 + arm_y**2)
            
            # User constraint: Keep wrist parallel to ground but tilted 5° down
            target_pitch = -5.0
            print(f"   📐 Calculating IK: Target=[{arm_x:+.0f}, {arm_y:+.0f}, {arm_z:+.0f}]mm | Pitch={target_pitch}° | Distance={rho_t:.0f}mm")
            
            grab_pos = solve_ik(arm_x, arm_y, arm_z,
                                end_pitch_deg=target_pitch,
                                current_joints=current_j,
                                wrist_roll_deg=START_POS.get("wrist_roll.pos", -155.96))
            if grab_pos is None:
                print("⚠️  IK: target outside workspace — restarting search")
                smooth_move(robot, START_POS, step_size=2.0, step_delay=0.03)
                STATE = "SEARCHING"
                continue

            print(f"\n📐 IK SOLUTION:")
            print(f"   Pan   → {grab_pos['shoulder_pan.pos']:+.1f}°")
            print(f"   Lift  → {grab_pos['shoulder_lift.pos']:+.1f}°")
            print(f"   Elbow → {grab_pos['elbow_flex.pos']:+.1f}°")
            print(f"   Wrist → {grab_pos['wrist_flex.pos']:+.1f}°")
            print(f"   Roll  → {grab_pos.get('wrist_roll.pos', -155.96):+.1f}°")

            if do_manual_lunge:
                # ── Manual Lunge Demonstration ────────────────────────────────────
                print("\n" + "═"*65)
                print(" 🛠️  MANUAL LUNGE DEMONSTRATION MODE")
                print("═"*65)
                print(" The robot has finished visual alignment and calculated the target!")
                print(f" Target coordinates: X={arm_x:+.0f}mm, Y={arm_y:+.0f}mm, Z={arm_z:+.0f}mm")
                print("\n Disabling motor torque... You can now move the arm manually by hand.")
                print(" Mimic the perfect lunge from this current position to grab the ball.")
                print(" Hold the arm exactly where you want the lunge to end, then press ENTER in terminal.")
                print("═"*65 + "\n")
                
                _set_torque(robot, False)
                input("👉 Press ENTER when you are holding the perfect lunge position...")
                
                pos = get_pos(robot)
                T_wb = forward_kinematics(pos)
                wx, wy, wz = T_wb[0, 3], T_wb[1, 3], T_wb[2, 3]
                
                print("\n\n" + "═"*65)
                print(" 📊 MANUAL LUNGE RESULTS")
                print("═"*65)
                print(f"   At Coordinates : X={wx:+.1f}mm, Y={wy:+.1f}mm, Z={wz:+.1f}mm")
                print(f"   At Joint Angles: Pan={pos.get('shoulder_pan.pos',0):.1f}°, Lift={pos.get('shoulder_lift.pos',0):.1f}°, "
                      f"Elb={pos.get('elbow_flex.pos',0):.1f}°, Wst={pos.get('wrist_flex.pos',0):.1f}°")
                print("═"*65 + "\n")
                
                print("🔌 Restoring motor torque... Returning to search.")
                _set_torque(robot, True)
                smooth_move(robot, START_POS, step_size=2.0, step_delay=0.03)
                STATE = "SEARCHING"
                print("\n🔍 Search loop resumed\n")
            else:
                # ── Full Automatic Lunge & Grab ───────────────────────────────────
                print("\n🦾 APPROACHING (level gripper)...")
                level_approach(robot, grab_pos,
                               step_size=3.0, step_delay=0.03)
                time.sleep(0.4)
    
                print("✊ GRIPPING (Adaptive Maximum Speed)...")
                grab_pos["gripper.pos"] = 0.7
                robot.send_action(grab_pos)
    
                start_t  = time.time()
                braked   = False
                current_g = 20.0
                while time.time() - start_t < 1.5:
                    load = 0
                    try:
                        load = robot.bus.read("Present_Load", ["gripper"])[0]
                    except Exception:
                        try:
                            for arm_obj in getattr(robot, "follower_arms", {}).values():
                                if "gripper" in arm_obj.motor_names:
                                    load = arm_obj.bus.read("Present_Load", ["gripper"])[0]
                        except Exception:
                            pass
    
                    if abs(load) > 150:
                        try:
                            current_g = get_pos(robot).get("gripper.pos", 20.0)
                            current_g = max(current_g - 15.0, 0.7)
                        except Exception:
                            current_g = 20.0
                        grab_pos["gripper.pos"] = current_g
                        robot.send_action(grab_pos)
                        print(f"   🛑 Resistance felt (Load={abs(load)})! Braked at {current_g:.1f}°")
                        braked = True
                        break
    
                    try:
                        if get_pos(robot).get("gripper.pos", 60.0) <= 2.0:
                            current_g = 0.7
                            braked    = True
                            break
                    except Exception:
                        pass
                    time.sleep(0.002)
    
                if not braked:
                    print("   ⚠️  Grip timeout — locking at current position")
                    try:
                        current_g = get_pos(robot).get("gripper.pos", 20.0)
                        current_g = max(current_g - 15.0, 0.7)
                    except Exception:
                        current_g = 20.0
                    grab_pos["gripper.pos"] = current_g
                    robot.send_action(grab_pos)
    
                time.sleep(0.5)
    
                print("🏠 RETURNING...")
                drop_pos = dict(START_POS)
                drop_pos["gripper.pos"] = current_g
                smooth_move(robot, drop_pos, step_size=2.0, step_delay=0.03,
                            hold_joints=["gripper.pos"])
                time.sleep(0.8)
    
                print("🖐 DROPPING...")
                drop_pos["gripper.pos"] = 60.0
                smooth_move(robot, drop_pos, step_size=2.0, step_delay=0.03)
                time.sleep(1.2)
    
                print("🔄 Back to search position...")
                smooth_move(robot, START_POS, step_size=2.0, step_delay=0.03)
                STATE = "SEARCHING"
                print("\n🔍 Search loop resumed\n")

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user.")
    except Exception as exc:
        print(f"\n❌  Unhandled exception: {exc}")
        import traceback; traceback.print_exc()
    finally:
        pass   # atexit _emergency_stow fires here


if __name__ == "__main__":
    main()
#v3 — RealSense D405 + Analytical IK