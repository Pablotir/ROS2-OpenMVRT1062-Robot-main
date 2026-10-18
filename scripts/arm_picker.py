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
import os, sys, yaml
# ── JetPack / aarch64 CUDA Runtime & Memory Safety ───────────────────────────
# On NVIDIA Jetson, torch MUST be imported before cv2 to properly initialize
# the CUDA runtime and OpenMP memory allocators cleanly. Importing cv2 first
# causes native glibc heap corruption ("corrupted size vs. prev_size").
try:
    import torch
except ImportError:
    torch = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Prevent PyTorch/Ultralytics Conv+BN fusion glibc heap corruption on Jetson
if torch is not None:
    try:
        import torch.nn.utils.fusion as _fusion
        _fusion.fuse_conv_bn_weights = lambda conv_w, bn_w, bn_b=None, bn_rm=None, bn_rv=None, eps=1e-5: (conv_w, bn_b)
    except Exception:
        pass

try:
    import ultralytics.utils.torch_utils as _utu
    _utu.fuse_conv_and_bn = lambda c, b: c
except Exception:
    pass

try:
    import ultralytics.nn.tasks as _tasks
    if hasattr(_tasks, "BaseModel"):
        _tasks.BaseModel.is_fused = lambda self: True
        _tasks.BaseModel.fuse = lambda self, *args, **kwargs: self
    if hasattr(_tasks, "DetectionModel"):
        _tasks.DetectionModel.is_fused = lambda self: True
        _tasks.DetectionModel.fuse = lambda self, *args, **kwargs: self
    if hasattr(_tasks, "WorldModel"):
        _tasks.WorldModel.is_fused = lambda self: True
        _tasks.WorldModel.fuse = lambda self, *args, **kwargs: self
except Exception:
    pass

import cv2, time, signal, base64, math, threading, atexit
import numpy as np
try:
    import pyrealsense2 as rs
except ImportError:
    rs = None
import requests

# ── Native Jetson Monitor Display Setup ───────────────────────────────────────
HEADLESS = False

def _setup_native_display():
    """
    Ensure the script has full permission to open native popup windows directly
    on the Jetson's monitor, handling root X11 permissions automatically.
    """
    global HEADLESS
    import glob, subprocess

    # 1. Authorize root using the desktop user's .Xauthority cookie
    cur_auth = os.environ.get("XAUTHORITY", "")
    if not cur_auth or not os.path.exists(cur_auth):
        candidates = [
            "/home/pablo/.Xauthority",
            "/home/jetson/.Xauthority",
            "/root/.Xauthority",
            "/run/user/1000/gdm/Xauthority",
            "/run/user/1000/Xauthority",
        ]
        candidates.extend(glob.glob("/home/*/.Xauthority"))
        for p in candidates:
            if os.path.exists(p) and os.path.getsize(p) > 0:
                os.environ["XAUTHORITY"] = p
                break

    # 2. Probe working DISPLAY (:0, :1, etc.)
    display_candidates = []
    if os.environ.get("DISPLAY"):
        display_candidates.append(os.environ["DISPLAY"])
    display_candidates.extend([":0", ":1", ":0.0", ":1.0"])

    seen = set()
    unique_displays = [d for d in display_candidates if not (d in seen or seen.add(d))]

    for disp in unique_displays:
        os.environ["DISPLAY"] = disp
        try:
            subprocess.run(["xhost", "+local:root"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=0.5)
        except Exception:
            try:
                subprocess.run(["xhost", "+"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=0.5)
            except Exception:
                pass

        try:
            test_win = "__display_probe__"
            cv2.namedWindow(test_win, cv2.WINDOW_AUTOSIZE)
            probe_frame = np.zeros((10, 10, 3), dtype=np.uint8)
            cv2.imshow(test_win, probe_frame)
            cv2.waitKey(1)
            cv2.destroyWindow(test_win)
            cv2.waitKey(1)
            HEADLESS = False
            print(f"🖥️  Jetson screen connected! Native live popup window enabled on DISPLAY={disp}.")
            return True
        except Exception:
            continue

    HEADLESS = True
    print("\n" + "═"*65)
    print(" ⚠️  COULD NOT OPEN NATIVE WINDOW ON THE JETSON SCREEN.")
    print(" 👉 In a terminal on your Jetson desktop, run this ONCE:")
    print("        xhost +")
    print("    Then re-run this script.")
    print("═"*65 + "\n")
    return False

def _init_display_mode() -> None:
    _setup_native_display()

_windows_created = set()

def _show_frame(name: str, img: np.ndarray) -> None:
    """Display the live feed in a native popup window directly on the Jetson monitor."""
    global HEADLESS
    if HEADLESS:
        return
    try:
        if name not in _windows_created:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            h, w = img.shape[:2]
            cv2.resizeWindow(name, min(w, 1280), min(h, 720))
            _windows_created.add(name)
        cv2.imshow(name, img)
        cv2.waitKey(1)
    except Exception as e:
        print(f"⚠️  Window display error: {e}")

def _destroy_windows() -> None:
    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except Exception:
        pass

def _make_vis(img: np.ndarray, depth_map=None) -> np.ndarray:
    """Helper to safely concatenate depth colormap if present and valid."""
    if depth_map is not None and isinstance(depth_map, np.ndarray) and depth_map.ndim == 3:
        if depth_map.shape[:2] == img.shape[:2]:
            return np.hstack((img, depth_map))
    return img





try:
    from lerobot.robots.so101_follower.so101_follower import SO101Follower as SOFollower
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig as SOFollowerRobotConfig
except (ImportError, ModuleNotFoundError):
    try:
        from lerobot.robots.so_follower.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    except (ImportError, ModuleNotFoundError):
        try:
            from lerobot.common.robot_devices.robots.feetech import SO100Follower as SOFollower
            from lerobot.common.robot_devices.robots.configs import SO100FollowerConfig as SOFollowerRobotConfig
        except (ImportError, ModuleNotFoundError):
            SOFollower = None
            SOFollowerRobotConfig = None

# ── Hardware ──────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
PORT       = "/dev/arm_controller"
ARM_ID     = "jetson_arm"

# ── Detection ─────────────────────────────────────────────────────────────────
TARGET_DESC      = "bottle"
YOLO_CLASS_ID    = 39     # 39 = bottle in standard COCO segmentation / detection models
TARGET_CLASS_IDS = [39]   # Set of active class IDs
YOLO_CONF        = 0.50   # Standard detection confidence (uncompromised high threshold)
SKIP_MOONDREAM   = True   # Set False to re-enable Moondream semantic verification

# ── Arm Geometry (SO-ARM101 — Physical Kinematic Constants) ───────────────────
IK_L1 = 115.0   # shoulder pivot  → elbow pivot  (11.5 cm)
IK_L2 = 137.5   # elbow pivot     → wrist flex pivot (13.75 cm)
IK_L3 = 153.0   # wrist flex pivot → gripper fingertips (15.3 cm, matching teach_postures.py)
IK_SHOULDER_HEIGHT_MM = 135.0  # Height of shoulder pivot above table/floor (mm)
# Note: Maximum total physical reach of arm from base = 115 + 137.5 + 153 = 405.5 mm

# Pan alignment calibration (measured: -4.6° corresponds to straight ahead X-axis)
PAN_ZERO_OFFSET_DEG = -4.6
PAN_MIN_DEG         = -113.8   # Far left user-preferred limit
PAN_MAX_DEG         =  113.8   # Far right user-preferred limit

# ── Empirical R3 Workspace Bounds (SO-ARM101 — Physical Reach Envelope) ───────
# Outer convex envelope limits measured in the ARM BASE frame
# (origin = shoulder pivot, +X forward, +Y left, +Z up).
WS_X_MIN_MM  = -140.0   # behind the robot
WS_X_MAX_MM  =  390.0   # max forward reach (calibrated limit with 153mm gripper assembly)
WS_Y_MAX_MM  =  280.0   # max lateral left
WS_Y_MIN_MM  = -280.0   # max lateral right
WS_Z_MIN_MM  = -280.0   # lowest reachable height below shoulder pivot (supports low tables and desks)
WS_Z_MAX_MM  =  300.0   # highest point straight up
WS_RHO_MAX_MM =  390.0  # max horizontal extension from shoulder pivot (405mm theoretical minus margin)
TABLE_FLOOR_Z_MM = -200.0  # fallback physical tabletop height (mm) below shoulder pivot when live camera depth unavailable

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

# Maximum realistic table grab depth (mm). The maximum physical extension of the SO-ARM101
# from shoulder pivot is 390mm (link lengths 152+153+100=405mm max). Readings larger than 380mm
# mean the depth sensor sampled background table/floor/wall noise, so we reject them.
MAX_GRAB_DEPTH_MM = 380.0

# How far PAST the object surface the gripper tip should be at grab time.
# Gripper throat depth is ~37mm. With 20.0mm penetration, the object is placed securely
# between the rubber gripper pads (with ~17mm clearance from rear servo face), ensuring
# firm grip without pushing or toppling.
GRASP_PENETRATION_MM = 20.0

# Radial reach offset for table-resting (lying flat / 3D diagonal) objects.
# Negative (-ve) = pulls gripper back along reach line towards the robot base,
# centering the gripper pads directly over the resting bottle neck and preventing overshoot.
GRASP_FLAT_RADIAL_OFFSET_MM = -30.0

# Preferred gripper approach pitch (degrees relative to horizontal table).
# 0.0° = perfectly parallel/horizontal to ground (ideal for upright bottles, cups, cans).
PREFERRED_GRAB_PITCH_DEG = 0.0


# ==============================================================================
# 🦀 GRASP LATERAL OFFSET (LEFT/RIGHT CLEARANCE) — DO NOT REMOVE
# Positive (+ve) = shifts claw to the LEFT (away from the static pincer)
# Negative (-ve) = shifts claw to the RIGHT
#
# 🛠️ HOW TO MODIFY IF ARM POKES OBJECT:
# - If the STATIC (right) claw pokes the object on approach: INCREASE this value (+5mm to +10mm).
# - If the MOVING (left) claw pokes the object on approach: DECREASE this value (-5mm to -10mm).
# - Current value: 20.0mm (pure lateral shift perpendicular to reach line; preserves radial distance rho)
# ==============================================================================
GRAB_LATERAL_OFFSET_MM = 48.0            # 48.0mm default (verified perfect for vertical flat)
GRAB_LATERAL_OFFSET_STANDING_MM = 76.2   # ~3.0 inches (76.2mm) for standing bottle
GRAB_LATERAL_OFFSET_RIGHT_MM = 76.2      # ~3.0 inches (76.2mm) for pointed-right flat bottle

# ── Gripper Jaw Angles ────────────────────────────────────────────────────────
# Measured physical limits: Closed = 1585 ticks (+4.14°), Open = 2748 ticks (+91.58°)
GRIPPER_CLOSED_POS_DEG = 4.14   # °: Target angle when closing the claw (1585 ticks, firm clamp on bottle)
GRIPPER_OPEN_POS_DEG   = 91.58  # °: Fully open jaws (2748 ticks)


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
        "/root/ros2_ws/scripts/hand_eye_calibration.yaml",
        "/root/ros2_ws/calibration/hand_eye_calibration.yaml",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_eye_calibration.yaml"),
        os.path.join(os.path.dirname(__file__), "..", "calibration", "hand_eye_calibration.yaml"),
        os.path.join(os.path.dirname(__file__), "calibration", "hand_eye_calibration.yaml"),
        "calibration/hand_eye_calibration.yaml",
        "hand_eye_calibration.yaml",
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

    # CAD fallback completely removed per user specification
    raise RuntimeError(
        "CRITICAL ERROR: hand_eye_calibration.yaml not found! "
        "The CAD fallback has been completely removed. Ensure hand_eye_calibration.yaml exists in /root/ros2_ws/scripts/ or /root/ros2_ws/calibration/."
    )

# Precompute at import time
T_CAM_WRIST: np.ndarray = None   # set in main() after math module is loaded

# ── Searching Behaviour ───────────────────────────────────────────────────────
SWEEP_ON_SEARCH    = False  # False = hold steady at START_POS (workspace/table view); True = sweep pan left/right
SEARCH_SWEEP_RANGE = 80.0   # ° pan left/right from START_POS centre
SEARCH_SWEEP_SPEED = 0.5    # ° per YOLO throttle tick (~5 fps = 2.5°/s)

# ── Alignment — closed-loop visual servoing ───────────────────────────────────
ALIGN_MAX_FRAMES   = 150   # max frames to attempt visual alignment (~5-7s)
ALIGN_THRESHOLD    = 30    # px   centred when dot within this many px of crosshair
ALIGN_CENTRED_NEED = 4     # consecutive centred frames to confirm
# NOTE (Review later): In low-light / dark conditions, YOLO detection can drop intermittently.
# Reduced grace period from 25 to 5 frames to fail fast instead of stalling.
ALIGN_LOST_GRACE   = 5     # consecutive not-found frames before abort (reduced from 25)

ALIGN_PAN_OFFSET   = 0     # px: optical center (lateral claw offset handles claw clearance at grab time)

ALIGN_PAN_K        = 0.08  # responsive pan tracking
ALIGN_WRIST_K      = 0.10  # responsive wrist tilt tracking (fast, direct optical pitch)
ALIGN_LIFT_K       = 0.04  # responsive shoulder elevation assistance
ALIGN_MAX_PAN_DEG  = 5.0   # max pan speed (°/frame)
ALIGN_MAX_WRIST_DEG = 5.0  # max wrist tilt speed (°/frame)
ALIGN_MAX_LIFT_DEG = 2.5   # max shoulder lift speed (°/frame)

# ==============================================================================
# ⚠️ CRITICAL REFERENCE POSTURES — DO NOT TOUCH / DO NOT EDIT CASUALLY ⚠️
# ==============================================================================
# These match the active physical motor calibration (jetson_arm.json):
#   - shoulder_lift: homing_offset -1498 (horizontal scan angle ≈ -104.92°)
#   - wrist_roll:    homing_offset -1120 (neutral horizontal angle ≈ +29.27°)
#
# 🛑 DO NOT set shoulder_lift to -52° or -57°: this will cause the arm to violently
# lift +52° into the air on startup and twist its wrist sideways by -65°!
#
# 🛠️ HOW TO SAFELY MODIFY IN THE FUTURE:
# 1. Run `python3 live_joint_angles.py` on the Jetson (cuts torque immediately).
# 2. Position the arm gently by hand into your exact desired tabletop scan posture.
# 3. Read the live angles printed on screen for shoulder_lift, wrist_roll, etc.
# 4. Only update these dictionaries (and arm_reference_poses.yaml) with live measured angles!
# ==============================================================================
# Default fallback scan posture (taught baseline: level wrist, table-facing camera)
_BASE = {
    "shoulder_pan.pos":   -11.60,
    "shoulder_lift.pos": -104.92,
    "elbow_flex.pos":      96.09,
    "wrist_flex.pos":      32.53,
    "wrist_roll.pos":      29.27,
    "gripper.pos":         91.58,
}
# Default fallback stow posture (compact folded safe baseline)
_STOW_BASE = {
    "shoulder_pan.pos":   -11.60,
    "shoulder_lift.pos": -100.00,
    "elbow_flex.pos":      96.09,
    "wrist_flex.pos":      76.84,
    "wrist_roll.pos":      29.27,
    "gripper.pos":         91.58,
}


def _load_reference_poses():
    """Load calibrated scan_base and stow_base postures from YAML if available."""
    search_paths = [
        "/root/ros2_ws/scripts/arm_reference_poses.yaml",
        "/root/ros2_ws/calibration/arm_reference_poses.yaml",
        os.path.join(os.path.dirname(__file__), "../calibration/arm_reference_poses.yaml"),
        os.path.join(os.path.dirname(__file__), "calibration/arm_reference_poses.yaml"),
        os.path.join(os.path.dirname(__file__), "arm_reference_poses.yaml"),
        "arm_reference_poses.yaml",
    ]
    for p in search_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    data = yaml.safe_load(f)
                if not data:
                    continue
                scan = data.get("scan_base", {}).get("joints")
                stow = data.get("stow_base", {}).get("joints")
                if scan:
                    for k, v in scan.items():
                        _BASE[k] = round(float(v), 2)
                    print(f"Loaded scan_base posture from: {p}")
                if stow:
                    for k, v in stow.items():
                        _STOW_BASE[k] = round(float(v), 2)
                    print(f"Loaded stow_base posture from: {p}")
                return p
            except Exception as e:
                print(f"[!] Failed reading {p}: {e}")
    return None

# Load at import time if file exists
_load_reference_poses()

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

    def __init__(self, width=848, height=480, fps=0):
        self._pipeline   = rs.pipeline()
        self._profile    = None
        self._bgr_convert = False

        # If fps <= 0 or None: uncapped mode (probes 90fps, then 60fps, then 30fps)
        if not fps or fps <= 0:
            target_fps_list = [90, 60, 30, 15]
        else:
            target_fps_list = [fps, 90, 60, 30, 15]
        # Preserve order without duplicates
        target_fps_list = list(dict.fromkeys(target_fps_list))

        candidates = []
        for f in target_fps_list:
            candidates.append((width, height, rs.format.yuyv, f, f"YUYV {width}x{height} {f}fps"))
            candidates.append((640,   480,    rs.format.yuyv, f, f"YUYV 640x480 {f}fps"))

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

        # Query actual color format and FPS the hardware selected
        color_stream = self._profile.get_stream(rs.stream.color)
        self._color_format = color_stream.format()
        try:
            self._actual_fps = color_stream.fps()
            print(f"   🔍 Actual hardware stream: {self._color_format} @ {self._actual_fps} FPS (Hardware Max)")
        except Exception:
            print(f"   🔍 Actual color format: {self._color_format}")

        self._colorizer  = None  # initialized lazily only if colorized depth is requested
        
        # Get depth scale for manual calculations
        depth_sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()
        
        self._lock            = threading.Lock()
        self._color           = np.zeros((height, width, 3), dtype=np.uint8)
        self._depth_img       = None
        self._depth_frame_raw = None
        self._intrinsics      = None
        self._frame_count     = 0
        self._new_frame_event = threading.Event()
        self._running         = True
        self._thread          = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                frames      = self._pipeline.wait_for_frames(timeout_ms=250)
                if not self._running:
                    break
                # D405 RGB and Depth share the same ISP sensor, so they are perfectly aligned natively.
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    print("⚠️  RealSense: got frames but color/depth missing — check USB cable")
                    time.sleep(0.1)
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
                    color = (color >> 8).astype(np.uint8)
                    color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
                else:
                    if color.ndim == 2:
                        color = cv2.cvtColor(color.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    elif color.shape[2] == 4:
                        color = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
                    elif color.shape[2] == 3 and self._bgr_convert:
                        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                # ALWAYS enforce uint8 BGR
                color = np.ascontiguousarray(color, dtype=np.uint8)
                
                # Extract raw depth array
                depth_img = np.asanyarray(depth_frame.get_data()).copy()
                
                with self._lock:
                    self._color = color
                    self._depth_img = depth_img
                    self._depth_frame_raw = depth_frame
                    self._intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
                    self._frame_count += 1
                self._new_frame_event.set()
            except Exception as e:
                if not self._running:
                    break
                print(f"⚠️  RealSense frame error: {e}")
                time.sleep(0.05)

    def read(self, wait_new=False, timeout=0.05):
        """Return (color_bgr_copy, has_depth, depth_colormap)."""
        if wait_new:
            self._new_frame_event.wait(timeout=timeout)
            self._new_frame_event.clear()
        with self._lock:
            if self._depth_img is None:
                return self._color.copy(), False, None
            return self._color.copy(), True, None

    def get_raw_depth(self):
        """Return (depth_img_uint16_copy, depth_scale_float) or (None, 0.001)."""
        with self._lock:
            if self._depth_img is None:
                return None, 0.001
            return self._depth_img.copy(), float(self._depth_scale)

    def get_colorized_depth(self):
        """Generate colorized depth map lazily on demand to avoid CPU overhead at high FPS."""
        with self._lock:
            depth_frame = self._depth_frame_raw
        if depth_frame is None:
            return None
        if self._colorizer is None:
            self._colorizer = rs.colorizer()
            self._colorizer.set_option(rs.option.color_scheme, 0)
            self._colorizer.set_option(rs.option.histogram_equalization_enabled, 1)
        colorized = np.asanyarray(self._colorizer.colorize(depth_frame).get_data())
        return cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR)

    def get_xyz(self, px: int, py: int, search_w=40, search_h=40):
        """
        Deprojects pixel (px, py) into 3-D camera-space coordinates (mm).
        Uses a dynamic median patch (search_w x search_h) for noise rejection.
        Optimized with NumPy slicing and foreground clustering to prevent table contamination.
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
            
        # Robust foreground surface depth extraction:
        # In a patch centered on a narrow neck or edge resting on a table,
        # background table pixels outnumber the foreground object.
        # np.percentile(valid, 5) rejects single-pixel dropouts, and foreground cluster
        # within 25mm isolates the true front surface of the object from the tabletop.
        d_min = float(np.percentile(valid, 5))
        fg = valid[(valid >= d_min) & (valid <= d_min + 0.025)]
        z_m = float(np.median(fg)) if fg.size >= 3 else float(np.median(valid))
        point = rs.rs2_deproject_pixel_to_point(intr, [float(px), float(py)], z_m)
        return point[0] * 1000.0, point[1] * 1000.0, point[2] * 1000.0  # → mm

    def get_xyz_from_mask(self, mask: np.ndarray, target_px: tuple[int, int] | None = None) -> tuple[float, float, float] | None:
        """
        Compute (cx_px, cy_px, z_mm) for the object described by a binary
        segmentation mask (same HxW as the color frame, dtype uint8, 255=object).
        If target_px=(tx, ty) is provided (e.g. from optimal grasp point analysis
        such as bottle neck/cap), it deprojects that point using local mask depth.

        Uses ONLY the depth pixels that belong to the mask so background
        depth values never contaminate the object distance estimate.
        Returns (cx_px, cy_px, median_depth_mm) or None if no valid depth pixels.
        """
        with self._lock:
            depth_img = self._depth_img
            intr      = self._intrinsics
            scale     = self._depth_scale
        if depth_img is None or intr is None:
            return None

        if target_px is not None:
            cx_px, cy_px = int(target_px[0]), int(target_px[1])
        else:
            # Centroid of the mask (pixel coordinates)
            M = cv2.moments(mask)
            if M["m00"] < 1:
                return None
            cx_px = int(M["m10"] / M["m00"])
            cy_px = int(M["m01"] / M["m00"])

        # Collect depth readings for mask pixels
        h, w  = depth_img.shape[:2]
        mh, mw = mask.shape[:2]
        if (mh, mw) != (h, w):
            mask_rs = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_rs = mask

        if target_px is not None:
            tx, ty = cx_px, cy_px
            rw = 20
            y1_p, y2_p = max(0, ty - rw), min(h, ty + rw + 1)
            x1_p, x2_p = max(0, tx - rw), min(w, tx + rw + 1)
            local_mask = mask_rs[y1_p:y2_p, x1_p:x2_p]
            local_depth = depth_img[y1_p:y2_p, x1_p:x2_p]
            raw_depths = local_depth[local_mask == 255].astype(float) * scale
            valid = raw_depths[(raw_depths > 0.001) & (raw_depths < 2.0)]
            if valid.size < 5:
                raw_depths = depth_img[mask_rs == 255].astype(float) * scale
                valid = raw_depths[(raw_depths > 0.001) & (raw_depths < 2.0)]
        else:
            raw_depths = depth_img[mask_rs == 255].astype(float) * scale
            valid = raw_depths[(raw_depths > 0.001) & (raw_depths < 2.0)]

        if valid.size == 0:
            return None

        z_m = float(np.median(valid))
        z_mm = z_m * 1000.0

        # Deproject target pixel using the median mask depth
        point = rs.rs2_deproject_pixel_to_point(
            intr, [float(cx_px), float(cy_px)], z_m)
        return float(point[0] * 1000.0), float(point[1] * 1000.0), float(z_mm)

    def stop(self):
        if not self._running:
            return
        self._running = False
        try:
            if hasattr(self, "_thread") and self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=1.5)
        except Exception:
            pass
        try:
            if self._pipeline is not None and hasattr(self, "_thread") and (not self._thread.is_alive()):
                self._pipeline.stop()
                self._pipeline = None
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


# STS3215 Hardware_Error_Status bit masks (address 72, 1 byte)
_HW_ERR_BITS = {
    0x01: "Input Voltage Error",
    0x02: "Motor Overheat",
    0x04: "Overload Error",     # ← most common: servo stalled against hard stop
    0x08: "ElectricalShock Error",
    0x10: "Overheated Error",
    0x20: "Instruction Error",
}
_MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll", "gripper"]

# Register name variants across LeRobot versions — tried in order
_ERR_REG_CANDIDATES = [
    "Hardware_Error_Status",   # most LeRobot versions
    "hardware_error_status",   # some builds use snake_case
    "Hw_Error_Status",         # older builds
    "HW_Error_Status",
]
# Load-based fallback: STS3215 Present_Load ≈ ±1023 range; >800 = likely stalled
_LOAD_REG_CANDIDATES = ["Present_Load", "present_load", "Load"]
_LOAD_STALL_THRESHOLD = 800

def check_servo_health(robot) -> bool:
    """
    Read error status from every servo BEFORE issuing any motion.
    Tries Hardware_Error_Status first (direct overload flag), then falls back
    to Present_Load as a proxy (high load = stalled/overloaded).
    Returns True if all servos appear healthy, False if any are faulted.
    STS3215 overload protection clears on power-cycle only.
    """
    print("🩺 Checking servo health...")

    # ── Step 1: find a working error-status register name ─────────────────
    err_reg = None
    for candidate in _ERR_REG_CANDIDATES:
        try:
            robot.bus.read(candidate, _MOTOR_NAMES[0])
            err_reg = candidate
            break
        except Exception:
            continue

    # ── Step 2: if error register found, read all motors ──────────────────
    if err_reg is not None:
        all_ok = True
        for name in _MOTOR_NAMES:
            try:
                val = int(robot.bus.read(err_reg, name))
                if val != 0:
                    flags = [desc for bit, desc in _HW_ERR_BITS.items() if val & bit]
                    print(f"   ❌  {name}: error=0x{val:02X}  ({', '.join(flags)})")
                    all_ok = False
                else:
                    print(f"   ✅  {name}: OK")
            except Exception as e:
                print(f"   ⚠️  {name}: read failed ({e})")
        if not all_ok:
            print("\n   ⛔  One or more servos are in an error/overload state.")
            print("   ⛔  Power-cycle the arm (unplug and replug the power supply),")
            print("   ⛔  then re-run arm_picker.py.")
            print("   ⛔  Do NOT attempt to move the arm while in this state.\n")
        else:
            print("   ✅ All servos healthy — safe to move.\n")
        return all_ok

    # ── Step 3: fallback — use Present_Load as a stall proxy ─────────────
    load_reg = None
    for candidate in _LOAD_REG_CANDIDATES:
        try:
            robot.bus.read(candidate, _MOTOR_NAMES[0])
            load_reg = candidate
            break
        except Exception:
            continue

    if load_reg is not None:
        all_ok = True
        for name in _MOTOR_NAMES:
            try:
                raw_val = abs(int(robot.bus.read(load_reg, name)))
                # Feetech STS3215 Present_Load register specification:
                # Bit 10 (0x400 = 1024) is the DIRECTION bit (0: CCW, 1: CW)
                # Bits 0-9 (0..1023) are the load magnitude (0% - 100% of max stall)
                # Without masking 0x3FF, a tiny 2% load in direction 1 reads as 1044,
                # causing false-positive stall alarms!
                load_mag = raw_val & 0x03FF
                if load_mag > _LOAD_STALL_THRESHOLD:
                    print(f"   ❌  {name}: high load ({load_mag}/1023) — may be stalled (raw={raw_val})")
                    all_ok = False
                else:
                    print(f"   ✅  {name}: load={load_mag}/1023")
            except Exception as e:
                print(f"   ⚠️  {name}: load read failed ({e})")
        if not all_ok:
            print("\n   ⛔  One or more servos show high load — possible overload state.")
            print("   ⛔  Power-cycle the arm, then re-run arm_picker.py.\n")
        else:
            print("   ✅ All servos healthy (load check) — safe to move.\n")
        return all_ok


    # ── Step 4: nothing worked — warn and proceed ─────────────────────────
    print("   ⚠️  Health check unavailable (register names not found in control table).")
    print("   ⚠️  Proceeding — if arm loses power immediately, power-cycle it.\n")
    return True



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
    continuously adjusting wrist_flex so the gripper smoothly tracks the
    desired approach pitch along the reach trajectory (pitch = 0° for horizontal,
    or steep/angled down for tilted or tabletop objects).

    All servos move at the same rate without staging.
    """
    cur = get_pos(robot)
    max_delta = max(abs(target[j] - cur.get(j, 0.0)) for j in target)
    if max_delta < 0.5:
        return
    n = max(1, int(max_delta / step_size))

    # Calculate initial and target pitch relative to horizontal
    cur_lift = cur.get("shoulder_lift.pos", 0.0)
    cur_elb  = cur.get("elbow_flex.pos", 0.0)
    cur_wst  = cur.get("wrist_flex.pos", 0.0)
    t1_0 = math.radians(90.0 - cur_lift)
    t2_0 = t1_0 - math.radians(cur_elb + 81.0)
    pitch_0 = math.degrees(t2_0 - math.radians(cur_wst + 5.0))

    tgt_lift = target.get("shoulder_lift.pos", cur_lift)
    tgt_elb  = target.get("elbow_flex.pos", cur_elb)
    tgt_wst  = target.get("wrist_flex.pos", cur_wst)
    t1_tgt = math.radians(90.0 - tgt_lift)
    t2_tgt = t1_tgt - math.radians(tgt_elb + 81.0)
    pitch_tgt = math.degrees(t2_tgt - math.radians(tgt_wst + 5.0))

    for s in range(1, n + 1):
        t = s / n
        interp = {j: cur.get(j, 0.0) + t * (target[j] - cur.get(j, 0.0))
                  for j in target}

        # Desired pitch smoothly transitions from initial pitch to target pitch
        desired_pitch_deg = pitch_0 + t * (pitch_tgt - pitch_0)

        lift_now = interp.get("shoulder_lift.pos", 0.0)
        elb_now  = interp.get("elbow_flex.pos", 0.0)
        t1_rad = math.radians(90.0 - lift_now)
        t2_rad = t1_rad - math.radians(elb_now + 81.0)

        # Exact wrist_flex motor angle for the interpolated approach pitch
        interp["wrist_flex.pos"] = math.degrees(t2_rad - math.radians(desired_pitch_deg)) - 5.0

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


def save_reference_poses(scan_joints=None, stow_joints=None, filepath=None):
    """Save scan_base and stow_base to arm_reference_poses.yaml and update in-memory dicts."""
    if filepath is None:
        target_dir = "/root/ros2_ws/calibration"
        if not os.path.exists(target_dir):
            target_dir = os.path.join(os.path.dirname(__file__), "../calibration")
        if not os.path.exists(target_dir):
            target_dir = os.path.join(os.path.dirname(__file__), "calibration")
        if not os.path.exists(target_dir):
            target_dir = os.path.dirname(__file__)
        filepath = os.path.join(target_dir, "arm_reference_poses.yaml")

    data = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            data = {}

    if scan_joints:
        try:
            T_wb = forward_kinematics(scan_joints)
            wx, wy, wz = float(T_wb[0, 3]), float(T_wb[1, 3]), float(T_wb[2, 3])
            rho = float(math.sqrt(wx**2 + wy**2 + wz**2))
        except Exception:
            wx, wy, wz, rho = 0.0, 0.0, 0.0, 0.0
        data["scan_base"] = {
            "joints": {k: float(v) for k, v in scan_joints.items()},
            "fk_xyz_mm": [round(wx, 2), round(wy, 2), round(wz, 2)],
            "reach_rho_mm": round(rho, 2),
            "recorded_at": datetime.now().isoformat(),
        }
        for k, v in scan_joints.items():
            _BASE[k] = round(float(v), 2)

    if stow_joints:
        try:
            T_wb = forward_kinematics(stow_joints)
            wx, wy, wz = float(T_wb[0, 3]), float(T_wb[1, 3]), float(T_wb[2, 3])
            rho = float(math.sqrt(wx**2 + wy**2 + wz**2))
        except Exception:
            wx, wy, wz, rho = 0.0, 0.0, 0.0, 0.0
        data["stow_base"] = {
            "joints": {k: float(v) for k, v in stow_joints.items()},
            "fk_xyz_mm": [round(wx, 2), round(wy, 2), round(wz, 2)],
            "reach_rho_mm": round(rho, 2),
            "recorded_at": datetime.now().isoformat(),
        }
        for k, v in stow_joints.items():
            _STOW_BASE[k] = round(float(v), 2)

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)
    print(f"\n💾 Saved reference poses successfully to:\n   {filepath}")
    return filepath


def connect_robot():
    """
    Connect to SO-ARM101 using calibrated JSON, registering typed calibration
    attributes with the motor bus, and verifying servo health.
    """
    print("🔌 Connecting to SO-ARM101...")
    config = SOFollowerRobotConfig(port=PORT, id=ARM_ID, use_degrees=True)
    robot  = SOFollower(config)

    # Locate calibration JSON
    import json as _json, pathlib as _pathlib, builtins as _builtins
    _hf_home = _pathlib.Path(os.environ.get("HF_HOME",
                  os.environ.get("TRANSFORMERS_CACHE",
                  str(_pathlib.Path.home() / ".cache" / "huggingface"))))
    _calib_search = [
        _pathlib.Path(f"/root/ros2_ws/calibration/{ARM_ID}.json"),
        _pathlib.Path(f"/root/ros2_ws/scripts/{ARM_ID}.json"),
        _pathlib.Path(__file__).parent / f"{ARM_ID}.json",
        _pathlib.Path(__file__).parent.parent / "calibration" / f"{ARM_ID}.json",
        _pathlib.Path(__file__).parent / "calibration" / f"{ARM_ID}.json",
        _pathlib.Path(f"calibration/{ARM_ID}.json"),
        _pathlib.Path(f"/data/models/huggingface/lerobot/calibration/robots/so101_follower/{ARM_ID}.json"),
        _pathlib.Path(f"/data/models/huggingface/lerobot/calibration/robots/so_follower/{ARM_ID}.json"),
        _hf_home / f"lerobot/calibration/robots/so101_follower/{ARM_ID}.json",
        _hf_home / f"lerobot/calibration/robots/so_follower/{ARM_ID}.json",
        _pathlib.Path(f"/root/.cache/huggingface/lerobot/calibration/robots/so101_follower/{ARM_ID}.json"),
        _pathlib.Path(f"/root/.cache/huggingface/lerobot/calibration/robots/so_follower/{ARM_ID}.json"),
        _pathlib.Path.home() / f".cache/huggingface/lerobot/calibration/robots/so101_follower/{ARM_ID}.json",
        _pathlib.Path.home() / f".cache/huggingface/lerobot/calibration/robots/so_follower/{ARM_ID}.json",
    ]
    _calib_path = next((p for p in _calib_search if p.exists()), None)

    _EMBEDDED_CALIB = {
        "shoulder_pan":  {"id": 1, "drive_mode": 0, "homing_offset": 1604,  "range_min": 962,  "range_max": 3486},
        "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": -1498, "range_min": 814,  "range_max": 3207},
        "elbow_flex":    {"id": 3, "drive_mode": 0, "homing_offset": 1619,  "range_min": 882,  "range_max": 3138},
        "wrist_flex":    {"id": 4, "drive_mode": 0, "homing_offset": -1885, "range_min": 887,  "range_max": 3243},
        "wrist_roll":    {"id": 5, "drive_mode": 0, "homing_offset": -1120, "range_min": 765,  "range_max": 3495},
        "gripper":       {"id": 6, "drive_mode": 0, "homing_offset": 1947,  "range_min": 1530, "range_max": 2860}
    }

    if _calib_path is not None:
        print(f"   📂 Calibration: {_calib_path}")
        with open(_calib_path) as _f:
            _calib_data = _json.load(_f)
    else:
        print(f"   📂 Calibration file not found on disk — using embedded calibrated profile for '{ARM_ID}'.")
        _calib_data = _EMBEDDED_CALIB
        # Auto-persist to /root/ros2_ws/calibration/jetson_arm.json
        try:
            _persist_p = _pathlib.Path(f"/root/ros2_ws/calibration/{ARM_ID}.json")
            _persist_p.parent.mkdir(parents=True, exist_ok=True)
            with open(_persist_p, "w") as _pf:
                _json.dump(_calib_data, _pf, indent=4)
            print(f"   💾 Auto-persisted calibration profile to: {_persist_p}")
        except Exception:
            pass

    # Check degenerate
    if "start_pos" in _calib_data:
        _s, _e = _calib_data["start_pos"], _calib_data["end_pos"]
        _is_degenerate = bool(_s) and all(a == b for a, b in zip(_s, _e))
    else:
        _ranges = [(v["range_min"], v["range_max"])
                   for v in _calib_data.values()
                   if isinstance(v, dict) and "range_min" in v]
        _is_degenerate = bool(_ranges) and all(mn == mx for mn, mx in _ranges)

    if _is_degenerate:
        raise RuntimeError("Degenerate calibration file — all ranges are identical. Delete and re-calibrate.")

    # Configure retry count on motor bus to make half-duplex UART robust against jitter
    if hasattr(robot, "bus") and hasattr(robot.bus, "default_num_retry"):
        robot.bus.default_num_retry = 3

    # Connect with automatic retries and port reset
    connected = False
    last_err = None
    for attempt in range(1, 4):
        try:
            try:
                robot.connect(calibrate=False)
            except TypeError:
                _real_input = _builtins.input
                def _auto_use_file(prompt=""):
                    if "enter" in prompt.lower() and "range" not in prompt.lower():
                        return ""
                    _builtins.input = _real_input
                    return _real_input(prompt)
                _builtins.input = _auto_use_file
                try:
                    robot.connect()
                finally:
                    _builtins.input = _real_input
            connected = True
            break
        except ConnectionError as ce:
            last_err = ce
            print(f"   ⚠️  Connection attempt {attempt}/3 failed: {ce}")
            if hasattr(robot, "bus"):
                try:
                    if hasattr(robot.bus, "port_handler") and robot.bus.port_handler:
                        robot.bus.port_handler.clearPort()
                except Exception:
                    pass
                try:
                    robot.bus.disconnect()
                except Exception:
                    pass
            time.sleep(0.6)
        except Exception as e:
            last_err = e
            print(f"   ⚠️  Connection attempt {attempt}/3 error: {e}")
            if hasattr(robot, "bus"):
                try:
                    robot.bus.disconnect()
                except Exception:
                    pass
            time.sleep(0.6)

    if not connected:
        print("\n" + "═"*65)
        print(" ⛔ ROBOT CONNECTION FAILED (Servo Communication / Overload Error)")
        print(f" ⚠️  Error details: {last_err}")
        print(" 🔧 Quick Recovery Steps:")
        print("    1. POWER CYCLE ARM: Unplug the arm power supply (barrel jack),")
        print("       wait 5 seconds, and plug it back in. STS3215 internal overload")
        print("       protection only clears on a power cycle!")
        print("    2. SUPPORT ARM: Gently support the arm by hand so Motor 2")
        print("       (shoulder_lift) is not strained against gravity during startup.")
        print("    3. USB CHECK: Ensure the arm controller USB cable is firmly plugged in.")
        print("═"*65 + "\n")
        raise RuntimeError(f"Could not connect to arm: {last_err}")

    # Build typed calibration objects for LeRobot _normalize attribute access
    from types import SimpleNamespace as _NS
    _MC = None
    for _mc_mod in ("lerobot.motors.motors_bus", "lerobot.motors.feetech",
                    "lerobot.common.robot_devices.motors.feetech"):
        try:
            import importlib as _il
            _mod = _il.import_module(_mc_mod)
            for _cname in ("MotorCalibration", "CalibrationData", "Calibration"):
                if hasattr(_mod, _cname):
                    _MC = getattr(_mod, _cname)
                    break
            if _MC:
                break
        except Exception:
            pass

    def _make_motor_calib(d: dict):
        if _MC is not None:
            try:
                import dataclasses as _dc
                if _dc.is_dataclass(_MC):
                    _fields = {f.name for f in _dc.fields(_MC)}
                    return _MC(**{k: v for k, v in d.items() if k in _fields})
                return _MC(**d)
            except Exception:
                pass
        return _NS(**d)

    _typed_calib = {
        _motor: _make_motor_calib(_jdata)
        for _motor, _jdata in _calib_data.items()
        if isinstance(_jdata, dict)
    }

    _registered = False
    for _method in ("set_calibration", "load_calibration", "_set_calibration"):
        if hasattr(robot.bus, _method):
            for _payload in (_typed_calib, _calib_data):
                try:
                    getattr(robot.bus, _method)(_payload)
                    _registered = True
                    break
                except Exception:
                    pass
            if _registered:
                break
    if not _registered:
        for _attr in ("calibration", "_calibration"):
            try:
                setattr(robot.bus, _attr, _typed_calib)
                _registered = True
                break
            except Exception:
                pass

    print("   ✅ Arm connected and calibration registered")

    # Servo health check
    if not check_servo_health(robot):
        robot.disconnect()
        raise RuntimeError("Servo health check failed — power cycle arm.")

    return robot


def teach_postures(robot=None):
    """
    Interactive teaching mode:
    1. Cuts torque immediately so arm can be freely guided by hand.
    2. Streams live joint angles.
    3. User captures Scan and Stow postures.
    4. Saves to arm_reference_poses.yaml and updates in-memory _BASE / _STOW_BASE.
    """
    owns_robot = False
    if robot is None:
        robot = connect_robot()
        owns_robot = True

    try:
        print("\n" + "═"*65)
        print(" 🎓 INTERACTIVE TEACHING MODE: RECORD REFERENCE POSTURES")
        print("═"*65)
        print(" ⚠️  DISABLING MOTOR TORQUE NOW — support the arm by hand!")
        time.sleep(0.5)
        _set_torque(robot, False)
        print(" 🔓 Torque DISABLED. You can freely guide the arm by hand.\n")

        print("---------------------------------------------------------------")
        print(" STEP 1: Set NEUTRAL SCAN / START Posture")
        print(" Guide the arm into your desired neutral scanning position:")
        print("   - Shoulder pan centered facing forward (~0°)")
        print("   - Shoulder lift & elbow set so camera views workspace")
        print("   - Wrist tilted ~45° down toward target area")
        print("   - Gripper open/ready")
        print("---------------------------------------------------------------")
        input(" 👉 Hold arm in SCAN posture, then press ENTER to capture... ")
        scan_pos = get_pos(robot)
        print("\n ✅ Captured SCAN posture:")
        for k, v in sorted(scan_pos.items()):
            print(f"    {k:20s}: {v:+6.2f}°")

        print("\n---------------------------------------------------------------")
        print(" STEP 2: Set STOW / PARK Posture")
        print(" Guide the arm into your desired resting/stow position:")
        print("   - Folded back safely, close to base, gripper compact")
        print("---------------------------------------------------------------")
        input(" 👉 Hold arm in STOW posture, then press ENTER to capture... ")
        stow_pos = get_pos(robot)
        print("\n ✅ Captured STOW posture:")
        for k, v in sorted(stow_pos.items()):
            print(f"    {k:20s}: {v:+6.2f}°")

        filepath = save_reference_poses(scan_pos, stow_pos)

        print("\n 📋 Python dict snippet for arm_picker.py:")
        print("_BASE = {")
        for k, v in sorted(scan_pos.items()):
            print(f'    "{k}": {round(v, 2):7.2f},')
        print("}")
        print("_STOW_BASE = {")
        for k, v in sorted(stow_pos.items()):
            print(f'    "{k}": {round(v, 2):7.2f},')
        print("}")
        print("═"*65)
        print(" ✅ Postures successfully calibrated and active in memory!\n")

    finally:
        print("🔌 Restoring motor torque...")
        _set_torque(robot, True)
        if owns_robot:
            robot.disconnect()


def solve_ik(x_mm: float, y_mm: float, z_mm: float,
             end_pitch_deg: float | None = None,
             current_joints: dict | None = None,
             wrist_roll_deg: float = 29.27) -> dict | None:
    """
    Closed-form analytical IK for SO-ARM101.
    Target (x_mm, y_mm, z_mm) is in the ARM BASE frame.
    Supports horizontal (0°), angled (-25° to -45°), and steep top-down (-65° to -85°) grasps.
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
        preferred_pitch = float(np.clip(natural_pitch, -85.0, -5.0))
    else:
        preferred_pitch = end_pitch_deg

    # ── Search for a reachable pitch ─────────────────────────────────────────
    pitch_candidates = [preferred_pitch]
    if end_pitch_deg is not None:
        if abs(preferred_pitch) < 1e-3:
            # Strictly horizontal approach (pitch = 0.0° parallel to ground)
            # Try level orientations first [-5.0° to +5.0°]
            for offset in [0.0, 2.5, -2.5, 5.0, -5.0]:
                cand = round(preferred_pitch + offset, 1)
                if cand not in pitch_candidates:
                    pitch_candidates.append(cand)
            # If level approach cannot reach (e.g. object is lower than shoulder pivot on table/floor),
            # search downward angled approaches [-10° to -85°] so arm can lunge and grasp naturally:
            for down_pitch in [-10.0, -15.0, -20.0, -25.0, -30.0, -35.0, -40.0, -45.0, -50.0, -55.0, -60.0, -65.0, -70.0, -75.0, -80.0, -85.0]:
                if down_pitch not in pitch_candidates:
                    pitch_candidates.append(down_pitch)
        elif preferred_pitch <= -80.0:
            # Strictly top-down vertical approach (perpendicular to ground)
            for offset in [0.0, 2.5, -2.5, 5.0, -5.0, 7.5, -7.5]:
                cand = round(preferred_pitch + offset, 1)
                if -90.0 <= cand <= -70.0 and cand not in pitch_candidates:
                    pitch_candidates.append(cand)
            for down_pitch in [-65.0, -60.0, -55.0, -50.0, -45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0]:
                if down_pitch not in pitch_candidates:
                    pitch_candidates.append(down_pitch)
        else:
            for offset in [0.0, 2.5, -2.5, 5.0, -5.0, 7.5, -7.5, 10.0, -10.0, -15.0, 15.0, -20.0, 20.0, -25.0, 25.0, -30.0, -35.0, -40.0, -45.0, -50.0, -55.0, -60.0, -65.0, -70.0, -75.0, -80.0]:
                cand = round(preferred_pitch + offset, 1)
                if -90.0 <= cand <= 30.0 and cand not in pitch_candidates:
                    pitch_candidates.append(cand)
    else:
        # Auto-pitch: search progressively steeper downwards
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

            # For forward reaching and grasping, reject configurations where the elbow
            # remains folded tightly backward against the upper arm (m_elbow > 45° is a stow fold, not a reach)
            if m_elbow > 45.0: continue

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
                
            # Prefer natural forward reaching posture (unfolded elbow: negative m_elbow)
            elbow_fold_penalty = 100.0 if c["elbow_flex.pos"] > 20.0 else 0.0
            cost = (3.0 * abs(c["shoulder_lift.pos"] - current_joints.get("shoulder_lift.pos", 0)) +
                    1.5 * abs(c["elbow_flex.pos"] - current_joints.get("elbow_flex.pos", 0)) +
                    1.0 * abs(c["wrist_flex.pos"] - current_joints.get("wrist_flex.pos", 0)) +
                    elbow_fold_penalty)
            if cost < best_cost:
                best_cost = cost
                best_solution = c

        if best_solution is not None:
            # Found a valid configuration for this pitch
            print(f"   ✅ IK matched at pitch {test_pitch:.1f}° (Lift={best_solution['shoulder_lift.pos']:+.1f}°, Elb={best_solution['elbow_flex.pos']:+.1f}°, Wst={best_solution['wrist_flex.pos']:+.1f}°)")
            return best_solution

    print(f"   ⚠️  IK reject: Target geometrically unreachable at any pitch.")
    return None


def depth_to_arm_target(xyz_cam: tuple[float, float, float],
                         robot,
                         penetration_mm: float | None = None,
                         verbose: bool = True) -> tuple[float, float, float] | None:
    """
    STEP 3 — Base Coordinate Transform (Camera → Wrist → Base).

    Implements the standard eye-in-hand matrix chain:

        P_base = T_wrist_base  ×  T_cam_wrist  ×  P_cam

    where:
      P_cam        = 3-D point in camera coordinates (mm, homogeneous)
      T_cam_wrist  = static calibration matrix (CAM_*_OFFSET_MM, measured once)
      T_wrist_base = live FK matrix from current servo angles
      P_base       = target position in arm base frame → fed to IK solver

    The approach depth is baked into P_cam using penetration_mm (defaults to GRASP_PENETRATION_MM)
    so the object is centered deep between the claw pads.

    Returns (x_mm, y_mm, z_mm) in base frame (origin = shoulder pivot), or
    None if the object is inside the D405 70 mm minimum-range blind zone.
    """
    x_cam, y_cam, z_cam = xyz_cam

    # Guard: D405 cannot see objects closer than 70 mm
    if z_cam < D405_MIN_RANGE_MM + 10.0:
        if verbose:
            print(f"   ⚠️  Object inside D405 blind zone ({z_cam:.0f} mm < "
                  f"{D405_MIN_RANGE_MM} mm) — skipping")
        return None

    # Bake grasp approach penetration into the camera-space z coordinate.
    pen = penetration_mm if penetration_mm is not None else GRASP_PENETRATION_MM
    approach_z = z_cam + pen
    if approach_z < D405_MIN_RANGE_MM:
        approach_z = D405_MIN_RANGE_MM
        if verbose:
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
    if verbose:
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


# ── SO-ARM101 Gripper Mechanical Specifications (Official Robonine / LeRobot) ─
# Standard parallel jaw stroke: 76.0mm to 84.0mm (driven by STS3215 servo).
GRIPPER_MAX_STROKE_MM       = 84.0   # Published maximum physical jaw opening stroke
GRIPPER_SAFE_CLEARANCE_MM   = 65.0   # Maximum safe object width to approach without collision
GRIPPER_OPTIMAL_MIN_MM      = 12.0   # Minimum grasp width for stable grip


def size_up_object(frame: np.ndarray | None,
                   seg_mask: np.ndarray | None,
                   bbox: tuple[int, int, int, int],
                   depth_mm: float = 280.0,
                   target_desc: str = "bottle",
                   fx: float = 452.5,
                   depth_frame: np.ndarray | None = None,
                   depth_scale: float = 0.001) -> dict:
    """
    Sizes up the object and profiles its cross-sectional width along its true principal axis using PCA.
    Accurately handles objects at arbitrary angles (upright, 30°-60° tilted, or lying flat on tabletop).
    
    Returns a dict with:
        body_width_mm: physical width of the widest body section (measured along minor axis)
        thinnest_width_mm: physical width of the narrowest graspable section
        grasp_width_mm: width at the chosen grasp point
        height_mm: physical total length/height estimate
        height_pct: percentage along the major axis from base to cap
        fits_gripper: True if thinnest section <= GRIPPER_SAFE_CLEARANCE_MM (65mm)
        opt_px: (x, y) pixel coordinates of the optimal grasp point (neck/cap)
        z_target_est: physical arm base Z elevation (mm) for level approach
        tilt_deg: orientation angle relative to vertical (-90° to +90°, 0° = upright)
        recommended_pitch_deg: optimal approach pitch (0° horizontal, or angled/top-down)
        recommended_roll_deg: optimal wrist_roll servo angle to align jaws with the object
        delta_roll_deg: relative roll angle offset from neutral
    """
    x1, y1, x2, y2 = bbox
    w_box = max(10, x2 - x1)
    h_box = max(10, y2 - y1)
    scale = depth_mm / fx
    default_cx = (x1 + x2) // 2
    default_cy = (y1 + y2) // 2

    is_bottle_like = any(w in target_desc.lower() for w in ["bottle", "flask", "can", "cup", "drink", "container", "mug"])

    # Extract silhouette mask within bounding box
    crop_mask = None
    if seg_mask is not None and np.sum(seg_mask == 255) >= 30:
        crop_mask = seg_mask[y1:y2, x1:x2].copy()
    elif depth_frame is not None and depth_frame.size > 0:
        try:
            depth_crop = depth_frame[y1:y2, x1:x2].astype(np.float32) * depth_scale
            valid_d = depth_crop[(depth_crop > 0.05) & (depth_crop < 1.5)]
            if valid_d.size >= 100:
                # Background table is at the higher depth (further away)
                table_depth_est = float(np.percentile(valid_d, 85))
                # Physical object is elevated off table by >= 8mm (rejects table shadows and surface reflections)
                obj_elev = (depth_crop > 0.05) & (depth_crop < table_depth_est - 0.008)
                if np.sum(obj_elev) >= 50:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    crop_mask = cv2.morphologyEx((obj_elev * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        except Exception:
            crop_mask = None

    if crop_mask is None and frame is not None and frame.size > 0:
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 25, 90)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        corner_mean = (float(thresh[0,0]) + float(thresh[0,-1]) + float(thresh[-1,0]) + float(thresh[-1,-1])) / 4.0
        if corner_mean > 128.0:
            thresh = cv2.bitwise_not(thresh)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        crop_mask = cv2.morphologyEx(cv2.bitwise_or(edges, thresh), cv2.MORPH_CLOSE, kernel)

    # Baseline defaults
    base_roll = START_POS.get("wrist_roll.pos", 29.27) if "START_POS" in globals() else _BASE.get("wrist_roll.pos", 29.27)
    tilt_deg = 0.0
    recommended_roll = base_roll
    recommended_pitch = PREFERRED_GRAB_PITCH_DEG
    thinnest_width_mm = 30.0
    body_width_mm = float(w_box * scale)
    grasp_width_mm = 30.0
    h_calc = float(np.clip(h_box * scale, 80.0, 280.0))
    if is_bottle_like:
        if h_box >= int(w_box * 1.15):
            # Upright bottle: slim neck is at top 25% of height
            opt_x = default_cx
            opt_y = int(y1 + 0.25 * h_box)
            height_pct = 75.0
            recommended_pitch = 0.0
        else:
            # Lying flat bottle: default to 25% from side, top-down grasp
            opt_x = int(x1 + 0.25 * w_box)
            opt_y = default_cy
            height_pct = 50.0
            recommended_pitch = -85.0
    else:
        opt_x, opt_y = default_cx, y1 + max(5, int(h_box * 0.08))
        height_pct = 90.0

    nz_y, nz_x = np.where(crop_mask > 0) if (crop_mask is not None and crop_mask.size > 0) else ([], [])
    if len(nz_y) >= 35:
        # ── PCA Principal Component Analysis for 2-D Orientation ───────────────
        pts_crop = np.column_stack((nz_x, nz_y)).astype(np.float32)
        mean, eigenvectors = cv2.PCACompute(pts_crop, mean=None)
        c_x, c_y = float(mean[0][0]), float(mean[0][1])
        v_maj = eigenvectors[0]   # unit vector along object length
        v_min = eigenvectors[1]   # unit vector along object thickness

        # Ensure major axis points downwards in the image frame (v_maj_y >= 0)
        if v_maj[1] < 0:
            v_maj = -v_maj
            v_min = -v_min

        theta_maj = math.degrees(math.atan2(v_maj[1], v_maj[0]))   # 0° to 180°
        tilt_deg = theta_maj - 90.0                               # -90° to +90° (0° = vertical upright)

        # Project points onto PCA coordinates: s (along major axis), t (along minor axis)
        s = (pts_crop[:, 0] - c_x) * v_maj[0] + (pts_crop[:, 1] - c_y) * v_maj[1]
        t = (pts_crop[:, 0] - c_x) * v_min[0] + (pts_crop[:, 1] - c_y) * v_min[1]
        s_min, s_max = float(np.min(s)), float(np.max(s))
        L_px = s_max - s_min
        h_calc = float(np.clip(L_px * scale, 80.0, 280.0))

        # Oriented cross-sectional profiling in discrete bins along the major axis
        n_bins = 30
        bin_edges = np.linspace(s_min, s_max, n_bins + 1)
        widths_px = []
        mid_ts = []
        bin_centers = []
        for i in range(n_bins):
            in_b = (s >= bin_edges[i]) & (s < bin_edges[i+1])
            if np.sum(in_b) >= 3:
                t_b = t[in_b]
                delta_s = max(1.0, bin_edges[i+1] - bin_edges[i])
                w_fill = float(len(t_b)) / delta_s
                if len(t_b) >= 6:
                    p15, p85 = np.percentile(t_b, 15), np.percentile(t_b, 85)
                    w_perc = (p85 - p15) / 0.70
                    w_val = min(w_perc, w_fill) if w_fill > 3.0 else w_perc
                else:
                    w_val = w_fill
                widths_px.append(float(w_val))
                mid_ts.append(float(np.median(t_b)))
                bin_centers.append(float((bin_edges[i] + bin_edges[i+1]) / 2.0))
            else:
                widths_px.append(0.0)
                mid_ts.append(0.0)
                bin_centers.append(float((bin_edges[i] + bin_edges[i+1]) / 2.0))

        raw_mm = np.array(widths_px, dtype=np.float32) * scale
        widths_mm = np.copy(raw_mm)
        for i in range(1, n_bins - 1):
            widths_mm[i] = float(np.median(raw_mm[max(0, i-1):min(n_bins, i+2)]))

        valid_ws = [w for w in widths_mm if w > 5.0]
        body_width_mm = float(np.percentile(valid_ws, 80)) if valid_ws else float(w_box * scale)

        if is_bottle_like:
            # Robustly distinguish the cone/cap end from the flat base end:
            # End 1 candidate bins (bins 4 to 9, ~13% to 30% inward from s_min)
            # End 2 candidate bins (bins 20 to 25, ~13% to 30% inward from s_max)
            cands1 = [(widths_mm[i], bin_centers[i], mid_ts[i], i) for i in range(4, 10) if widths_mm[i] >= 12.0]
            cands2 = [(widths_mm[i], bin_centers[i], mid_ts[i], i) for i in range(20, 26) if widths_mm[i] >= 12.0]

            min_w1 = min(c[0] for c in cands1) if cands1 else 999.0
            min_w2 = min(c[0] for c in cands2) if cands2 else 999.0
            med_w1 = float(np.median([c[0] for c in cands1])) if cands1 else body_width_mm
            med_w2 = float(np.median([c[0] for c in cands2])) if cands2 else body_width_mm

            # A true bottle neck MUST be narrow (physically <= 45mm, typically 22-38mm).
            # The body/base of a bottle is physically 60-85mm.
            if min_w1 <= 45.0 and min_w2 > 45.0:
                cap_is_end1 = True
            elif min_w2 <= 45.0 and min_w1 > 45.0:
                cap_is_end1 = False
            elif min_w1 <= 45.0 and min_w2 <= 45.0:
                # Both have a narrow section: pick the one with the deeper neck constriction
                cap_is_end1 = (min_w1 <= min_w2)
            else:
                # Neither end measured <= 45mm: fallback to comparing median widths
                # If medians are close (within 15%), prefer the end closer to the robot (larger y in image)
                # to prevent lunging past the object into empty air!
                if abs(med_w1 - med_w2) < 0.15 * max(med_w1, med_w2):
                    y_end1 = y1 + c_y + s_min * v_maj[1]
                    y_end2 = y1 + c_y + s_max * v_maj[1]
                    cap_is_end1 = (y_end1 >= y_end2)
                else:
                    cap_is_end1 = (med_w1 <= med_w2)

            recommended_pitch = -85.0  # Steep top-down approach (perpendicular to ground)
            height_pct = 50.0

            # Once the cone/cap end is identified, locate the optimal grasp point right at the neck cylinder:
            if cap_is_end1:
                # Search bins 4 to 8 near End 1 for the neck cylinder (avoiding extreme cap tip in bins 0-3)
                neck_cands = [(widths_mm[i], bin_centers[i], mid_ts[i]) for i in range(4, 9) if widths_mm[i] >= 16.0]
                if neck_cands:
                    best_cand = min(neck_cands, key=lambda x: abs(x[0] - 28.0) if x[0] <= 45.0 else x[0])
                    thinnest_width_mm, s_grasp, mid_t_grasp = best_cand
                    s_grasp += 0.03 * L_px  # Inward bias (~6mm) into neck cylinder away from cap edge
                else:
                    s_grasp = s_min + 0.18 * L_px
                    mid_t_grasp = 0.0
                    thinnest_width_mm = med_w1
            else:
                # Search bins 21 to 25 near End 2 for the neck cylinder (avoiding extreme cap tip in bins 26-29)
                neck_cands = [(widths_mm[i], bin_centers[i], mid_ts[i]) for i in range(21, 26) if widths_mm[i] >= 16.0]
                if neck_cands:
                    best_cand = min(neck_cands, key=lambda x: abs(x[0] - 28.0) if x[0] <= 45.0 else x[0])
                    thinnest_width_mm, s_grasp, mid_t_grasp = best_cand
                    s_grasp -= 0.03 * L_px  # Inward bias (~6mm) into neck cylinder away from cap edge
                else:
                    s_grasp = s_max - 0.18 * L_px
                    mid_t_grasp = 0.0
                    thinnest_width_mm = med_w2

            grasp_width_mm = thinnest_width_mm
            # Optimal 2D grasp point centered on the neck cylinder (including minor-axis centerline):
            opt_x = int(np.clip(x1 + c_x + s_grasp * v_maj[0] + mid_t_grasp * v_min[0], x1 + 2, x2 - 2))
            opt_y = int(np.clip(y1 + c_y + s_grasp * v_maj[1] + mid_t_grasp * v_min[1], y1 + 2, y2 - 2))
        else:
            # Non-bottle objects: target centroid along major/minor axes
            opt_x = int(x1 + c_x)
            opt_y = int(y1 + c_y)
            grasp_width_mm = body_width_mm
            thinnest_width_mm = body_width_mm
            height_pct = 50.0
            abs_tilt = abs(tilt_deg)
            if abs_tilt <= 25.0:
                recommended_pitch = 0.0
            else:
                recommended_pitch = -85.0

        # Calculate recommended wrist_roll servo angle
        delta_roll = -tilt_deg
        recommended_roll = float(np.clip(base_roll + delta_roll, -170.0, 170.0))

    fits_gripper = (thinnest_width_mm <= GRIPPER_SAFE_CLEARANCE_MM)
    z_target_est = -130.0 + h_calc * (height_pct / 100.0) - 5.0

    delta_roll_val = -tilt_deg
    return {
        "body_width_mm": round(body_width_mm, 1),
        "thinnest_width_mm": round(thinnest_width_mm, 1),
        "grasp_width_mm": round(grasp_width_mm, 1),
        "height_mm": round(h_calc, 1),
        "height_pct": round(height_pct, 1),
        "fits_gripper": fits_gripper,
        "opt_px": (int(opt_x), int(opt_y)),
        "z_target_est": round(z_target_est, 1),
        "tilt_deg": round(tilt_deg, 1),
        "recommended_pitch_deg": round(recommended_pitch, 1),
        "recommended_roll_deg": round(recommended_roll, 1),
        "delta_roll_deg": round(delta_roll_val, 1)
    }


def find_optimal_grasp_point(seg_mask: np.ndarray | None, bbox: tuple[int, int, int, int], target_desc: str = "bottle", frame: np.ndarray | None = None, depth_frame: np.ndarray | None = None, depth_scale: float = 0.001) -> tuple[int, int]:
    """
    Computes optimal grasp point (x_px, y_px) on the object using PCA oriented profiling.
    Directly delegates to size_up_object with robust fallback to contour moments or Canny center.
    """
    x1, y1, x2, y2 = bbox
    default_cx, default_cy = (x1 + x2) // 2, (y1 + y2) // 2

    try:
        sizing = size_up_object(frame, seg_mask, bbox, 280.0, target_desc, depth_frame=depth_frame, depth_scale=depth_scale)
        return sizing["opt_px"]
    except Exception:
        pass

    # Non-bottle or sparse mask fallback
    if seg_mask is not None and np.sum(seg_mask == 255) >= 30:
        contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

    if frame is not None:
        return canny_centre(frame, x1, y1, x2, y2)
    return default_cx, default_cy


# ═══════════════════════════════════════════════════════════════════════════════
# 3D Depth Orientation, Diagonal Vectors & Closed-Form Optimal Roll
# ═══════════════════════════════════════════════════════════════════════════════
def compute_wrist_basis(pan_deg: float, pitch_deg: float):
    """
    Computes the 3D unit coordinate axes (X_wrist, Y_wrist, Z_wrist) of the arm wrist
    in the base coordinate frame for a given pan angle and approach pitch.
    """
    pan = math.radians(pan_deg)
    pitch = math.radians(pitch_deg)

    # Approach direction (Wrist X)
    ax = math.cos(pitch) * math.cos(pan)
    ay = math.cos(pitch) * math.sin(pan)
    az = math.sin(pitch)

    # Perpendicular horizontal direction (Wrist Z)
    zx = -math.sin(pan)
    zy =  math.cos(pan)
    zz = 0.0

    # Wrist Y = Z cross X
    yx = zy * az - zz * ay
    yy = zz * ax - zx * az
    yz = zx * ay - zy * ax

    X_wrist = np.array([ax, ay, az], dtype=np.float64)
    Y_wrist = np.array([yx, yy, yz], dtype=np.float64)
    Z_wrist = np.array([zx, zy, zz], dtype=np.float64)

    return X_wrist, Y_wrist, Z_wrist


def solve_optimal_roll(pan_deg: float, pitch_deg: float, V_obj: np.ndarray, base_roll_deg: float = 29.27) -> tuple[float, np.ndarray, float]:
    """
    Analytically solves the closed-form wrist_roll servo angle such that the
    gripper jaw opening axis J is strictly perpendicular to the 3D object axis V (J . V = 0).
    Also ensures J points toward the LEFT (away from the static pincer).
    Calibrated to physical robot hardware where base_roll_deg (+29.27°) is horizontally level.
    """
    X_w, Y_w, Z_w = compute_wrist_basis(pan_deg, pitch_deg)
    norm = np.linalg.norm(V_obj)
    V = (V_obj / norm) if norm > 1e-6 else np.array([1.0, 0.0, 0.0])

    dot_Y = float(np.dot(Y_w, V))
    dot_Z = float(np.dot(Z_w, V))

    # Analytical solution for J . V = 0:
    # J = cos(delta) * Z_w - sin(delta) * Y_w
    # J . V = cos(delta) * dot_Z - sin(delta) * dot_Y = 0 => delta = atan2(dot_Z, dot_Y)
    delta_candidate_rad = math.atan2(dot_Z, dot_Y)
    cand_delta_deg = math.degrees(delta_candidate_rad)

    cands_deg = [
        base_roll_deg + cand_delta_deg,
        base_roll_deg + cand_delta_deg + 180.0,
        base_roll_deg + cand_delta_deg - 180.0
    ]
    norm_cands = []
    for c in cands_deg:
        while c > 180.0: c -= 360.0
        while c < -180.0: c += 360.0
        norm_cands.append(c)

    valid = [c for c in norm_cands if -170.0 <= c <= 170.0]
    if not valid:
        valid = norm_cands

    best_roll = min(valid, key=lambda c: abs(c - base_roll_deg))
    delta = best_roll - base_roll_deg
    d_rad = math.radians(delta)
    # At delta=0, J points along Z_w (pure left: [-sin(pan), cos(pan), 0])
    # As wrist rolls by delta, J rotates in the Y_w/Z_w plane
    J = math.cos(d_rad) * Z_w - math.sin(d_rad) * Y_w
    norm_J = np.linalg.norm(J)
    if norm_J > 1e-6:
        J /= norm_J

    # Ensure J points to the LEFT (in positive Y / pan-left direction relative to reach)
    pan_rad = math.radians(pan_deg)
    left_ref = np.array([-math.sin(pan_rad), math.cos(pan_rad), 0.0])
    if float(np.dot(J, left_ref)) < 0.0:
        J = -J

    err = abs(float(np.dot(J, V)))
    return best_roll, J, err


def analyze_3d_orientation(cap, robot, bbox: tuple[int, int, int, int],
                           seg_mask: np.ndarray | None = None,
                           sizing: dict | None = None) -> dict:
    """
    Full 3D Orientation & Geometric Vector Analysis:
    Resolves objects at arbitrary 3D diagonal poses across (X, Y) and (X, Z) space.
    Accurately handles:
      1) Lying flat on tabletop (radial or angled 0°-90°)
      2) 3D diagonal vectors (angled in XY tabletop plane and tilted up in XZ elevation)
      3) Standing upright (or tilted propped against obstacles)

    Uses 3D PCA to extract the true 3D spatial axis vector V, determines the optimal
    approach pitch -(90° - elev), and computes the closed-form wrist roll with jaw vector J.
    """
    x1, y1, x2, y2 = bbox
    w_box = max(10, x2 - x1)
    h_box = max(10, y2 - y1)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # Baseline defaults
    base_roll = START_POS.get("wrist_roll.pos", 29.27) if "START_POS" in globals() else _BASE.get("wrist_roll.pos", 29.27)
    default_pitch = sizing.get("recommended_pitch_deg", PREFERRED_GRAB_PITCH_DEG) if sizing else PREFERRED_GRAB_PITCH_DEG
    default_roll  = sizing.get("recommended_roll_deg", base_roll) if sizing else base_roll
    default_delta = sizing.get("delta_roll_deg", 0.0) if sizing else 0.0

    # Determine 2D sampling axis from sizing or bounding box
    tilt_deg = sizing.get("tilt_deg", 0.0) if sizing else 0.0
    theta_rad = math.radians(tilt_deg + 90.0)
    ux = math.cos(theta_rad)
    uy = math.sin(theta_rad)

    # Half-span in pixels along the axis
    L_px = math.hypot(w_box, h_box) * 0.42

    # Sample strictly along the true object 2D principal axis (body centerline)
    # Never sample bounding box corners, which lie on the table surface!
    sample_fractions = [-0.35, -0.26, -0.18, -0.09, 0.0, 0.09, 0.18, 0.26, 0.35]
    sample_pixels = []
    for f in sample_fractions:
        px = int(np.clip(cx + f * L_px * ux, x1 + 2, x2 - 2))
        py = int(np.clip(cy + f * L_px * uy, y1 + 2, y2 - 2))
        sample_pixels.append((px, py))

    # Query 3D camera coordinates and convert to arm base frame
    valid_samples = []
    for px, py in sample_pixels:
        query_px, query_py = px, py
        if seg_mask is not None and seg_mask.size > 0:
            if seg_mask[py, px] == 0:
                y_lo, y_hi = max(0, py - 7), min(seg_mask.shape[0], py + 8)
                x_lo, x_hi = max(0, px - 7), min(seg_mask.shape[1], px + 8)
                sub = seg_mask[y_lo:y_hi, x_lo:x_hi]
                my, mx = np.where(sub == 255)
                if len(my) > 0:
                    query_px = x_lo + int(mx[len(mx) // 2])
                    query_py = y_lo + int(my[len(my) // 2])

        xyz_c = cap.get_xyz(query_px, query_py, search_w=16, search_h=16)
        if xyz_c is not None and 70.0 < xyz_c[2] <= MAX_GRAB_DEPTH_MM:
            pt_b = depth_to_arm_target(xyz_c, robot, penetration_mm=0.0, verbose=False)
            if pt_b is not None:
                valid_samples.append((query_px, query_py, xyz_c, pt_b))

    if len(valid_samples) < 2:
        return {
            "is_lying_flat": False,
            "is_3d_diagonal": False,
            "is_upright": True,
            "h_span_mm": 0.0,
            "v_span_mm": 0.0,
            "elev_deg": 90.0,
            "yaw_deg": 0.0,
            "target_pitch_deg": default_pitch,
            "target_roll_deg": default_roll,
            "delta_roll_deg": default_delta,
            "midpoint_base": None,
            "midpoint_cam": None,
            "V_axis": None,
            "J_vec": None,
            "confidence": "insufficient_3d_samples"
        }

    pts_b = np.array([s[3] for s in valid_samples], dtype=np.float64)
    pts_c = [s[2] for s in valid_samples]

    # 3D Centroid
    midpoint_b = np.mean(pts_b, axis=0)
    mid_x_b, mid_y_b, mid_z_b = float(midpoint_b[0]), float(midpoint_b[1]), float(midpoint_b[2])

    # 3D PCA to extract the true 3D spatial axis vector V of the cylinder
    centered = pts_b - midpoint_b
    if len(pts_b) >= 3:
        cov = np.cov(centered, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        V = evecs[:, -1]
    else:
        diff = pts_b[-1] - pts_b[0]
        norm_d = np.linalg.norm(diff)
        V = (diff / norm_d) if norm_d > 1e-4 else np.array([1.0, 0.0, 0.0])

    # Ensure V points upwards (V_z >= 0) or forward
    if V[2] < -1e-4:
        V = -V
    elif abs(V[2]) < 0.05 and V[0] < 0:
        V = -V

    dx_all = float(np.max(pts_b[:, 0]) - np.min(pts_b[:, 0]))
    dy_all = float(np.max(pts_b[:, 1]) - np.min(pts_b[:, 1]))
    h_span = math.hypot(dx_all, dy_all)
    v_span = float(np.max(pts_b[:, 2]) - np.min(pts_b[:, 2]))

    # 3D Elevation angle above horizontal table (0° = flat, 90° = vertical upright)
    V_xy = math.hypot(V[0], V[1])
    elev_deg = math.degrees(math.atan2(abs(V[2]), V_xy))

    # Tabletop yaw angle
    pan_rad = math.atan2(mid_y_b, mid_x_b)
    pan_deg = math.degrees(pan_rad)
    bottle_yaw_deg = math.degrees(math.atan2(V[1], V[0]))
    rel_table_deg = bottle_yaw_deg - pan_deg
    while rel_table_deg > 90.0:
        rel_table_deg -= 180.0
    while rel_table_deg < -90.0:
        rel_table_deg += 180.0

    # ── 3D Geometric Classification ──────────────────────────────────────────
    # - Upright bottle: stands vertically on tabletop (elev_deg >= 55°, v_span >= h_span * 1.15, abs(tilt_deg) < 30°)
    # - Lying flat bottle: rests horizontally on tabletop (elev_deg <= 40° or h_span >= v_span * 0.95 or abs(tilt_deg) >= 40° or w_box >= int(h_box * 1.15))
    # - 3D Diagonal bottle: genuinely propped up at an angle against an obstacle (elev_deg > 40° and v_span > h_span * 1.1 and abs(tilt_deg) < 40°)
    # ── 3D & 2D Unified Geometric Classification ─────────────────────────────
    # Upright standing bottle detection:
    # 1) 3D Point cloud metrics:
    is_upright_3d = (v_span >= 100.0 and v_span >= h_span * 1.2 and elev_deg >= 50.0)
    # 2) 2D Vision & Camera Gaze metrics:
    # A standing bottle is tall and vertical in the image: h_box >= 1.5 * w_box and |tilt| <= 22°
    # AND camera gaze is elevated (wst < +18° or lift > -92°) to view the tall standing object
    cur_j = get_pos(robot) if robot is not None else None
    wst_cur = cur_j.get("wrist_flex.pos", 32.53) if cur_j else 32.53
    lift_cur = cur_j.get("shoulder_lift.pos", -104.92) if cur_j else -104.92
    is_upright_2d = (h_box >= int(w_box * 1.5) and abs(tilt_deg) <= 22.0 and (wst_cur < 18.0 or lift_cur > -92.0))

    if is_upright_3d or is_upright_2d:
        is_upright = True
        is_lying_flat = False
        is_3d_diagonal = False
    elif (v_span <= 90.0) or (elev_deg <= 40.0) or (h_span >= v_span * 0.95) or (abs(tilt_deg) >= 40.0) or (w_box >= int(h_box * 1.15)):
        is_lying_flat = True
        is_upright = False
        is_3d_diagonal = False
    elif (v_span > 90.0 and 40.0 < elev_deg < 65.0):
        is_3d_diagonal = True
        is_upright = False
        is_lying_flat = False
    else:
        is_lying_flat = True
        is_upright = False
        is_3d_diagonal = False

    # Approach pitch calculation:
    # - Upright bottle: 0.0° (strictly parallel to ground)
    # Approach pitch calculation:
    # - Upright bottle: 0.0° (strictly parallel to ground, horizontal level approach)
    # - Lying flat bottle: -85.0° (steep top-down grasp from directly above, claw perpendicular to tabletop)
    # - 3D diagonal bottle: -(90° - elev_deg), clamped strictly to [-85.0°, -45.0°]
    if is_lying_flat:
        target_pitch = -85.0
        pose_desc = f"LYING FLAT (top-down grasp, tabletop tilt={tilt_deg:+.1f}°, yaw={rel_table_deg:+.1f}°)"
    elif is_3d_diagonal:
        target_pitch = -float(np.clip(90.0 - elev_deg, 45.0, 85.0))
        pose_desc = f"3D DIAGONAL (elev={elev_deg:.1f}°, yaw={rel_table_deg:+.1f}°)"
    else:
        target_pitch = 0.0
        pose_desc = "STANDING UPRIGHT"

    # Closed-form optimal wrist_roll and jaw axis J calculation:
    pan_rad = math.radians(pan_deg)
    if is_upright:
        target_roll = base_roll
        delta_roll = 0.0
        J_vec = np.array([-math.sin(pan_rad), math.cos(pan_rad), 0.0], dtype=np.float64)
    else:
        # For objects lying flat or diagonal on the tabletop:
        # In a top-down grasp, wrist_roll rotates the jaws in the horizontal tabletop plane.
        # Rotating wrist_roll by -tilt_deg aligns the gripper jaws perpendicular to the object.
        # Clamped to +/-65.0° for full rotational alignment with diagonal objects.
        delta_roll = float(np.clip(-tilt_deg, -65.0, 65.0))
        target_roll = base_roll + delta_roll

        # Compute true 3D jaw vector J in the robot base frame
        X_w, Y_w, Z_w = compute_wrist_basis(pan_deg, target_pitch)
        d_rad = math.radians(delta_roll)
        J_vec = math.cos(d_rad) * Z_w - math.sin(d_rad) * Y_w
        norm_J = np.linalg.norm(J_vec)
        if norm_J > 1e-6:
            J_vec /= norm_J
        left_ref = np.array([-math.sin(pan_rad), math.cos(pan_rad), 0.0])
        if float(np.dot(J_vec, left_ref)) < 0.0:
            J_vec = -J_vec

    # Slimmer neck identification and 3D grasp location
    target_px, target_py = (cx, cy)
    if sizing is not None and "opt_px" in sizing:
        target_px, target_py = sizing["opt_px"]

    # Direct 3D query at optimal grasp point (target_px, target_py):
    xyz_opt = cap.get_xyz(target_px, target_py, search_w=20, search_h=20)
    pt_opt_b = None
    if xyz_opt is not None and 70.0 < xyz_opt[2] <= MAX_GRAB_DEPTH_MM:
        pt_opt_b = depth_to_arm_target(xyz_opt, robot, penetration_mm=0.0, verbose=False)

    closest_sample = min(valid_samples, key=lambda s: math.hypot(s[0] - target_px, s[1] - target_py))
    midpoint_cam = xyz_opt if (xyz_opt is not None and 70.0 < xyz_opt[2] <= MAX_GRAB_DEPTH_MM) else closest_sample[2]

    # ── Live Table Floor Estimation from Real 3D Camera Depth ──────────────
    # Uses actual 3D points on the object resting on the table rather than a hardcoded height.
    # Use 15th percentile to reject single-pixel depth dropouts / shadow noise.
    z_15pct = float(np.percentile(pts_b[:, 2], 15)) if len(pts_b) >= 10 else float(np.min(pts_b[:, 2]))
    body_w = 70.0
    if sizing is not None:
        body_w = sizing.get("grasp_width_mm", sizing.get("body_width_mm", 70.0))
    cyl_dia = float(np.clip(body_w, 45.0, 85.0))
    # Table floor surface directly under the resting bottle (supports floor/low desks up to elevated risers):
    live_table_z = float(np.clip(z_15pct - 5.0, -250.0, -80.0))

    if is_upright:
        # Upright bottle: grasp at the slim neck (+35mm to +45mm above mid-height, within safe reach)
        neck_offset = min(45.0, 0.25 * max(v_span, 140.0))
        neck_z = max(mid_z_b + neck_offset, -90.0)
        safe_z_b = float(neck_z)
        midpoint_base = (mid_x_b, mid_y_b, safe_z_b)
    else:
        # Lying flat / diagonal bottle:
        # Target the slimmer cone/cap section identified by size_up_object:
        if pt_opt_b is not None:
            target_xy = (float(pt_opt_b[0]), float(pt_opt_b[1]))
            measured_z = float(pt_opt_b[2])
        elif sizing is not None and "opt_px" in sizing:
            target_xy = (float(closest_sample[3][0]), float(closest_sample[3][1]))
            measured_z = float(closest_sample[3][2])
        else:
            target_xy = (float(midpoint_b[0]), float(midpoint_b[1]))
            measured_z = float(midpoint_b[2])

        # Live vertical grasp height:
        # Bottle resting on tabletop/floor: the centerline of the neck cylinder is physically ~18mm above table surface.
        safe_z_b = live_table_z + 18.0
        safe_z_b = float(np.clip(safe_z_b, -240.0, -50.0))
        midpoint_base = (float(target_xy[0]), float(target_xy[1]), float(safe_z_b))

    print(f"   📐 3D Orientation: {pose_desc} | Elev={elev_deg:.1f}°, TableYaw={rel_table_deg:+.1f}° | H-Span={h_span:.0f}mm, V-Span={v_span:.0f}mm")
    print(f"   🎯 Grasp Command: Pitch={target_pitch:.1f}° | Wrist Roll={target_roll:.1f}° | Target Z={safe_z_b:.1f}mm | Live Table Z={live_table_z:.1f}mm")

    return {
        "is_lying_flat": is_lying_flat,
        "is_3d_diagonal": is_3d_diagonal,
        "is_upright": is_upright,
        "h_span_mm": round(h_span, 1),
        "v_span_mm": round(v_span, 1),
        "elev_deg": round(elev_deg, 1),
        "yaw_deg": round(rel_table_deg, 1),
        "target_pitch_deg": round(target_pitch, 1),
        "target_roll_deg": round(target_roll, 1),
        "delta_roll_deg": round(target_roll - base_roll, 1),
        "midpoint_base": midpoint_base,
        "midpoint_cam": midpoint_cam,
        "live_table_z": round(live_table_z, 1),
        "safe_z_b": round(safe_z_b, 1),
        "V_axis": V,
        "J_vec": J_vec,
        "confidence": "high_3d_span"
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Angle / Orientation-Invariant YOLO Inference
# ═══════════════════════════════════════════════════════════════════════════════
class DetectionBox:
    def __init__(self, xyxy, cls_id, conf=0.8):
        self.xyxy = np.array([xyxy], dtype=np.float32)
        self.cls = np.array([cls_id], dtype=np.float32)
        self.conf = np.array([conf], dtype=np.float32)

class DetectionMasks:
    def __init__(self, polygons):
        self.xy = [np.array(p, dtype=np.float32) for p in polygons]

class DetectionResult:
    def __init__(self, boxes, masks=None):
        self.boxes = boxes
        self.masks = masks


def derotate_detections(raw_result, rot_type: str, orig_w: int, orig_h: int) -> DetectionResult:
    """
    Transforms bounding boxes and segmentation masks from rotated inference image
    back into the original camera image coordinate frame with exact pixel precision.
    """
    if raw_result is None or not hasattr(raw_result, 'boxes') or raw_result.boxes is None or len(raw_result.boxes) == 0:
        return DetectionResult([])

    new_boxes = []
    new_masks_xy = []
    has_masks = hasattr(raw_result, 'masks') and raw_result.masks is not None and hasattr(raw_result.masks, 'xy')

    for i, box in enumerate(raw_result.boxes):
        bx1, by1, bx2, by2 = map(float, box.xyxy[0].tolist())
        cls_id = int(box.cls[0].item())
        conf_val = float(box.conf[0].item()) if hasattr(box, 'conf') and len(box.conf) > 0 else 0.8

        poly = None
        if has_masks and i < len(raw_result.masks.xy):
            poly = np.array(raw_result.masks.xy[i], dtype=np.float32)

        if rot_type == "rot90_cw":
            pts = np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]], dtype=np.float32)
            ox = pts[:, 1]
            oy = (orig_h - 1) - pts[:, 0]
            nx1, ny1, nx2, ny2 = float(np.min(ox)), float(np.min(oy)), float(np.max(ox)), float(np.max(oy))
            if poly is not None and len(poly) > 0:
                new_poly = np.column_stack([poly[:, 1], (orig_h - 1) - poly[:, 0]])
                new_masks_xy.append(new_poly)

        elif rot_type == "rot180":
            nx1 = float((orig_w - 1) - bx2)
            ny1 = float((orig_h - 1) - by2)
            nx2 = float((orig_w - 1) - bx1)
            ny2 = float((orig_h - 1) - by1)
            if poly is not None and len(poly) > 0:
                new_poly = np.column_stack([(orig_w - 1) - poly[:, 0], (orig_h - 1) - poly[:, 1]])
                new_masks_xy.append(new_poly)

        elif rot_type == "rot90_ccw":
            pts = np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]], dtype=np.float32)
            ox = (orig_w - 1) - pts[:, 1]
            oy = pts[:, 0]
            nx1, ny1, nx2, ny2 = float(np.min(ox)), float(np.min(oy)), float(np.max(ox)), float(np.max(oy))
            if poly is not None and len(poly) > 0:
                new_poly = np.column_stack([(orig_w - 1) - poly[:, 1], poly[:, 0]])
                new_masks_xy.append(new_poly)

        elif rot_type in ("rot45", "rot_neg45"):
            angle = 45.0 if rot_type == "rot45" else -45.0
            M_inv = cv2.getRotationMatrix2D((orig_w / 2.0, orig_h / 2.0), -angle, 1.0)
            pts = np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]], dtype=np.float32)
            orig_pts = cv2.transform(pts.reshape(1, -1, 2), M_inv)[0]
            nx1 = float(np.clip(np.min(orig_pts[:, 0]), 0, orig_w - 1))
            ny1 = float(np.clip(np.min(orig_pts[:, 1]), 0, orig_h - 1))
            nx2 = float(np.clip(np.max(orig_pts[:, 0]), 0, orig_w - 1))
            ny2 = float(np.clip(np.max(orig_pts[:, 1]), 0, orig_h - 1))
            if poly is not None and len(poly) > 0:
                new_poly = cv2.transform(poly.reshape(1, -1, 2), M_inv)[0]
                new_masks_xy.append(new_poly)
        else:
            nx1, ny1, nx2, ny2 = bx1, by1, bx2, by2
            if poly is not None:
                new_masks_xy.append(poly)

        new_boxes.append(DetectionBox([nx1, ny1, nx2, ny2], cls_id, conf_val))

    masks_obj = DetectionMasks(new_masks_xy) if has_masks and len(new_masks_xy) > 0 else None
    return DetectionResult(new_boxes, masks_obj)


def detect_with_rotation(model, frame: np.ndarray, conf: float = 0.50, iou: float = 0.45, target_class_ids=None):
    """
    Multi-angle rotation-aware inference.
    1. First tries normal 0° upright inference. If target is detected, returns immediately (zero overhead).
    2. If no target is detected (e.g. bottle is lying on its side, upside down, or tilted),
       tests rotated views where the object appears upright to standard YOLO weights.
    3. De-rotates detected bounding boxes and masks back to the original image coordinate frame.
    Guarantees detection at any angle without lowering confidence or adding extra vocabulary!
    """
    if model is None or frame is None or frame.size == 0:
        return [DetectionResult([])]

    # Step 1: Normal upright inference
    results = model(frame, verbose=False, conf=conf, iou=iou)

    target_found = False
    if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
        if target_class_ids is None:
            target_found = True
        else:
            for box in results[0].boxes:
                if int(box.cls[0].item()) in target_class_ids:
                    target_found = True
                    break

    if target_found:
        return results

    # Step 2: Test rotated views where non-upright objects appear upright to YOLO
    orig_h, orig_w = frame.shape[:2]
    rotations = [
        ("rot90_cw", lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
        ("rot90_ccw", lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ("rot180", lambda img: cv2.rotate(img, cv2.ROTATE_180)),
        ("rot45", lambda img: cv2.warpAffine(img, cv2.getRotationMatrix2D((orig_w / 2.0, orig_h / 2.0), 45.0, 1.0), (orig_w, orig_h))),
        ("rot_neg45", lambda img: cv2.warpAffine(img, cv2.getRotationMatrix2D((orig_w / 2.0, orig_h / 2.0), -45.0, 1.0), (orig_w, orig_h))),
    ]

    for rot_name, rot_fn in rotations:
        rot_frame = rot_fn(frame)
        res_rot = model(rot_frame, verbose=False, conf=conf, iou=iou)
        if res_rot and len(res_rot) > 0 and res_rot[0].boxes is not None and len(res_rot[0].boxes) > 0:
            has_match = False
            for box in res_rot[0].boxes:
                if target_class_ids is None or int(box.cls[0].item()) in target_class_ids:
                    has_match = True
                    break
            if has_match:
                derotated_res = derotate_detections(res_rot[0], rot_name, orig_w, orig_h)
                return [derotated_res]

    return results


def compute_iou(boxA, boxB) -> float:
    """Computes Intersection-over-Union (IoU) of two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, float(xB - xA)) * max(0.0, float(yB - yA))
    areaA = max(1.0, float((boxA[2] - boxA[0]) * (boxA[3] - boxA[1])))
    areaB = max(1.0, float((boxB[2] - boxB[0]) * (boxB[3] - boxB[1])))
    return inter / float(areaA + areaB - inter)


# ═══════════════════════════════════════════════════════════════════════════════
# ObjectTracker: Persistent Spatial-Temporal Memory & Geometric Centroid
# ═══════════════════════════════════════════════════════════════════════════════
class ObjectTracker:
    """
    Robust Semantic Object Tracker with Persistent Spatial-Temporal Memory.
    Features:
      1) Exponential Moving Average (EMA) bounding box smoothing across frames.
      2) Combined IoU + spatial proximity association (rejects background clutter and jumps).
      3) Stable geometric centroid: (cx, cy) = ((x1+x2)/2, (y1+y2)/2).
         Never re-runs noisy Otsu/Canny edge thresholds on every frame!
      4) Long-term coasting memory: holds target in memory across scene shifts/arm lunges
         for up to 15 frames (~1.0s).
      5) Locked target coordinates invariant to camera ego-motion.
    """
    def __init__(self, ema_alpha: float = 0.65, max_memory_frames: int = 15):
        self.smooth_box = None          # [x1, y1, x2, y2] float
        self.smooth_center = None       # (cx, cy) int
        self.last_box = None            # (x1, y1, x2, y2) int
        self.last_center = None         # (cx, cy) int
        self.last_area = None
        self.last_mask_polygon = None   # np.ndarray mask polygon
        self.locked = False
        self.lost_frames = 0
        self.ema_alpha = ema_alpha
        self.max_memory_frames = max_memory_frames
        self.confirmed_frames = 0

    def lock_on(self, frame, x1, y1, x2, y2):
        fh, fw = frame.shape[:2]
        self.smooth_box = [float(x1), float(y1), float(x2), float(y2)]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        self.smooth_center = (cx, cy)
        self.last_box = (int(x1), int(y1), int(x2), int(y2))
        self.last_center = (cx, cy)
        self.last_area = ((x2 - x1) / fw) * ((y2 - y1) / fh)
        self.last_mask_polygon = None
        self.locked = True
        self.lost_frames = 0
        self.confirmed_frames = 1
        print(f"   🔒 Tracker locked on target: box=({x1},{y1},{x2},{y2}) stable_center={self.smooth_center}")

    def detect(self, frame, yolo_results, model_class_id):
        fh, fw = frame.shape[:2]
        valid_candidates = []

        if yolo_results is not None and len(yolo_results) > 0 and yolo_results[0].boxes is not None:
            has_masks = hasattr(yolo_results[0], 'masks') and yolo_results[0].masks is not None and hasattr(yolo_results[0].masks, 'xy')
            for box_idx, box in enumerate(yolo_results[0].boxes):
                cls_id = int(box.cls[0].item())
                if isinstance(model_class_id, (list, tuple, set)):
                    if cls_id not in model_class_id:
                        continue
                else:
                    if cls_id != model_class_id:
                        continue

                bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(fw, bx2), min(fh, by2)
                cand_box = [float(bx1), float(by1), float(bx2), float(by2)]
                cand_cx = (bx1 + bx2) / 2.0
                cand_cy = (by1 + by2) / 2.0
                cand_area = ((bx2 - bx1) / fw) * ((by2 - by1) / fh)
                poly = yolo_results[0].masks.xy[box_idx] if (has_masks and box_idx < len(yolo_results[0].masks.xy)) else None

                if self.smooth_box is not None:
                    iou = compute_iou(self.smooth_box, cand_box)
                    cur_cx = (self.smooth_box[0] + self.smooth_box[2]) / 2.0
                    cur_cy = (self.smooth_box[1] + self.smooth_box[3]) / 2.0
                    dist = math.hypot(cand_cx - cur_cx, cand_cy - cur_cy)

                    if iou >= 0.10 or dist < 250.0:
                        score = iou * 0.65 + max(0.0, 1.0 - (dist / float(fw))) * 0.35
                        valid_candidates.append((score, cand_box, cand_area, poly))
                else:
                    valid_candidates.append((1.0, cand_box, cand_area, poly))

        if valid_candidates:
            valid_candidates.sort(key=lambda item: item[0], reverse=True)
            best_score, best_box, best_area, best_poly = valid_candidates[0]
            if best_poly is not None:
                self.last_mask_polygon = best_poly

            if self.smooth_box is not None:
                a = self.ema_alpha
                self.smooth_box = [
                    a * best_box[0] + (1.0 - a) * self.smooth_box[0],
                    a * best_box[1] + (1.0 - a) * self.smooth_box[1],
                    a * best_box[2] + (1.0 - a) * self.smooth_box[2],
                    a * best_box[3] + (1.0 - a) * self.smooth_box[3],
                ]
            else:
                self.smooth_box = list(best_box)

            cx = int((self.smooth_box[0] + self.smooth_box[2]) / 2.0)
            cy = int((self.smooth_box[1] + self.smooth_box[3]) / 2.0)
            self.smooth_center = (cx, cy)
            self.last_box = (int(round(self.smooth_box[0])), int(round(self.smooth_box[1])),
                             int(round(self.smooth_box[2])), int(round(self.smooth_box[3])))
            self.last_center = (cx, cy)
            self.last_area = best_area
            self.lost_frames = 0
            self.confirmed_frames += 1
            return True, self.smooth_center, self.last_area, "yolo"

        # Coasting / Memory Hold across scene shifts and arm motion
        if self.locked and self.smooth_box is not None and self.lost_frames < self.max_memory_frames:
            self.lost_frames += 1
            cx = int((self.smooth_box[0] + self.smooth_box[2]) / 2.0)
            cy = int((self.smooth_box[1] + self.smooth_box[3]) / 2.0)
            self.smooth_center = (cx, cy)
            self.last_center = (cx, cy)
            return True, self.smooth_center, self.last_area, "hold"

        self.lost_frames += 1
        return False, self.smooth_center, self.last_area, "lost"


# ═══════════════════════════════════════════════════════════════════════════════
# Visual servoing alignment loop
# ═══════════════════════════════════════════════════════════════════════════════
def align_arm(robot, cap: RealSenseStream, model,
              tracker: ObjectTracker,
              max_frames: int | None = None) -> tuple[bool, tuple[int, int] | None]:
    """
    Closed-loop visual servoing with active 3-DOF tracking (pan, wrist-flex tilt, shoulder lift).
    Follows target smoothly in real-time as it moves or changes elevation.
    Returns (success, (obj_px, obj_py)).
    """
    centred_streak = 0
    lost_streak    = 0
    sign_flip_pan  = 0
    last_pan_sign  = 0
    last_obj_pixel = None
    effective_max  = max_frames if max_frames is not None else ALIGN_MAX_FRAMES

    # Record initial posture
    initial_pos = get_pos(robot)
    start_pan   = initial_pos.get("shoulder_pan.pos", 0.0)
    start_lift  = initial_pos.get("shoulder_lift.pos", 0.0)
    start_elb   = initial_pos.get("elbow_flex.pos", 0.0)
    start_wst   = initial_pos.get("wrist_flex.pos", 0.0)
    start_grp   = initial_pos.get("gripper.pos", 60.0)

    for frame_idx in range(effective_max):
        color, has_depth, depth_colormap = cap.read()
        h, w     = color.shape[:2]
        frame_cx = (w // 2) + ALIGN_PAN_OFFSET
        frame_cy =  h // 2

        results = detect_with_rotation(model, color, conf=YOLO_CONF, iou=0.45, target_class_ids=TARGET_CLASS_IDS)
        found, center, area, src = tracker.detect(color, results, TARGET_CLASS_IDS)

        display = color.copy()
        cv2.line(display, (frame_cx-20, frame_cy), (frame_cx+20, frame_cy), (0,0,255), 1)
        cv2.line(display, (frame_cx, frame_cy-20), (frame_cx, frame_cy+20), (0,0,255), 1)
        cv2.circle(display, (frame_cx, frame_cy), 6, (0,0,255), -1)

        if not found:
            lost_streak += 1
            cv2.putText(display,
                f"🎯 ALIGNING — lost ({lost_streak}/{ALIGN_LOST_GRACE})",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,100,255), 2)
            _show_frame("Picker Vision", display)
            if not HEADLESS: cv2.waitKey(1)
            if lost_streak >= ALIGN_LOST_GRACE:
                if last_obj_pixel is not None:
                    last_pan_err = last_obj_pixel[0] - frame_cx
                    last_lift_err = last_obj_pixel[1] - frame_cy
                    if abs(last_pan_err) <= 50 and abs(last_lift_err) <= 50:
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

        err_mag = math.hypot(pan_err, lift_err)
        # Responsive velocity scaling: fast when far, smooth linear deceleration within 30px
        decay = float(np.clip(err_mag / 30.0, 0.35, 1.0))

        # Command calculations
        pan_cmd   = float(np.clip(pan_err * ALIGN_PAN_K * decay, -ALIGN_MAX_PAN_DEG, ALIGN_MAX_PAN_DEG))
        # Positive lift_err (obj_py > cy, object lower) -> wrist pitches down (wst increases)
        # Negative lift_err (obj_py < cy, object higher) -> wrist pitches up (wst decreases)
        wrist_cmd = float(np.clip(lift_err * ALIGN_WRIST_K * decay, -ALIGN_MAX_WRIST_DEG, ALIGN_MAX_WRIST_DEG))
        # Negative lift_err (object higher) -> shoulder lifts up (lift becomes less negative, so + cmd)
        lift_cmd  = float(np.clip(-lift_err * ALIGN_LIFT_K * decay, -ALIGN_MAX_LIFT_DEG, ALIGN_MAX_LIFT_DEG))

        cur = get_pos(robot)
        cur_pan  = cur.get("shoulder_pan.pos", start_pan)
        cur_lift = cur.get("shoulder_lift.pos", start_lift)
        cur_wst  = cur.get("wrist_flex.pos", start_wst)

        next_pan  = float(np.clip(cur_pan + pan_cmd, PAN_MIN_DEG, PAN_MAX_DEG))
        next_wst  = float(np.clip(cur_wst + wrist_cmd, -35.0, 55.0))
        # Safe bounds on shoulder_lift: avoid excessive backward lean (below start_lift - 5.0)
        # while allowing upward reach (up to start_lift + 22.0)
        next_lift = float(np.clip(cur_lift + lift_cmd, start_lift - 5.0, start_lift + 22.0))

        centred_pan  = abs(pan_err) < ALIGN_THRESHOLD
        centred_lift = abs(lift_err) < ALIGN_THRESHOLD
        centred = centred_pan and centred_lift

        if centred:
            centred_streak += 1
            if centred_streak >= ALIGN_CENTRED_NEED:
                print(f"   ✅ Aligned in {frame_idx+1} frames (pan={pan_err:+d}px, lift={lift_err:+d}px | wst={next_wst:+.1f}°, lift={next_lift:+.1f}°)")
                return True, last_obj_pixel
        else:
            centred_streak = 0

        robot.send_action({
            "shoulder_pan.pos":  next_pan,
            "shoulder_lift.pos": next_lift,
            "elbow_flex.pos":    start_elb,
            "wrist_flex.pos":    next_wst,
            "gripper.pos":       start_grp,
        })

        print(f"   [{src.upper():5s}] frame {frame_idx+1:03d}: "
              f"err=({pan_err:+4d},{lift_err:+4d})px  "
              f"cmd=(pan={pan_cmd:+4.1f}°, wst={wrist_cmd:+4.1f}°, lift={lift_cmd:+4.1f}°)  "
              f"pose=(wst={next_wst:+4.1f}°, lift={next_lift:+4.1f}°)")

        src_color = (0,255,0) if src == "yolo" else ((0,255,255) if src == "hold" else (0,165,255))
        cv2.circle(display, (obj_px, obj_py), 7, src_color, -1)
        cv2.line(display, (frame_cx, frame_cy), (obj_px, obj_py), (0,255,255), 1)
        status = "✅ CENTRED" if centred else "🎯 TRACKING"
        cv2.putText(display,
            f"{status} [{src.upper()}]  err=({pan_err:+d},{lift_err:+d})px  wst={next_wst:+.0f}deg",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        _show_frame("Picker Vision", display)
        if not HEADLESS: cv2.waitKey(1)

    if last_obj_pixel is not None:
        print(f"   ⏳ Alignment hardcap reached ({effective_max} frames). Proceeding with best alignment.")
        return True, last_obj_pixel
    else:
        print(f"   ❌ Alignment timed out and no object found — restarting")
        return False, None


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


def run_manual_calibration(robot):
    """
    Cuts motor torque, allowing the user to move the arm manually by hand.
    Continuously measures joint angles, calculates FK (X, Y, Z), tracks
    minimum/maximum joint angles and maximum extension reached.
    """
    print("\n" + "═"*65)
    print(" 🛠️  MANUAL MOVEMENT & RANGE-OF-MOTION EXPLORATION MODE")
    print("═"*65)
    print(" ⚠️  Disabling motor torque NOW — please support the arm with your hand!")
    time.sleep(0.5)
    _set_torque(robot, False)
    print(" 🔓 Motor torque DISABLED. You can now move the arm manually by hand.")
    print(" Move joints to explore reach, or hold at a target pose and press Ctrl+C to finish.")
    print("═"*65 + "\n")

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
    print("   [3] Record / Teach Reference Postures (Scan & Stow) [Torque OFF]")
    print("   [4] Test Full Automatic Pick-and-Place (Vision + IK Lunge)")
    print("   [5] Exit")
    print("═"*65)
    choice = input("Enter choice (1-5): ").strip()

    if choice == "2":
        show_workspace_limits()
        do_move = input("Do you want to physically move the arm to these boundary edges now? (y/n): ").strip().lower()
        if do_move.startswith("y"):
            robot = connect_robot()
            try:
                demonstrate_workspace_boundaries(robot)
            finally:
                robot.disconnect()
        return
    elif choice == "3":
        robot = connect_robot()
        try:
            print("\n [1] Teach / Record Reference Postures (Scan & Stow) [Recommended]")
            print(" [2] Free-move Workspace & Reach Exploration")
            sub = input("Choose (1 or 2) [default 1]: ").strip()
            if sub == "2":
                run_manual_calibration(robot)
            else:
                teach_postures(robot)
        finally:
            robot.disconnect()
        return
    elif choice == "5":
        print("Exiting.")
        return
    elif choice not in ["1", "4"]:
        print("Invalid selection. Exiting.")
        return
        
    do_manual_lunge = (choice == "1")

    # ── Verify graphical display or switch to headless cleanly ────────────────
    _init_display_mode()

    print("🚀 Initialising RealSense D405 + IK Pick-and-Place Pipeline...")

    global T_CAM_WRIST
    T_CAM_WRIST = _build_T_cam_wrist()
    print(f"📐 T_cam_wrist built:")
    print(f"   offsets X={CAM_X_OFFSET_MM}mm  Y={CAM_Y_OFFSET_MM}mm  Z={CAM_Z_OFFSET_MM}mm  pitch={CAM_PITCH_DEG}°")

    global TARGET_DESC, YOLO_CLASS_ID, TARGET_CLASS_IDS
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
    try:
        if hasattr(model, "model") and model.model is not None:
            model.model.is_fused = lambda: True
            model.model.fuse = lambda *args, **kwargs: model.model
        if hasattr(model, "fuse"):
            model.fuse = lambda *args, **kwargs: model
    except Exception:
        pass

    if "world" in chosen_path.lower() or hasattr(model, "set_classes"):
        user_target = TARGET_DESC.strip()
        query_variants = [user_target]
        model.set_classes(query_variants)
        TARGET_CLASS_IDS = list(range(len(query_variants)))
        YOLO_CLASS_ID = 0  # Primary class ID
        print(f"   🌍 YOLO-World loaded and targeting: {query_variants} (Class IDs: {TARGET_CLASS_IDS})")
    else:
        TARGET_CLASS_IDS = [YOLO_CLASS_ID]
        print(f"   🎯 Standard YOLO targeting class {YOLO_CLASS_ID} ('{TARGET_DESC}')")

    try:
        if hasattr(model, "model") and model.model is not None:
            model.model.is_fused = lambda: True
            model.model.fuse = lambda *args, **kwargs: model.model
        if hasattr(model, "fuse"):
            model.fuse = lambda *args, **kwargs: model
    except Exception:
        pass

    # ── YOLO GPU warmup BEFORE arm connect ───────────────────────────────────
    # The first inference triggers CUDA/model warmup (can take 2-5 s).  If this
    # happens after the arm is already powered and sitting at start pose, the
    # servos are left unmonitored during a blocking GPU call.  Worse, a Ctrl+C
    # during warmup fires a signal into live C/CUDA code → fatal heap corruption.
    # Warm the model NOW while servos are still off the bus.
    print("   🔥 Warming up YOLO model on GPU (first inference)...")
    _dummy = np.zeros((480, 848, 3), dtype=np.uint8)
    try:
        model(_dummy, verbose=False, conf=YOLO_CONF)
    except Exception:
        pass  # ignore warmup errors — just ensure cuda graph is built
    print("   ✅ YOLO warmup complete")

    # ── Robot arm ─────────────────────────────────────────────────────────────
    try:
        robot = connect_robot()
    except Exception as e:
        print(f"❌ Arm connection / health check failed: {e}")
        return

    # Refresh reference postures (in case taught recently)
    _load_reference_poses()
    START_POS = dict(_BASE)
    STOW      = dict(_STOW_BASE)
    print(f"📍 Start Position : Pan={START_POS.get('shoulder_pan.pos',0):+.1f}° Lift={START_POS.get('shoulder_lift.pos',0):+.1f}° Elb={START_POS.get('elbow_flex.pos',0):+.1f}°")
    print(f"📍 Stow Position  : Pan={STOW.get('shoulder_pan.pos',0):+.1f}° Lift={STOW.get('shoulder_lift.pos',0):+.1f}° Elb={STOW.get('elbow_flex.pos',0):+.1f}°")


    _shutdown_requested = [False]  # mutable flag safe to set from signal handler

    def handle_exit(sig, frame):
        """SIGINT handler — only sets a flag; never calls into robot/C-ext directly.
        Calling smooth_move() from a signal handler that fires mid-C-extension
        (e.g. torch.linalg) corrupts Python's internal state → fatal abort."""
        if not _shutdown_requested[0]:
            print("\n📍 Shutdown requested — finishing current step and stowing...")
            _shutdown_requested[0] = True
        # Re-raise KeyboardInterrupt so the main try/except loop exits cleanly.
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, handle_exit)

    # ── RealSense D405 ────────────────────────────────────────────────────────
    print("📷 Connecting to RealSense D405...")
    cap = RealSenseStream()
    time.sleep(2.0)   # let first frames arrive and intrinsics populate

    # ── Move to start ─────────────────────────────────────────────────────────
    # Use smaller steps + longer delay for the FIRST move — freshly powered
    # servos can trip overload protection if commanded too aggressively from
    # an unknown starting position.
    print("\n▶ Moving to Start Position (slow start)...")
    smooth_move(robot, START_POS, step_size=1.0, step_delay=0.05)
    time.sleep(1.0)


    last_yolo_t = 0.0
    STATE       = "SEARCHING"
    sweep_dir   = 1.0
    sweep_pan   = START_POS["shoulder_pan.pos"]

    _arm_is_stowed = [False]

    # ── Emergency stow ────────────────────────────────────────────────────────
    def _emergency_stow():
        if _arm_is_stowed[0]:
            return
        _arm_is_stowed[0] = True
        print("\n⚠️  Emergency stow triggered...")
        try:
            # Probe the arm first — if it's dead (power-lost / disconnected)
            # smooth_move will raise ConnectionError and corrupt state further.
            get_pos(robot)
            smooth_move(robot, STOW, step_size=0.8, step_delay=0.025)
            time.sleep(0.3)
        except Exception as e:
            print(f"   Stow skipped (arm unreachable): {e}")
        try:
            robot.disconnect()
        except Exception:
            pass
        try:
            cap.stop()
            _destroy_windows()
        except Exception:
            pass
    atexit.register(_emergency_stow)


    _last_fps_t = [time.time()]
    _fps_smoothed = [15.0]

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

            # ── Live FPS calculation and overlay ──────────────────────────────
            t_now = time.time()
            dt = t_now - _last_fps_t[0]
            _last_fps_t[0] = t_now
            if dt > 0:
                _fps_smoothed[0] = 0.9 * _fps_smoothed[0] + 0.1 * (1.0 / dt)
            cv2.putText(display, f"FPS: {_fps_smoothed[0]:.1f}",
                        (w - 160, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)


            if STATE == "SEARCHING":
                cv2.putText(display, f"STATE: {STATE} | YOLO: {TARGET_DESC}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # ── Throttle YOLO to ~5 fps during search ─────────────────────────
            if time.time() - last_yolo_t < 0.2:
                _show_frame("Picker Vision", _make_vis(display, depth_colormap))
                if not HEADLESS: cv2.waitKey(1)
                continue
            last_yolo_t = time.time()

            results = detect_with_rotation(model, color, conf=YOLO_CONF, iou=0.45, target_class_ids=TARGET_CLASS_IDS)
            
            # ── Draw YOLO detections ──────────────────────────────────────────
            target_box     = None
            target_box_idx = -1
            for i, box in enumerate(results[0].boxes):
                if int(box.cls[0].item()) in TARGET_CLASS_IDS:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                    
                    # Check if the bounding box has ANY valid depth inside it
                    cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
                    box_w, box_h = max(10, bx2 - bx1), max(10, by2 - by1)
                    
                    depth_sample = cap.get_xyz(cx, cy, search_w=box_w//2, search_h=box_h//2)
                    if depth_sample is None:
                        # Grid sample across box for diagonal or tilted objects
                        grid_samples = [
                            (bx1 + box_w // 4, by1 + box_h // 4),
                            (bx1 + 3 * box_w // 4, by1 + box_h // 4),
                            (bx1 + box_w // 4, by1 + 3 * box_h // 4),
                            (bx1 + 3 * box_w // 4, by1 + 3 * box_h // 4),
                            (bx1 + box_w // 2, by1 + box_h // 4),
                            (bx1 + box_w // 2, by1 + 3 * box_h // 4),
                        ]
                        for sx, sy in grid_samples:
                            depth_sample = cap.get_xyz(sx, sy, search_w=max(10, box_w // 3), search_h=max(10, box_h // 3))
                            if depth_sample is not None:
                                break

                    if depth_sample is None:
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
                if STATE == "SEARCHING" and SWEEP_ON_SEARCH:
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
                
                _show_frame("Picker Vision", _make_vis(display, depth_colormap))
                if not HEADLESS: cv2.waitKey(1)
                continue


            # ── YOLO candidate found ──────────────────────────────────────────
            x1, y1, x2, y2 = map(int, target_box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Use segmentation mask optimal grasp point if available (e.g. neck/cap of bottle)
            seg_mask = None
            if (results[0].masks is not None and
                    target_box_idx >= 0 and
                    target_box_idx < len(results[0].masks.xy)):
                polygon  = results[0].masks.xy[target_box_idx].astype(np.int32)
                seg_mask = np.zeros(color.shape[:2], dtype=np.uint8)
                cv2.fillPoly(seg_mask, [polygon], 255)
                obj_px, obj_py = find_optimal_grasp_point(seg_mask, (x1, y1, x2, y2), TARGET_DESC, frame=color)
            else:
                obj_px, obj_py = find_optimal_grasp_point(None, (x1, y1, x2, y2), TARGET_DESC, frame=color)

            cv2.rectangle(display, (x1, y1), (x2, y2), (0,255,255), 2)
            cv2.circle(display, (obj_px, obj_py), 6, (0,255,0), -1)
            cv2.putText(display, f"YOLO: {TARGET_DESC}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            
            _show_frame("Picker Vision", _make_vis(display, depth_colormap))
            if not HEADLESS: cv2.waitKey(1)

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
            print(f"🎯 Aligning arm to {TARGET_DESC} (scan pose)...")
            tracker = ObjectTracker()
            tracker.lock_on(color, x1, y1, x2, y2)
            aligned, final_pixel = align_arm(robot, cap, model, tracker)
            if not aligned or final_pixel is None:
                print("❌ Alignment failed — restarting search")
                smooth_move(robot, START_POS, step_size=2.0, step_delay=0.03)
                STATE = "SEARCHING"
                continue

            obj_px, obj_py = final_pixel

            print(f"✅ Target ({TARGET_DESC}) centred — calculating direct trajectory...")
            STATE = "GRABBING"
            print("\n🎯 Reading depth for IK from scan posture...")

            # ── Step 2a: Settle & Consensus Burst using Tracker Memory ────────
            # Give servos 150ms to settle to a mechanical standstill after visual servoing.
            time.sleep(0.15)

            # Consensus burst across 3 consecutive frames:
            # Updates tracker.smooth_box via ObjectTracker.detect, maintaining persistent
            # spatial-temporal track across the scene shift!
            color_aligned = None
            seg_mask = None
            best_box = tracker.last_box if tracker.last_box is not None else (x1, y1, x2, y2)

            for _ in range(3):
                c_snap, _, _ = cap.read()
                if c_snap is not None:
                    color_aligned = c_snap
                    res_snap = detect_with_rotation(model, color_aligned, conf=YOLO_CONF, iou=0.45, target_class_ids=TARGET_CLASS_IDS)
                    found_snap, _, _, _ = tracker.detect(color_aligned, res_snap, TARGET_CLASS_IDS)
                    if found_snap and tracker.last_box is not None:
                        best_box = tracker.last_box
                        if hasattr(tracker, 'last_mask_polygon') and tracker.last_mask_polygon is not None:
                            h_s, w_s = color_aligned.shape[:2]
                            m = np.zeros((h_s, w_s), dtype=np.uint8)
                            cv2.fillPoly(m, [tracker.last_mask_polygon.astype(np.int32)], 255)
                            n_px = int(np.sum(m == 255))
                            MAX_MASK_PIXELS = int(h_s * w_s * 0.15)
                            if n_px <= MAX_MASK_PIXELS:
                                seg_mask = m
                time.sleep(0.02)

            if color_aligned is None:
                color_aligned, _, _ = cap.read()
            h_a, w_a = color_aligned.shape[:2]

            x1_a, y1_a, x2_a, y2_a = best_box
            x1_a, y1_a = max(0, x1_a), max(0, y1_a)
            x2_a, y2_a = min(w_a, x2_a), min(h_a, y2_a)

            # Locked geometric center of the tracked object (inviolable, immune to edge jitter)
            obj_px = int((x1_a + x2_a) / 2)
            obj_py = int((y1_a + y2_a) / 2)
            print(f"   🔒 Target Track Locked in Memory: box=({x1_a},{y1_a},{x2_a},{y2_a}) stable_center=({obj_px},{obj_py})")

            # Size up object to determine dimensions & orientation
            depth_raw, d_scale = cap.get_raw_depth() if hasattr(cap, "get_raw_depth") else (None, 0.001)
            sizing = size_up_object(color_aligned, seg_mask, (x1_a, y1_a, x2_a, y2_a), depth_mm=280.0, target_desc=TARGET_DESC, depth_frame=depth_raw, depth_scale=d_scale)
            if sizing is not None and "opt_px" in sizing:
                obj_px, obj_py = sizing["opt_px"]
            fit_status = "✅ Fits gripper" if sizing["fits_gripper"] else "⚠️ Body too wide — targeting narrow section"
            print(f"   📏 Object Sized Up: Body={sizing['body_width_mm']}mm | Grasp={sizing['grasp_width_mm']}mm | "
                  f"Est Height={sizing['height_mm']}mm ({sizing['height_pct']:.0f}% height) | {fit_status}")
            print(f"   📐 Orientation: Tilt={sizing['tilt_deg']:+.1f}° | Target Roll={sizing['recommended_roll_deg']:+.1f}° (Δroll={sizing['delta_roll_deg']:+.1f}°) | Recommended Pitch={sizing['recommended_pitch_deg']:+.1f}°")
            print(f"   🎯 Slimmer Section Grasp Point: ({obj_px},{obj_py})")

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
                xyz = cap.get_xyz_from_mask(seg_mask, target_px=(obj_px, obj_py))
                if xyz is not None and xyz[2] <= MAX_GRAB_DEPTH_MM:
                    print(f"   🎭 Mask depth: x={xyz[0]:+.0f}mm  y={xyz[1]:+.0f}mm  z={xyz[2]:.0f}mm  "
                          f"(from {np.sum(seg_mask==255)} mask pixels)")
                else:
                    xyz = None

            if xyz is None:
                # Centered depth samples along the target grasp point and object body
                xyz_samples  = []
                c_box_x = (x1_a + x2_a) // 2
                c_box_y = (y1_a + y2_a) // 2
                sample_pts = [
                    (obj_px, obj_py),
                    (c_box_x, c_box_y),
                    ((obj_px + c_box_x) // 2, (obj_py + c_box_y) // 2),
                ]
                if y2_a > y1_a:
                    h_b = y2_a - y1_a
                    sample_pts.append((obj_px, min(h_a - 1, y1_a + int(h_b * 0.25))))
                    sample_pts.append((obj_px, min(h_a - 1, y1_a + int(h_b * 0.50))))

                for pt in sample_pts:
                    for _ in range(2):
                        s = cap.get_xyz(pt[0], pt[1], search_w=14, search_h=14)
                        if s is not None and s[2] > 70.0 and s[2] <= MAX_GRAB_DEPTH_MM:
                            xyz_samples.append(s)
                        time.sleep(0.015)
                if not xyz_samples:
                    for _ in range(3):
                        s = cap.get_xyz(obj_px, obj_py, search_w=18, search_h=18)
                        if s is not None and s[2] > 70.0 and s[2] <= MAX_GRAB_DEPTH_MM:
                            xyz_samples.append(s)
                        time.sleep(0.015)
                if not xyz_samples:
                    print(f"⚠️  No valid object depth (depth > {MAX_GRAB_DEPTH_MM:.0f}mm or in blind zone) — restarting search")
                    smooth_move(robot, START_POS, step_size=2.0, step_delay=0.03)
                    STATE = "SEARCHING"
                    continue
                zs = [s[2] for s in xyz_samples]
                med_z = float(np.median(zs))
                # Deproject the true CAP pixel (obj_px, obj_py) using median depth
                intr = cap._intrinsics
                if intr is not None:
                    pt_3d = rs.rs2_deproject_pixel_to_point(intr, [float(obj_px), float(obj_py)], med_z / 1000.0)
                    xyz = (pt_3d[0] * 1000.0, pt_3d[1] * 1000.0, med_z)
                else:
                    xyz = (float(np.median([s[0] for s in xyz_samples])),
                           float(np.median([s[1] for s in xyz_samples])),
                           med_z)
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

            # Refresh sizing with exact live depth measurement
            if sizing is not None and xyz is not None:
                depth_raw, d_scale = cap.get_raw_depth() if hasattr(cap, "get_raw_depth") else (None, 0.001)
                sizing = size_up_object(color_aligned, seg_mask, (x1_a, y1_a, x2_a, y2_a), depth_mm=xyz[2], target_desc=TARGET_DESC, depth_frame=depth_raw, depth_scale=d_scale)

            # ── STEP 2b: 3D Orientation Analysis (Disambiguate Lying Flat vs Upright) ──
            orient_3d = analyze_3d_orientation(cap, robot, (x1_a, y1_a, x2_a, y2_a), seg_mask, sizing)

            # ── Convert to arm base frame & Solve IK with Adaptive Penetration ──
            # Gripper throat is ~37mm deep. GRASP_PENETRATION_MM (20.0mm) centers the object
            # securely between the rubber gripper pads (~17mm clearance from rear servo face).
            # If 20mm is slightly out of reach, fall back to 16mm, 12mm, or 8mm.
            if orient_3d.get("is_lying_flat") or orient_3d.get("is_3d_diagonal"):
                # Table-resting objects (top-down or steep approach):
                # Pull back along the reach line so the gripper pads land squarely
                # over the resting bottle neck without overshooting past it into empty air.
                # Base pullback of 15.0mm aligns the rubber pads (which sit behind the fingertips) with the bottle.
                # Farther bottles (raw_dist > 260mm) receive proportional extra pullback to keep reach inside
                # the ~255-260mm physical sweet spot where steep (-85°) top-down grasp is reachable.
                mid_base = orient_3d.get("midpoint_base")
                raw_dist = math.hypot(mid_base[0], mid_base[1]) if mid_base is not None else 250.0

                base_pullback = 15.0
                excess_dist = max(0.0, raw_dist - 260.0)
                needed_pullback = base_pullback + excess_dist
                scaled_pullback = float(np.clip(needed_pullback, 15.0, 50.0))

                flat_rad_offset = -scaled_pullback
                penetration_candidates = [
                    flat_rad_offset,           # Optimal pullback into sweet spot
                    flat_rad_offset + 5.0,     # fallback
                    flat_rad_offset + 10.0,    # fallback
                    flat_rad_offset + 15.0,    # fallback
                    0.0,                       # neutral zero fallback
                ]
            else:
                # Upright bottle: level horizontal approach with full pad penetration
                penetration_candidates = [
                    GRASP_PENETRATION_MM,         # 20.0mm: ~17mm clearance from rear servo face
                    GRASP_PENETRATION_MM - 4.0,   # 16.0mm: ~21mm clearance
                    GRASP_PENETRATION_MM - 8.0,   # 12.0mm: ~25mm clearance
                    8.0,                          # 8.0mm: firm pad grip
                ]
            grab_pos = None
            final_target = None
            current_j = get_pos(robot)
            target_pitch = PREFERRED_GRAB_PITCH_DEG  # Preferred parallel-to-ground pitch (0.0° = horizontal)
            target_roll = START_POS.get("wrist_roll.pos", 29.27)
            delta_roll = 0.0

            if sizing is not None:
                target_pitch = sizing.get("recommended_pitch_deg", PREFERRED_GRAB_PITCH_DEG)
                target_roll  = sizing.get("recommended_roll_deg", target_roll)
                delta_roll   = sizing.get("delta_roll_deg", 0.0)

            # 3D Orientation override (detects bottles lying flat pointing towards arm)
            if orient_3d.get("confidence") == "high_3d_span":
                target_pitch = orient_3d["target_pitch_deg"]
                target_roll  = orient_3d["target_roll_deg"]
                delta_roll   = orient_3d["delta_roll_deg"]
                if orient_3d["is_lying_flat"] and orient_3d.get("midpoint_cam") is not None:
                    xyz = orient_3d["midpoint_cam"]

            for pen_mm in penetration_candidates:
                if orient_3d.get("confidence") == "high_3d_span" and orient_3d.get("midpoint_base") is not None:
                    mid_x, mid_y, mid_z = orient_3d["midpoint_base"]
                    arm_x = mid_x
                    arm_y = mid_y
                    live_tbl_z = orient_3d.get("live_table_z") if orient_3d.get("confidence") == "high_3d_span" else None
                    table_floor_z = live_tbl_z if live_tbl_z is not None else TABLE_FLOOR_Z_MM
                    table_floor_z = float(np.clip(table_floor_z, -250.0, -80.0))

                    if orient_3d["is_lying_flat"] or orient_3d.get("is_3d_diagonal"):
                        # Lying flat or 3D diagonal: penetrate horizontally along reach line (pan)
                        # NEVER subtract Z, so the gripper does not plunge into the table surface!
                        pan_rad = math.atan2(arm_y, arm_x)
                        arm_x = mid_x + pen_mm * math.cos(pan_rad)
                        arm_y = mid_y + pen_mm * math.sin(pan_rad)
                        arm_z = mid_z
                    else:
                        # Upright bottle: level horizontal approach directly into neck cylinder
                        pan_rad = math.atan2(mid_y, mid_x)
                        arm_x = mid_x + pen_mm * math.cos(pan_rad)
                        arm_y = mid_y + pen_mm * math.sin(pan_rad)
                        arm_z = mid_z
                else:
                    target = depth_to_arm_target(xyz, robot, penetration_mm=pen_mm)
                    if target is None:
                        continue
                    arm_x, arm_y, arm_z = target

                # ── Apply true lateral claw offset (strictly perpendicular to reach line) ──
                # Shifts claw strictly to the robot's LEFT (-sin(pan), +cos(pan)) away from the static right pincer.
                # Strictly preserves radial reach distance (rho) so the arm NEVER overshoots or undershoots!
                tilt_val = sizing.get("tilt_deg", 0.0) if sizing is not None else 0.0
                if orient_3d.get("is_upright"):
                    lat_offset = GRAB_LATERAL_OFFSET_STANDING_MM  # ~3.0 inches (76.2mm)
                elif tilt_val <= -20.0:
                    lat_offset = GRAB_LATERAL_OFFSET_RIGHT_MM     # ~3.0 inches (76.2mm)
                else:
                    lat_offset = GRAB_LATERAL_OFFSET_MM           # 48.0mm (verified perfect)

                if abs(lat_offset) > 0.01:
                    pan_t = math.atan2(arm_y, arm_x)
                    arm_x += -lat_offset * math.sin(pan_t)
                    arm_y +=  lat_offset * math.cos(pan_t)
                    print(f"   📐 Applied lateral grasp offset: {lat_offset:+.1f}mm (LEFT claw clearance shift perpendicular to reach)")

                # ── Dynamic Tabletop Safeguard with Jaw Droop Protection ───────────
                # When wrist is pitched and rolled, lower jaw tip hangs lower than wrist/grasp centerline.
                # Droop = 45mm * |sin(delta_roll)| * cos(pitch).
                # Table surface protection guarantees fingertip stays >= 12mm above physical tabletop surface!
                live_tbl_z = orient_3d.get("live_table_z") if orient_3d.get("confidence") == "high_3d_span" else None
                table_floor_z = live_tbl_z if live_tbl_z is not None else TABLE_FLOOR_Z_MM
                table_floor_z = float(np.clip(table_floor_z, -250.0, -80.0))
                droll_actual = target_roll - (START_POS.get("wrist_roll.pos", 29.27) if "START_POS" in globals() else _BASE.get("wrist_roll.pos", 29.27))
                jaw_droop_mm = 45.0 * abs(math.sin(math.radians(droll_actual))) * math.cos(math.radians(target_pitch))
                min_safe_z = table_floor_z + jaw_droop_mm + 12.0
                if arm_z < min_safe_z:
                    print(f"   🛡️ Table Floor Guard: arm_z was {arm_z:+.1f}mm (< {min_safe_z:+.1f}mm with {jaw_droop_mm:.1f}mm jaw droop) → elevating to {min_safe_z:+.1f}mm")
                    arm_z = min_safe_z

                if not workspace_in_bounds(arm_x, arm_y, arm_z) and not do_manual_lunge:
                    continue

                sol = solve_ik(arm_x, arm_y, arm_z,
                               end_pitch_deg=target_pitch,
                               current_joints=current_j,
                               wrist_roll_deg=target_roll)
                if sol is not None:
                    grab_pos = sol
                    final_target = (arm_x, arm_y, arm_z, pen_mm)
                    break

            if grab_pos is None:
                if do_manual_lunge:
                    print("⚠️  Analytical IK found no rigid solution at this pose, but proceeding to Manual Lunge Demonstration...")
                else:
                    print("⚠️  IK: target outside reachable joint configuration at all penetration depths — restarting search")
                    smooth_move(robot, START_POS, step_size=2.0, step_delay=0.03)
                    time.sleep(2.0)
                    STATE = "SEARCHING"
                    continue

            if final_target is not None:
                arm_x, arm_y, arm_z, pen_used = final_target
                rho_t = math.sqrt(arm_x**2 + arm_y**2)
                if pen_used < 0:
                    grasp_type = f"PULLBACK GRASP ({abs(pen_used):.0f}mm pullback along reach line)"
                else:
                    clearance_val = 37.0 - pen_used
                    grasp_type = f"TIP GRASP (~{clearance_val:.0f}mm servo clearance)"
                print(f"   🎯 Grasp Target: X={arm_x:+.0f}mm, Y={arm_y:+.0f}mm, Z={arm_z:+.0f}mm | Dist={rho_t:.0f}mm")
                print(f"   📐 Penetration: {pen_used:+.0f}mm ({grasp_type}) | Approach Pitch={target_pitch:.1f}° | Wrist Roll={target_roll:.1f}°")

            if grab_pos is not None:
                print(f"\n📐 IK SOLUTION:")
                print(f"   Pan   → {grab_pos['shoulder_pan.pos']:+.1f}°")
                print(f"   Lift  → {grab_pos['shoulder_lift.pos']:+.1f}°")
                print(f"   Elbow → {grab_pos['elbow_flex.pos']:+.1f}°")
                print(f"   Wrist → {grab_pos['wrist_flex.pos']:+.1f}°")
                print(f"   Roll  → {grab_pos.get('wrist_roll.pos', target_roll):+.1f}°")



            if do_manual_lunge:
                # ── Manual Lunge Demonstration ────────────────────────────────────
                print("\n" + "═"*65)
                print(" 🛠️  MANUAL LUNGE DEMONSTRATION MODE")
                print("═"*65)
                print(" The robot has finished visual alignment and calculated the target!")
                print(f" Target coordinates: X={arm_x:+.0f}mm, Y={arm_y:+.0f}mm, Z={arm_z:+.0f}mm")
                print("\n Disabling motor torque... You can now move the arm manually by hand.")
                print(f" Mimic the perfect lunge from this current position to grab the {TARGET_DESC}.")
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
                print("\n🦾 APPROACHING (level gripper, jaws wide open)...")
                grab_pos["gripper.pos"] = max(75.0, START_POS.get("gripper.pos", 72.6))
                level_approach(robot, grab_pos,
                               step_size=3.0, step_delay=0.03)
                time.sleep(0.4)

    
                print(f"✊ GRIPPING (closing jaws firmly to {GRIPPER_CLOSED_POS_DEG:.1f}°)...")
                grab_pos["gripper.pos"] = GRIPPER_CLOSED_POS_DEG
                smooth_move(robot, grab_pos, step_size=3.0, step_delay=0.02, hold_joints=[])
                time.sleep(0.6)

                print("🏠 RETURNING...")
                drop_pos = dict(START_POS)
                drop_pos["gripper.pos"] = GRIPPER_CLOSED_POS_DEG
                # Maintain the grasp wrist roll so the wrist never twists while carrying the bottle:
                drop_pos["wrist_roll.pos"] = grab_pos.get("wrist_roll.pos", target_roll)
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
        print("\n⏹️  Interrupted by user — stowing arm smoothly...")
        if not _arm_is_stowed[0]:
            try:
                get_pos(robot)           # probe — raises if arm is dead/power-lost
                smooth_move(robot, STOW, step_size=0.8, step_delay=0.025)
                print("   ✅ Arm stowed smoothly.")
                _arm_is_stowed[0] = True
            except Exception as e:
                print(f"   ⚠️  Stow skipped (arm unreachable): {e}")
        try:
            robot.disconnect()
        except Exception:
            pass
    except Exception as exc:
        print(f"\n❌  Unhandled exception: {exc}")
        import traceback; traceback.print_exc()
        if not _arm_is_stowed[0]:
            try:
                smooth_move(robot, STOW, step_size=0.8, step_delay=0.025)
                _arm_is_stowed[0] = True
            except Exception:
                pass
        try:
            robot.disconnect()
        except Exception:
            pass
    finally:
        try:
            atexit.unregister(_emergency_stow)
        except Exception:
            pass
        try:
            cap.stop()
        except Exception:
            pass
        try:
            _destroy_windows()
        except Exception:
            pass

        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        # Immediate clean exit prevents C++ shared library static destructor collision
        # between librealsense2 and PyTorch CUDA runtime on Jetson glibc
        os._exit(0)




if __name__ == "__main__":
    main()
#v3 — RealSense D405 + Analytical IK