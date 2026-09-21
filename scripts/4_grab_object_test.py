#!/usr/bin/env python3
"""
4_grab_object_test.py — Safe, Clean Object Grab Test (SO-ARM101 + RealSense D405)
================================================================================
State machine: SEARCHING -> ALIGNING -> DEPTH_READ -> IK_LUNGE -> GRAB -> RETURN

Key Safety Guarantees:
  1. NO artificial JOINT_LIMITS_DEG clamping (prevents servo stall trips).
  2. NO multi-stage folding startup moves (prevents cantilever torque overloads).
  3. Direct linear interpolation smooth_move (step_size=2.0, step_delay=0.02).
  4. Active control of 5 primary joints only; wrist_roll is held neutral (no twisting).
  5. Correct positive pan convention and verified forward kinematics.

Usage:
  python3 scripts/4_grab_object_test.py
"""

import os
import sys
import time
import signal
import math
import yaml
import numpy as np
import cv2
import pyrealsense2 as rs

# ── Ensure Hugging Face and CLIP models persist to host-mounted SSD storage ────
_hf_dir = os.environ.get("HF_HOME") or "/root/ros2_ws/models/huggingface"
if not os.path.exists(_hf_dir):
    _local_hf = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "huggingface"))
    if os.path.isdir(os.path.dirname(_local_hf)):
        _hf_dir = _local_hf
try:
    os.makedirs(_hf_dir, exist_ok=True)
except Exception:
    pass
os.environ["HF_HOME"] = _hf_dir
os.environ["TRANSFORMERS_CACHE"] = _hf_dir
os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(_hf_dir), "torch")

from ultralytics import YOLO

# ── Dynamic LeRobot Imports ───────────────────────────────────────────────────
SOFollower = None
SOFollowerRobotConfig = None

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
            pass

PORT   = os.environ.get("ARM_PORT", "/dev/arm_controller")
ARM_ID = os.environ.get("ARM_ID", "jetson_arm")

# ── Detection ─────────────────────────────────────────────────────────────────
TARGET_DESC   = "bottle"
YOLO_CLASS_ID = 0

# ── Arm Geometry Constants (SO-ARM101) ────────────────────────────────────────
IK_L1 = 115.0   # shoulder pivot -> elbow pivot  (mm)
IK_L2 = 137.5   # elbow pivot    -> wrist pivot   (mm)
IK_L3 = 90.0    # wrist pivot    -> gripper tip   (mm)

PAN_ZERO_OFFSET_DEG = -4.6
PAN_MIN_DEG         = -113.8
PAN_MAX_DEG         =  113.8

# Empirical workspace limits in base frame (origin = shoulder pivot)
WS_X_MIN_MM   = -140.0
WS_X_MAX_MM   =  285.0
WS_Y_MIN_MM   = -235.0
WS_Y_MAX_MM   =  235.0
WS_Z_MIN_MM   = -225.0
WS_Z_MAX_MM   =  275.0
WS_RHO_MAX_MM =  285.0

# ── Camera Offsets ────────────────────────────────────────────────────────────
CAM_X_OFFSET_MM      = 0.0
CAM_Y_OFFSET_MM      = 50.0
CAM_Z_OFFSET_MM      = 0.0
CAM_PITCH_DEG        = 45.0
D405_MIN_RANGE_MM    = 70.0
MAX_GRAB_DEPTH_MM    = 600.0
GRASP_PENETRATION_MM = 20.0

# ── Verified Starting & Stow Postures ────────────────────────────────────────
_BASE = {
    "shoulder_pan.pos":   -1.4,
    "shoulder_lift.pos": -57.6,
    "elbow_flex.pos":     -3.3,
    "wrist_flex.pos":     86.0,
    "gripper.pos":        60.0,
}

_STOW_BASE = {
    "shoulder_pan.pos":   -1.6,
    "shoulder_lift.pos": -104.5,
    "elbow_flex.pos":     96.5,
    "wrist_flex.pos":      0.0,
    "gripper.pos":        60.0,
}

# ── Visual Servoing Constants ─────────────────────────────────────────────────
ALIGN_THRESHOLD  = 35
ALIGN_PAN_OFFSET = 35
ALIGN_PAN_K      = 0.04
ALIGN_LIFT_K     = 0.05
ALIGN_MAX_PAN    = 3.0
ALIGN_MAX_LIFT   = 1.5

GUI_AVAILABLE = True

def safe_imshow(winname: str, mat: np.ndarray, wait_ms: int = 1):
    global GUI_AVAILABLE
    if not GUI_AVAILABLE:
        return
    try:
        cv2.imshow(winname, mat)
        cv2.waitKey(wait_ms)
    except Exception:
        GUI_AVAILABLE = False


def workspace_in_bounds(x_mm: float, y_mm: float, z_mm: float) -> bool:
    rho = math.sqrt(x_mm**2 + y_mm**2)
    if x_mm < WS_X_MIN_MM or x_mm > WS_X_MAX_MM: return False
    if y_mm < WS_Y_MIN_MM or y_mm > WS_Y_MAX_MM: return False
    if z_mm < WS_Z_MIN_MM or z_mm > WS_Z_MAX_MM: return False
    if rho > WS_RHO_MAX_MM: return False
    return True


def build_default_T_cam_wrist() -> np.ndarray:
    alpha = math.radians(CAM_PITCH_DEG)
    sin_a, cos_a = math.sin(alpha), math.cos(alpha)
    R = np.array([
        [ 0.0, -sin_a,  cos_a],
        [ 0.0,  cos_a,  sin_a],
        [-1.0,   0.0,    0.0 ],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [CAM_Z_OFFSET_MM, -CAM_Y_OFFSET_MM, CAM_X_OFFSET_MM]
    return T


def load_calibrated_T_cam_wrist() -> np.ndarray:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "calibration", "hand_eye_calibration.yaml"),
        os.path.join(script_dir, "calibration", "hand_eye_calibration.yaml"),
        "/root/ros2_ws/calibration/hand_eye_calibration.yaml",
        "calibration/hand_eye_calibration.yaml",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    data = yaml.safe_load(f)
                R = np.array(data["rotation_matrix"], dtype=float)
                t = np.array(data["translation_mm"], dtype=float).flatten()
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t
                print(f"✅ Loaded calibrated T_cam_wrist from: {p}")
                print(f"   Translation (mm): X={t[0]:.1f}, Y={t[1]:.1f}, Z={t[2]:.1f}")
                return T
            except Exception as e:
                print(f"⚠️ Could not parse {p}: {e}")
    print("ℹ️ Using default geometric T_cam_wrist matrix.")
    return build_default_T_cam_wrist()


T_CAM_WRIST = build_default_T_cam_wrist()


# ═══════════════════════════════════════════════════════════════════════════════
# Arm Helpers & Verified Movement Functions
# ═══════════════════════════════════════════════════════════════════════════════
def get_pos(robot) -> dict:
    try:
        obs = robot.get_observation()
        joints = {"shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                  "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"}
        return {k: float(v) for k, v in obs.items() if k in joints}
    except Exception:
        return {}


def smooth_move(robot, target: dict, step_size=2.0, step_delay=0.02, hold_joints=None):
    """Proven linear smooth trajectory with direct timing."""
    if hold_joints is None:
        hold_joints = []
    cur = get_pos(robot)
    if not cur:
        return
    for j in hold_joints:
        if j in target:
            cur[j] = target[j]
    max_delta = max(abs(target[j] - cur.get(j, 0.0)) for j in target)
    if max_delta < 0.5:
        return
    n = max(1, int(max_delta / step_size))
    for s in range(1, n + 1):
        t = s / n
        interp = {j: cur.get(j, 0.0) + t * (target[j] - cur.get(j, 0.0)) for j in target}
        robot.send_action(interp)
        time.sleep(step_delay)


def level_approach(robot, target: dict, step_size=2.0, step_delay=0.03):
    """Moves toward target while keeping gripper parallel to table/floor."""
    cur = get_pos(robot)
    if not cur:
        return
    max_delta = max(abs(target[j] - cur.get(j, 0.0)) for j in target)
    if max_delta < 0.5:
        return
    n = max(1, int(max_delta / step_size))

    for s in range(1, n + 1):
        t = s / n
        interp = {j: cur.get(j, 0.0) + t * (target[j] - cur.get(j, 0.0)) for j in target}
        lift_now = interp.get("shoulder_lift.pos", 0.0)
        elb_now  = interp.get("elbow_flex.pos", 0.0)
        t1_rad = math.radians(90.0 - lift_now)
        t2_rad = t1_rad - math.radians(elb_now + 81.0)
        level_wrist = math.degrees(t2_rad) - 5.0
        final_wrist = target.get("wrist_flex.pos", level_wrist)
        interp["wrist_flex.pos"] = level_wrist + t * (final_wrist - level_wrist)
        robot.send_action(interp)
        time.sleep(step_delay)


def set_torque(robot, enable: bool) -> bool:
    if robot is None or not hasattr(robot, "bus"):
        return False
    val = 1 if enable else 0
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll", "gripper"]
    try:
        if enable and hasattr(robot.bus, "enable_torque"):
            robot.bus.enable_torque()
            return True
        elif not enable and hasattr(robot.bus, "disable_torque"):
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
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Verified Kinematics (FK & IK)
# ═══════════════════════════════════════════════════════════════════════════════
def forward_kinematics(q: dict) -> np.ndarray:
    pan  = math.radians(q.get("shoulder_pan.pos", 0.0) - PAN_ZERO_OFFSET_DEG)
    lift = q.get("shoulder_lift.pos", 0.0)
    elb  = q.get("elbow_flex.pos",    0.0)
    wst  = q.get("wrist_flex.pos",    0.0)

    t1 = math.radians(90.0 - lift)
    t2 = t1 - math.radians(elb + 81.0)
    t3 = t2 - math.radians(wst + 5.0)

    rho_w = IK_L1 * math.cos(t1) + IK_L2 * math.cos(t2)
    z_w   = IK_L1 * math.sin(t1) + IK_L2 * math.sin(t2)

    wx = rho_w * math.cos(pan)
    wy = rho_w * math.sin(pan)
    wz = z_w

    ax = math.cos(t3) * math.cos(pan)
    ay = math.cos(t3) * math.sin(pan)
    az = math.sin(t3)

    zx = -math.sin(pan)
    zy =  math.cos(pan)
    zz = 0.0

    yx = zy * az - zz * ay
    yy = zz * ax - zx * az
    yz = zx * ay - zy * ax

    return np.array([
        [ax, yx, zx, wx],
        [ay, yy, zy, wy],
        [az, yz, zz, wz],
        [0., 0., 0., 1.],
    ])


def solve_ik(x_mm: float, y_mm: float, z_mm: float,
             end_pitch_deg: float = -5.0,
             current_joints: dict | None = None) -> dict | None:
    """Computes joint angles for the 5 active joints (pan, lift, elbow, wrist_flex, gripper)."""
    pan_rad = math.atan2(y_mm, x_mm)
    pan_deg = math.degrees(pan_rad) + PAN_ZERO_OFFSET_DEG

    if pan_deg < PAN_MIN_DEG or pan_deg > PAN_MAX_DEG:
        return None

    rho = math.sqrt(x_mm**2 + y_mm**2)
    pitch_candidates = [end_pitch_deg]
    p = end_pitch_deg - 5.0
    while p >= -85.0:
        pitch_candidates.append(p)
        p -= 5.0

    best_solution = None
    best_cost = float('inf')

    for test_pitch in pitch_candidates:
        pitch_rad = math.radians(test_pitch)
        wrist_x = rho  - IK_L3 * math.cos(pitch_rad)
        wrist_z = z_mm - IK_L3 * math.sin(pitch_rad)

        D = math.sqrt(wrist_x**2 + wrist_z**2)
        D_max = IK_L1 + IK_L2 - 1.0
        D_min = abs(IK_L1 - IK_L2) + 1.0
        if D > D_max or D < D_min:
            continue

        cos_t2 = (D**2 - IK_L1**2 - IK_L2**2) / (2.0 * IK_L1 * IK_L2)
        cos_t2 = float(np.clip(cos_t2, -1.0, 1.0))
        alpha = math.atan2(wrist_z, wrist_x)

        for sign in (-1.0, +1.0):
            t2 = sign * math.acos(cos_t2)
            beta = math.atan2(IK_L2 * math.sin(t2), IK_L1 + IK_L2 * math.cos(t2))
            t1 = alpha - beta

            m_lift  = 90.0 - math.degrees(t1)
            m_elbow = -math.degrees(t2) - 81.0
            m_wrist = math.degrees(t1 + t2 - pitch_rad) - 5.0

            if not (-110 <= m_lift <= 150): continue
            if not (-120 <= m_elbow <= 120): continue
            if not (-120 <= m_wrist <= 120): continue
            if m_lift < -90.0: continue

            sol = {
                "shoulder_pan.pos":  pan_deg,
                "shoulder_lift.pos": m_lift,
                "elbow_flex.pos":    m_elbow,
                "wrist_flex.pos":    m_wrist,
                "gripper.pos":       60.0,
            }

            if current_joints is None:
                return sol

            cost = (3.0 * abs(m_lift - current_joints.get("shoulder_lift.pos", 0)) +
                    2.0 * abs(m_elbow - current_joints.get("elbow_flex.pos", 0)) +
                    1.0 * abs(m_wrist - current_joints.get("wrist_flex.pos", 0)))
            if cost < best_cost:
                best_cost = cost
                best_solution = sol

        if best_solution is not None:
            return best_solution

    return None


def depth_to_arm_target(xyz_cam: tuple[float, float, float], robot) -> tuple[float, float, float] | None:
    x_cam, y_cam, z_cam = xyz_cam
    if z_cam < D405_MIN_RANGE_MM + 10.0:
        return None

    approach_z = max(D405_MIN_RANGE_MM, z_cam + GRASP_PENETRATION_MM)
    P_cam = np.array([x_cam, y_cam, approach_z, 1.0])
    P_wrist = T_CAM_WRIST @ P_cam

    cur = get_pos(robot)
    T_wrist_base = forward_kinematics(cur)
    P_base = T_wrist_base @ P_wrist

    return float(P_base[0]), float(P_base[1]), float(P_base[2])


# ═══════════════════════════════════════════════════════════════════════════════
# RealSense D405 Stream
# ═══════════════════════════════════════════════════════════════════════════════
class RealSenseStream:
    def __init__(self, width=848, height=480, fps=15):
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.yuyv, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self._profile = self._pipeline.start(cfg)
        self._depth_scale = self._profile.get_device().first_depth_sensor().get_depth_scale()
        self._intrinsics = self._profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    def read(self):
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                return None, None
            raw = np.asanyarray(color_frame.get_data())
            h, w = color_frame.get_height(), color_frame.get_width()
            yuyv = raw.view(np.uint8).reshape(h, w, 2)
            color = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
            depth_img = np.asanyarray(depth_frame.get_data())
            return color, depth_img
        except Exception:
            return None, None

    def get_xyz(self, depth_img, px: int, py: int, search_w=30, search_h=30):
        if depth_img is None:
            return None
        h, w = depth_img.shape[:2]
        x1, x2 = max(0, px - search_w // 2), min(w, px + search_w // 2 + 1)
        y1, y2 = max(0, py - search_h // 2), min(h, py + search_h // 2 + 1)
        patch = depth_img[y1:y2, x1:x2].astype(float) * self._depth_scale
        valid = patch[(patch > 0.07) & (patch < 2.0)]
        if valid.size == 0:
            return None
        z_m = float(np.median(valid))
        pt = rs.rs2_deproject_pixel_to_point(self._intrinsics, [float(px), float(py)], z_m)
        return pt[0] * 1000.0, pt[1] * 1000.0, pt[2] * 1000.0

    def stop(self):
        try:
            self._pipeline.stop()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Main Program Flow
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("🤖 SO-ARM101 + REALSENSE D405 PICK-AND-PLACE TEST")
    print("=" * 70)

    global T_CAM_WRIST, TARGET_DESC, YOLO_CLASS_ID, GUI_AVAILABLE
    T_CAM_WRIST = load_calibrated_T_cam_wrist()

    if "--headless" in sys.argv or not os.environ.get("DISPLAY"):
        GUI_AVAILABLE = False
        print("🖥️ Running in HEADLESS mode.")

    target_in = input(f"🎯 Target object name to grab [default '{TARGET_DESC}']: ").strip()
    if target_in:
        TARGET_DESC = target_in

    # Resolve YOLO model
    model_paths = [
        "/root/ros2_ws/models/yolov8s-worldv2.pt",
        os.path.join(os.path.dirname(__file__), "..", "models", "yolov8s-worldv2.pt"),
        "models/yolov8s-worldv2.pt",
        "yolov8s-worldv2.pt",
        "yolo11n-seg.pt",
    ]
    model_path = next((p for p in model_paths if os.path.exists(p)), "yolov8s-worldv2.pt")
    print(f"Loading YOLO: {model_path}...")
    model = YOLO(model_path)
    if "world" in model_path.lower() or hasattr(model, "set_classes"):
        model.set_classes([TARGET_DESC])
        YOLO_CLASS_ID = 0
        print(f"   🌍 YOLO-World configured for: ['{TARGET_DESC}'] (ID: 0)")
    else:
        print(f"   🎯 Standard YOLO model loaded.")

    # ── Auto-sync persistent calibration if available ─────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    calib_source_paths = [
        "/root/ros2_ws/calibration/jetson_arm.json",
        os.path.join(script_dir, "..", "calibration", "jetson_arm.json"),
        os.path.join(script_dir, "calibration", "jetson_arm.json"),
        "calibration/jetson_arm.json",
    ]
    target_calib_dirs = [
        "/data/models/huggingface/lerobot/calibration/robots/so101_follower",
        os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots/so101_follower"),
    ]
    for src in calib_source_paths:
        if os.path.exists(src) and os.path.getsize(src) > 0:
            for tdir in target_calib_dirs:
                try:
                    os.makedirs(tdir, exist_ok=True)
                    dst = os.path.join(tdir, f"{ARM_ID}.json")
                    if not os.path.exists(dst) or os.path.getsize(dst) == 0:
                        import shutil
                        shutil.copyfile(src, dst)
                        print(f"   📋 Synced verified calibration: {src} -> {dst}")
                except Exception:
                    pass
            break

    # Connect Robot
    print(f"🔌 Connecting to SO-ARM101 on {PORT}...")
    import builtins
    _orig_input = builtins.input
    builtins.input = lambda prompt="": ""
    try:
        config = SOFollowerRobotConfig(port=PORT, id=ARM_ID, use_degrees=True)
        robot = SOFollower(config)
        robot.connect()
        print("   ✅ Arm connected successfully.")
    except Exception as e:
        print(f"❌ Failed to connect to arm: {e}")
        sys.exit(1)
    finally:
        builtins.input = _orig_input

    START_POS = dict(_BASE)
    STOW = dict(_STOW_BASE)

    def handle_exit(sig=None, frame=None):
        print("\n📍 Stowing arm safely...")
        try:
            smooth_move(robot, STOW, step_size=2.0, step_delay=0.02)
            time.sleep(0.5)
            robot.disconnect()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)

    print("📷 Connecting to RealSense D405...")
    cap = RealSenseStream()
    time.sleep(1.5)

    print("\n▶ Moving to Start Position directly...")
    smooth_move(robot, START_POS, step_size=2.0, step_delay=0.02)
    time.sleep(1.0)

    print("\n" + "=" * 65)
    print(f"🔍 Searching for '{TARGET_DESC}'...")
    print("=" * 65 + "\n")

    sweep_pan = START_POS["shoulder_pan.pos"]
    sweep_dir = 1.0

    try:
        while True:
            color, depth_img = cap.read()
            if color is None or depth_img is None:
                time.sleep(0.02)
                continue

            h, w = color.shape[:2]
            frame_cx, frame_cy = (w // 2) + ALIGN_PAN_OFFSET, h // 2

            results = model(color, verbose=False, conf=0.45)
            target_box = None

            for box in results[0].boxes:
                if int(box.cls[0].item()) == YOLO_CLASS_ID:
                    target_box = box
                    break

            if target_box is None:
                # Sweep search
                sweep_pan += sweep_dir * 0.4
                if sweep_pan >= START_POS["shoulder_pan.pos"] + 50.0:
                    sweep_dir = -1.0
                elif sweep_pan <= START_POS["shoulder_pan.pos"] - 50.0:
                    sweep_dir = 1.0
                cmd = dict(START_POS)
                cmd["shoulder_pan.pos"] = sweep_pan
                robot.send_action(cmd)

                display = color.copy()
                cv2.putText(display, f"SEARCHING: {TARGET_DESC}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                safe_imshow("Arm Vision", display)
                continue

            # Found target
            bx1, by1, bx2, by2 = map(int, target_box.xyxy[0].tolist())
            obj_px = (bx1 + bx2) // 2
            obj_py = (by1 + by2) // 2

            display = color.copy()
            cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.circle(display, (obj_px, obj_py), 5, (0, 0, 255), -1)
            cv2.putText(display, f"LOCKED: {TARGET_DESC}", (bx1, max(20, by1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            safe_imshow("Arm Vision", display)

            pan_err  = obj_px - frame_cx
            lift_err = obj_py - frame_cy

            if abs(pan_err) > ALIGN_THRESHOLD or abs(lift_err) > ALIGN_THRESHOLD:
                # Align step
                p_cmd = float(np.clip(pan_err * ALIGN_PAN_K, -ALIGN_MAX_PAN, ALIGN_MAX_PAN))
                l_cmd = float(np.clip(lift_err * ALIGN_LIFT_K, -ALIGN_MAX_LIFT, ALIGN_MAX_LIFT))
                cur_j = get_pos(robot)
                robot.send_action({
                    "shoulder_pan.pos":  cur_j.get("shoulder_pan.pos", 0.0) + p_cmd,
                    "shoulder_lift.pos": cur_j.get("shoulder_lift.pos", 0.0) + l_cmd,
                    "elbow_flex.pos":    START_POS["elbow_flex.pos"],
                    "wrist_flex.pos":    START_POS["wrist_flex.pos"],
                    "gripper.pos":       START_POS["gripper.pos"],
                })
                time.sleep(0.04)
                continue

            # Centred! Proceed to grab
            print(f"🎯 Target centred! Reading 3D depth...")
            time.sleep(0.2)
            color, depth_img = cap.read()
            xyz = cap.get_xyz(depth_img, obj_px, obj_py)

            if xyz is None or xyz[2] > MAX_GRAB_DEPTH_MM:
                print(f"⚠️ Invalid depth ({xyz[2] if xyz else 'None'} mm) — skipping.")
                continue

            print(f"   Camera coords: X={xyz[0]:+.0f}mm, Y={xyz[1]:+.0f}mm, Depth={xyz[2]:.0f}mm")
            target_base = depth_to_arm_target(xyz, robot)

            if target_base is None or not workspace_in_bounds(*target_base):
                print(f"⚠️ Target out of reachable workspace envelope — skipping.")
                continue

            arm_x, arm_y, arm_z = target_base
            print(f"   Base coords: X={arm_x:+.0f}mm, Y={arm_y:+.0f}mm, Z={arm_z:+.0f}mm")

            cur_j = get_pos(robot)
            grab_pos = solve_ik(arm_x, arm_y, arm_z, end_pitch_deg=-5.0, current_joints=cur_j)
            if grab_pos is None:
                print("⚠️ IK unreachable for this position — skipping.")
                continue

            print(f"\n🦾 EXECUTING LUNGE & GRAB:")
            print(f"   Pan:  {grab_pos['shoulder_pan.pos']:+.1f}°")
            print(f"   Lift: {grab_pos['shoulder_lift.pos']:+.1f}°")
            print(f"   Elb:  {grab_pos['elbow_flex.pos']:+.1f}°")
            print(f"   Wst:  {grab_pos['wrist_flex.pos']:+.1f}°")

            # 1. Level approach
            level_approach(robot, grab_pos, step_size=2.0, step_delay=0.03)
            time.sleep(0.3)

            # 2. Close gripper
            print("✊ Closing gripper...")
            grab_pos["gripper.pos"] = 0.7
            robot.send_action(grab_pos)
            time.sleep(1.0)

            # 3. Lift and return
            print("🏠 Returning to start posture...")
            return_pos = dict(START_POS)
            return_pos["gripper.pos"] = 0.7
            smooth_move(robot, return_pos, step_size=2.0, step_delay=0.02, hold_joints=["gripper.pos"])
            time.sleep(1.0)

            # 4. Release
            print("🖐 Releasing object...")
            return_pos["gripper.pos"] = 60.0
            smooth_move(robot, return_pos, step_size=2.0, step_delay=0.02)
            time.sleep(1.0)

            print("✅ Pick-and-place cycle complete! Resuming search...\n")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user.")
    finally:
        cap.stop()
        cv2.destroyAllWindows()
        handle_exit()


if __name__ == "__main__":
    main()
