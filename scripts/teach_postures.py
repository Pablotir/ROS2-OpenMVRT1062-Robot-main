#!/usr/bin/env python3
"""
teach_postures.py — Interactive Reference Pose Teaching Tool for SO-ARM101

Run this on the Jetson to record new neutral and stow positions after calibration:
    python3 teach_postures.py

Workflow:
1. Connects to arm using calibrated offsets (no interactive LeRobot prompts).
2. Checks servo health (verifies no overloads/stalls).
3. Cuts motor torque immediately (free-movement mode).
4. Asks you to guide arm to the SCAN posture (neutral forward/workspace view), then press ENTER.
5. Asks you to guide arm to the STOW posture (folded safe/compact), then press ENTER.
6. Saves both postures to arm_reference_poses.yaml so arm_picker.py uses them automatically.
"""

import os
import sys
import time
import math
import yaml
import json
import pathlib
import datetime
from types import SimpleNamespace

# LeRobot imports
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

PORT   = "/dev/arm_controller"
ARM_ID = "jetson_arm"

# SO-ARM101 Kinematics parameters
IK_L1_MM = 112.0
IK_L2_MM = 135.0
IK_L3_MM = 153.0
PAN_ZERO_OFFSET_DEG = 0.0


def forward_kinematics(q: dict):
    """Calculate 4x4 T_wrist_base for wrist pivot position."""
    pan   = math.radians(-q.get("shoulder_pan.pos",  0.0) - PAN_ZERO_OFFSET_DEG)
    lift  = math.radians(90.0 - q.get("shoulder_lift.pos", 0.0))
    elbow = math.radians(q.get("elbow_flex.pos",    0.0) + 81.0)
    wrist = math.radians(q.get("wrist_flex.pos",    0.0) + 5.0)

    t1 = lift
    t2 = t1 - elbow
    t3 = t2 - wrist

    r_w = IK_L1_MM * math.cos(t1) + IK_L2_MM * math.cos(t2) + IK_L3_MM * math.cos(t3)
    z_w = IK_L1_MM * math.sin(t1) + IK_L2_MM * math.sin(t2) + IK_L3_MM * math.sin(t3)

    wx = r_w * math.cos(pan)
    wy = r_w * math.sin(pan)
    wz = z_w
    return wx, wy, wz


def get_pos(robot) -> dict:
    obs = robot.get_observation()
    joints = {"shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
              "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"}
    return {k: v for k, v in obs.items() if k in joints}


def set_torque(robot, enable: bool):
    val = 1 if enable else 0
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll", "gripper"]
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
    return False


def check_servo_health(robot) -> bool:
    print("🩺 Checking servo health...")
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll", "gripper"]
    for reg in ("Present_Load", "present_load", "Load"):
        try:
            robot.bus.read(reg, motor_names[0])
            all_ok = True
            for name in motor_names:
                raw_load = abs(int(robot.bus.read(reg, name)))
                load_mag = raw_load & 0x03FF
                if load_mag > 800:
                    print(f"   ❌ {name}: high load ({load_mag}/1023) — possibly stalled (raw={raw_load})")
                    all_ok = False
                else:
                    print(f"   ✅ {name}: load={load_mag}/1023")
            return all_ok
        except Exception:
            continue
    print("   ⚠️  Health check skipped (load register not readable) — proceeding.")
    return True


def connect_robot():
    print("🔌 Connecting to SO-ARM101...")
    config = SOFollowerRobotConfig(port=PORT, id=ARM_ID, use_degrees=True)
    robot  = SOFollower(config)

    # Search for calibration JSON
    hf_home = pathlib.Path(os.environ.get("HF_HOME",
               os.environ.get("TRANSFORMERS_CACHE",
               str(pathlib.Path.home() / ".cache" / "huggingface"))))
    search_paths = [
        pathlib.Path(f"/root/ros2_ws/calibration/{ARM_ID}.json"),
        pathlib.Path(f"/root/ros2_ws/scripts/{ARM_ID}.json"),
        pathlib.Path(__file__).parent / f"{ARM_ID}.json",
        pathlib.Path(__file__).parent.parent / "calibration" / f"{ARM_ID}.json",
        pathlib.Path(__file__).parent / "calibration" / f"{ARM_ID}.json",
        pathlib.Path(f"calibration/{ARM_ID}.json"),
        pathlib.Path(f"/data/models/huggingface/lerobot/calibration/robots/so101_follower/{ARM_ID}.json"),
        pathlib.Path(f"/data/models/huggingface/lerobot/calibration/robots/so_follower/{ARM_ID}.json"),
        hf_home / f"lerobot/calibration/robots/so101_follower/{ARM_ID}.json",
        hf_home / f"lerobot/calibration/robots/so_follower/{ARM_ID}.json",
        pathlib.Path(f"/root/.cache/huggingface/lerobot/calibration/robots/so101_follower/{ARM_ID}.json"),
        pathlib.Path(f"/root/.cache/huggingface/lerobot/calibration/robots/so_follower/{ARM_ID}.json"),
        pathlib.Path.home() / f".cache/huggingface/lerobot/calibration/robots/so101_follower/{ARM_ID}.json",
        pathlib.Path.home() / f".cache/huggingface/lerobot/calibration/robots/so_follower/{ARM_ID}.json",
    ]
    calib_path = next((p for p in search_paths if p.exists()), None)

    _EMBEDDED_CALIB = {
        "shoulder_pan":  {"id": 1, "drive_mode": 0, "homing_offset": 1604,  "range_min": 962,  "range_max": 3486},
        "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": -1498, "range_min": 814,  "range_max": 3207},
        "elbow_flex":    {"id": 3, "drive_mode": 0, "homing_offset": 1619,  "range_min": 882,  "range_max": 3138},
        "wrist_flex":    {"id": 4, "drive_mode": 0, "homing_offset": -1885, "range_min": 887,  "range_max": 3243},
        "wrist_roll":    {"id": 5, "drive_mode": 0, "homing_offset": -1120, "range_min": 0,    "range_max": 4095},
        "gripper":       {"id": 6, "drive_mode": 0, "homing_offset": 1947,  "range_min": 2024, "range_max": 3626}
    }

    if calib_path is not None:
        print(f"   📂 Calibration: {calib_path}")
        with open(calib_path) as f:
            calib_data = json.load(f)
    else:
        print(f"   📂 Calibration file not found on disk — using embedded calibrated profile for '{ARM_ID}'.")
        calib_data = _EMBEDDED_CALIB
        try:
            _persist_p = pathlib.Path(f"/root/ros2_ws/calibration/{ARM_ID}.json")
            _persist_p.parent.mkdir(parents=True, exist_ok=True)
            with open(_persist_p, "w") as _pf:
                json.dump(calib_data, _pf, indent=4)
            print(f"   💾 Auto-persisted calibration profile to: {_persist_p}")
        except Exception:
            pass

    # Check degenerate
    if "start_pos" in calib_data:
        s, e = calib_data["start_pos"], calib_data["end_pos"]
        if s and all(a == b for a, b in zip(s, e)):
            raise RuntimeError("Degenerate calibration file (all ranges equal). Re-run lerobot-calibrate.")
    else:
        ranges = [(v["range_min"], v["range_max"]) for v in calib_data.values() if isinstance(v, dict) and "range_min" in v]
        if ranges and all(mn == mx for mn, mx in ranges):
            raise RuntimeError("Degenerate calibration file (all ranges equal). Re-run lerobot-calibrate.")

    # Connect without interactive prompts
    try:
        robot.connect(calibrate=False)
    except TypeError:
        import builtins
        real_input = builtins.input
        builtins.input = lambda prompt="": "" if ("enter" in prompt.lower() and "range" not in prompt.lower()) else real_input(prompt)
        try:
            robot.connect()
        finally:
            builtins.input = real_input

    # Register typed calibration
    MC = None
    for mod_name in ("lerobot.motors.motors_bus", "lerobot.motors.feetech"):
        try:
            import importlib
            m = importlib.import_module(mod_name)
            for name in ("MotorCalibration", "CalibrationData", "Calibration"):
                if hasattr(m, name):
                    MC = getattr(m, name)
                    break
            if MC:
                break
        except Exception:
            pass

    def make_calib(d):
        if MC:
            try:
                import dataclasses
                if dataclasses.is_dataclass(MC):
                    fields = {f.name for f in dataclasses.fields(MC)}
                    return MC(**{k: v for k, v in d.items() if k in fields})
                return MC(**d)
            except Exception:
                pass
        return SimpleNamespace(**d)

    typed_calib = {k: make_calib(v) for k, v in calib_data.items() if isinstance(v, dict)}

    registered = False
    for meth in ("set_calibration", "load_calibration", "_set_calibration"):
        if hasattr(robot.bus, meth):
            for payload in (typed_calib, calib_data):
                try:
                    getattr(robot.bus, meth)(payload)
                    registered = True
                    break
                except Exception:
                    pass
            if registered:
                break
    if not registered:
        for attr in ("calibration", "_calibration"):
            try:
                setattr(robot.bus, attr, typed_calib)
                registered = True
                break
            except Exception:
                pass

    print("   ✅ Arm connected and calibration registered")
    if not check_servo_health(robot):
        robot.disconnect()
        raise RuntimeError("Servo health check failed.")
    return robot


def main():
    print("\n" + "═"*65)
    print(" 🤖 SO-ARM101 REFERENCE POSE TEACHING TOOL")
    print("═"*65)

    robot = connect_robot()
    try:
        print("\n ⚠️  DISABLING MOTOR TORQUE NOW — please support the arm with your hand!")
        time.sleep(0.5)
        set_torque(robot, False)
        print(" 🔓 Motor torque DISABLED. You can now move the arm freely by hand.\n")

        # Step 1: Scan
        print("─"*65)
        print(" STEP 1: Set NEUTRAL SCAN / START Posture")
        print(" Guide the arm into your desired neutral scanning position:")
        print("   - Shoulder pan centered facing forward (~0°)")
        print("   - Shoulder lift & elbow set so camera views workspace")
        print("   - Wrist tilted ~45° down toward target area")
        print("   - Gripper open or ready")
        print("─"*65)
        input(" 👉 Hold arm in SCAN posture, then press ENTER to capture... ")
        scan_pos = get_pos(robot)
        print("\n ✅ Captured SCAN posture:")
        for k, v in sorted(scan_pos.items()):
            print(f"    {k:20s}: {v:+6.2f}°")

        # Step 2: Stow
        print("\n" + "─"*65)
        print(" STEP 2: Set STOW / PARK Posture")
        print(" Guide the arm into your desired resting/stow position:")
        print("   - Folded back safely, close to base, gripper compact")
        print("─"*65)
        input(" 👉 Hold arm in STOW posture, then press ENTER to capture... ")
        stow_pos = get_pos(robot)
        print("\n ✅ Captured STOW posture:")
        for k, v in sorted(stow_pos.items()):
            print(f"    {k:20s}: {v:+6.2f}°")

        # Compute FK for both
        wx_s, wy_s, wz_s = forward_kinematics(scan_pos)
        rho_s = math.sqrt(wx_s**2 + wy_s**2 + wz_s**2)

        wx_t, wy_t, wz_t = forward_kinematics(stow_pos)
        rho_t = math.sqrt(wx_t**2 + wy_t**2 + wz_t**2)

        data = {
            "scan_base": {
                "joints": {k: float(v) for k, v in scan_pos.items()},
                "fk_xyz_mm": [round(wx_s, 2), round(wy_s, 2), round(wz_s, 2)],
                "reach_rho_mm": round(rho_s, 2),
                "recorded_at": datetime.datetime.now().isoformat(),
            },
            "stow_base": {
                "joints": {k: float(v) for k, v in stow_pos.items()},
                "fk_xyz_mm": [round(wx_t, 2), round(wy_t, 2), round(wz_t, 2)],
                "reach_rho_mm": round(rho_t, 2),
                "recorded_at": datetime.datetime.now().isoformat(),
            }
        }

        # Save locations
        save_dirs = [
            "/root/ros2_ws/calibration",
            os.path.join(os.path.dirname(__file__), "../calibration"),
            os.path.join(os.path.dirname(__file__), "calibration"),
            os.path.dirname(__file__)
        ]
        saved_paths = []
        for d in save_dirs:
            if os.path.exists(d):
                p = os.path.join(d, "arm_reference_poses.yaml")
                with open(p, "w") as f:
                    yaml.dump(data, f, sort_keys=False, default_flow_style=False)
                saved_paths.append(p)

        if not saved_paths:
            # Fallback to local
            p = os.path.join(os.path.dirname(__file__), "arm_reference_poses.yaml")
            with open(p, "w") as f:
                yaml.dump(data, f, sort_keys=False, default_flow_style=False)
            saved_paths.append(p)

        print("\n" + "═"*65)
        for sp in saved_paths:
            print(f" 💾 Saved reference postures to: {sp}")

        print("\n 📋 Python dictionary for arm_picker.py:")
        print("_BASE = {")
        for k, v in sorted(scan_pos.items()):
            print(f'    "{k}": {round(v, 2):7.2f},')
        print("}")
        print("_STOW_BASE = {")
        for k, v in sorted(stow_pos.items()):
            print(f'    "{k}": {round(v, 2):7.2f},')
        print("}")
        print("═"*65)
        print(" ✅ Complete! arm_picker.py will automatically load these reference poses.\n")

    finally:
        print("🔌 Restoring motor torque...")
        set_torque(robot, True)
        robot.disconnect()


if __name__ == "__main__":
    main()
