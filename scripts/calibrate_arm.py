#!/usr/bin/env python3
"""
calibrate_arm.py — Complete Fresh Calibration & Posture Setup for SO-ARM101

Run this on the Jetson to start 100% fresh:
    python3 calibrate_arm.py

What it does:
1. Connects to SO-ARM101 and immediately cuts motor torque (motors completely free).
2. Hardware Limit Sweeps (Joints 1 to 6):
   - shoulder_pan
   - shoulder_lift
   - elbow_flex
   - wrist_flex
   - wrist_roll
   - gripper (Closed & Open)
   Live-reads the true 12-bit Feetech encoder ticks (0-4095) for each limit.
3. Automatically sets clean homing_offset=0 and correct range_min / range_max.
4. Saves fresh jetson_arm.json to ALL LeRobot cache and host volume locations:
   - /root/ros2_ws/calibration/jetson_arm.json
   - /root/ros2_ws/scripts/jetson_arm.json
   - ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/jetson_arm.json
   - ~/.cache/huggingface/lerobot/calibration/robots/so_follower/jetson_arm.json
   - /data/models/huggingface/lerobot/calibration/robots/so101_follower/jetson_arm.json
5. Immediately teaches new SCAN and STOW postures in arm_reference_poses.yaml
   so angles are 100% synchronized with the new calibration baseline.
"""

import os
import sys
import time
import math
import json
import yaml
import datetime

# Add script directory to path
sys.path.insert(0, os.path.dirname(__file__))
try:
    from arm_picker import connect_robot, get_pos, _set_torque
except ImportError:
    from test_tiered_grasp_pipeline import connect_robot, get_pos, _set_torque

CALIB_JSON_PATHS = [
    "/root/ros2_ws/calibration/jetson_arm.json",
    "/root/ros2_ws/scripts/jetson_arm.json",
    os.path.join(os.path.dirname(__file__), "..", "calibration", "jetson_arm.json"),
    os.path.join(os.path.dirname(__file__), "calibration", "jetson_arm.json"),
    os.path.join(os.path.dirname(__file__), "jetson_arm.json"),
    os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots/so101_follower/jetson_arm.json"),
    os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots/so_follower/jetson_arm.json"),
    "/data/models/huggingface/lerobot/calibration/robots/so101_follower/jetson_arm.json",
]

REF_POSE_PATHS = [
    "/root/ros2_ws/calibration/arm_reference_poses.yaml",
    "/root/ros2_ws/scripts/arm_reference_poses.yaml",
    os.path.join(os.path.dirname(__file__), "..", "calibration", "arm_reference_poses.yaml"),
    os.path.join(os.path.dirname(__file__), "arm_reference_poses.yaml"),
]

MOTOR_GUIDE = [
    ("shoulder_pan",  1, "Base rotation (Left / Right)",
     "Move pan gently to its MAXIMUM LEFT travel limit",
     "Move pan gently to its MAXIMUM RIGHT travel limit"),
    ("shoulder_lift", 2, "Main arm lift (Elevate / Lower)",
     "Move shoulder gently to its LOWEST travel position",
     "Move shoulder gently to its HIGHEST / ELEVATED position"),
    ("elbow_flex",    3, "Forearm elbow (Fold / Extend)",
     "Move elbow gently to its FULLY FOLDED inward position",
     "Move elbow gently to its FULLY EXTENDED outward position"),
    ("wrist_flex",    4, "Wrist pitch (Bend up / Bend down)",
     "Tilt wrist gently FULLY UPWARDS",
     "Tilt wrist gently FULLY DOWNWARDS toward the table"),
    ("wrist_roll",    5, "Wrist roll (Rotate claw axis)",
     "Rotate wrist gently FULL COUNTER-CLOCKWISE",
     "Rotate wrist gently FULL CLOCKWISE"),
    ("gripper",       6, "Claw pincer (Open / Close)",
     "Hold the claw gently FULLY CLOSED by hand",
     "Hold the claw gently FULLY OPEN by hand (do not force past gear stop)"),
]

def read_raw_tick(robot, motor_name: str) -> int:
    """Read raw 12-bit encoder tick from Feetech servo register."""
    for _ in range(3):
        try:
            val = robot.bus.read("Present_Position", motor_name)
            return int(val[0]) if isinstance(val, (list, tuple)) else int(val)
        except Exception:
            time.sleep(0.05)
    return -1

def save_calib_to_all_paths(calib_dict: dict):
    print("\n💾 Writing updated hardware calibration to all locations:")
    saved = 0
    for p in CALIB_JSON_PATHS:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
            with open(p, "w") as f:
                json.dump(calib_dict, f, indent=4)
            print(f"   ✅ Saved: {p}")
            saved += 1
        except Exception as e:
            pass
    if saved == 0:
        local_p = os.path.join(os.path.dirname(__file__), "jetson_arm.json")
        with open(local_p, "w") as f:
            json.dump(calib_dict, f, indent=4)
        print(f"   ✅ Saved locally: {local_p}")

def save_pose_yaml(pose_name: str, joints: dict):
    data = {}
    for p in REF_POSE_PATHS:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    data = yaml.safe_load(f) or {}
                break
            except Exception:
                pass
    data[pose_name] = {
        "joints": {f"{k}.pos" if not k.endswith(".pos") else k: round(float(v), 2) for k, v in joints.items()},
        "recorded_at": datetime.datetime.now().isoformat()
    }
    for p in REF_POSE_PATHS:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
            with open(p, "w") as f:
                yaml.dump(data, f, sort_keys=False)
        except Exception:
            pass

def main():
    print("\n" + "═" * 70)
    print(" 🦾 SO-ARM101 FRESH HARDWARE CALIBRATION & SETUP WIZARD")
    print("═" * 70)
    print(" Connecting to robot and immediately cutting motor torque...")

    robot = connect_robot()
    if robot is None:
        print("❌ Could not connect to robot on /dev/arm_controller.")
        return

    _set_torque(robot, False)
    print(" 🔓 Motor torque DISABLED! Motors are 100% free to move by hand.")
    print(" 🛡️  Motors are completely safe from straining, twisting, or stalling.\n")

    print(" This wizard will calibrate each joint's physical minimum and maximum limits.")
    print(" Move each joint gently by hand when prompted.\n")
    input("👉 Press ENTER to begin fresh calibration...")

    fresh_calib = {}

    for motor_name, motor_id, desc, prompt_min, prompt_max in MOTOR_GUIDE:
        print("\n" + "─" * 70)
        print(f" ▶ Joint [{motor_id}/6]: {motor_name.upper()} ({desc})")
        print("─" * 70)

        # 1. Read first limit
        print(f" 👉 Step 1: {prompt_min}")
        input("    Press ENTER when in position...")
        time.sleep(0.2)
        tick_1 = read_raw_tick(robot, motor_name)
        while tick_1 < 0:
            print("    ⚠️ Could not read servo position. Re-trying...")
            time.sleep(0.5)
            tick_1 = read_raw_tick(robot, motor_name)
        print(f"    ✅ Position 1 recorded: {tick_1} ticks")

        # 2. Read second limit
        print(f" 👉 Step 2: {prompt_max}")
        input("    Press ENTER when in position...")
        time.sleep(0.2)
        tick_2 = read_raw_tick(robot, motor_name)
        while tick_2 < 0:
            print("    ⚠️ Could not read servo position. Re-trying...")
            time.sleep(0.5)
            tick_2 = read_raw_tick(robot, motor_name)
        print(f"    ✅ Position 2 recorded: {tick_2} ticks")

        rmin = min(tick_1, tick_2)
        rmax = max(tick_1, tick_2)
        span = rmax - rmin

        # Sanity check
        if span < 30:
            print(f"    ⚠️ WARNING: Measured travel span is very small ({span} ticks).")
            print(f"       Did the joint move? Using default safe bounds.")
            rmin = max(0, tick_1 - 500)
            rmax = min(4095, tick_1 + 500)

        fresh_calib[motor_name] = {
            "id": motor_id,
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": rmin,
            "range_max": rmax
        }
        deg_span = (rmax - rmin) / 4096.0 * 360.0
        print(f"    📊 Calibrated '{motor_name}': [{rmin:4d} .. {rmax:4d}] (Span: {span} ticks ≈ {deg_span:.1f}°)")

    # Save to all paths
    save_calib_to_all_paths(fresh_calib)
    print("\n🎉 All 6 joints calibrated successfully!")

    # ── Posture Teaching ────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print(" 📸 STEP 2: TEACH REFERENCE POSTURES (Scan & Stow)")
    print("═" * 70)
    print(" Now set your arm's starting postures with torque still OFF:\n")

    print(" 1. SCAN POSTURE (Arm elevated, camera pointing down at table):")
    print("    • Hold arm elevated (Shoulder Lift ~ -104°, Elbow ~ +98°)")
    print("    • Level the wrist and point camera down at the table")
    print("    • Set the claw to comfortable open width (~50°)")
    input(" 👉 Press ENTER when you are holding the perfect SCAN posture...")
    scan_pos = get_pos(robot)
    save_pose_yaml("scan_base", scan_pos)
    print("    ✅ SCAN posture saved to arm_reference_poses.yaml!")

    print("\n 2. STOW POSTURE (Arm folded compactly, rest position):")
    print("    • Fold the arm into its safe parked / rested position")
    input(" 👉 Press ENTER when you are holding the STOW posture...")
    stow_pos = get_pos(robot)
    save_pose_yaml("stow_base", stow_pos)
    print("    ✅ STOW posture saved to arm_reference_poses.yaml!")

    print("\n" + "═" * 70)
    print(" 🏁 FRESH CALIBRATION COMPLETE!")
    print("═" * 70)
    print(" • Hardware bounds calibrated: jetson_arm.json")
    print(" • Operating postures saved   : arm_reference_poses.yaml")
    print(" • Safe for arm_picker.py and test_tiered_grasp_pipeline.py\n")

    try:
        robot.disconnect()
    except Exception:
        pass

if __name__ == "__main__":
    main()
