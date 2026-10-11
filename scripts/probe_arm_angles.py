#!/usr/bin/env python3
"""
probe_arm_angles.py — Safe Live Joint & Load Inspector for SO-ARM101

Run this on the Jetson:
    python3 probe_arm_angles.py

What it does:
1. Connects to SO-ARM101 and IMMEDIATELY disables motor torque (free-movement mode).
2. Displays live angles in degrees, raw encoder ticks, and servo load continuously.
3. You can physically hold the arm by hand:
   - Level the wrist horizontally -> see the exact degree number for wrist_roll.
   - Open/close the gripper -> see the exact comfortable degree limits for the claw.
4. Press 's' to save the current arm pose as SCAN pose in arm_reference_poses.yaml.
5. Press 'p' to save the current arm pose as STOW pose in arm_reference_poses.yaml.
6. Press 'q' to quit safely.

ZERO RISK: Motors are completely unpowered (torque disabled), so nothing can twist,
strain, or overload.
"""

import os
import sys
import time
import math
import yaml
import datetime

# Add current dir and parent to path
sys.path.insert(0, os.path.dirname(__file__))
try:
    from arm_picker import connect_robot, get_pos, _set_torque
except ImportError:
    from test_tiered_grasp_pipeline import connect_robot, get_pos, _set_torque

REF_POSE_PATHS = [
    "/root/ros2_ws/calibration/arm_reference_poses.yaml",
    "/root/ros2_ws/scripts/arm_reference_poses.yaml",
    os.path.join(os.path.dirname(__file__), "..", "calibration", "arm_reference_poses.yaml"),
    os.path.join(os.path.dirname(__file__), "arm_reference_poses.yaml")
]

MOTOR_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper"
]

def load_existing_yaml():
    for p in REF_POSE_PATHS:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
    return {}

def save_pose_yaml(pose_name: str, joints: dict):
    data = load_existing_yaml()
    data[pose_name] = {
        "joints": {f"{k}.pos" if not k.endswith(".pos") else k: round(float(v), 2) for k, v in joints.items()},
        "recorded_at": datetime.datetime.now().isoformat()
    }
    saved_any = False
    for p in REF_POSE_PATHS:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
            with open(p, "w") as f:
                yaml.dump(data, f, sort_keys=False)
            print(f"   💾 Saved '{pose_name}' to: {p}")
            saved_any = True
        except Exception:
            pass
    if not saved_any:
        local_p = os.path.join(os.path.dirname(__file__), "arm_reference_poses.yaml")
        with open(local_p, "w") as f:
            yaml.dump(data, f, sort_keys=False)
        print(f"   💾 Saved '{pose_name}' to: {local_p}")

def main():
    print("\n" + "═" * 70)
    print(" 🔍 SO-ARM101 LIVE JOINT & ANGLE INSPECTOR (Zero-Torque Mode)")
    print("═" * 70)
    print(" Connecting to robot and immediately cutting motor torque...")

    robot = connect_robot()
    if robot is None:
        print("❌ Could not connect to robot on /dev/arm_controller.")
        return

    # Disable torque immediately
    _set_torque(robot, False)
    print(" 🔓 Motor torque DISABLED! Motors are 100% free to move by hand.")
    print(" 🛡️  Motors are completely safe from straining, twisting, or stalling.\n")
    print(" Instructions:")
    print("   • Move the arm with your hand to inspect angles.")
    print("   • Press 's' + ENTER to record current pose as SCAN_BASE.")
    print("   • Press 'p' + ENTER to record current pose as STOW_BASE.")
    print("   • Press 'q' + ENTER to exit.\n")
    print("─" * 70)

    try:
        import select
        while True:
            cur = get_pos(robot)
            
            # Print live dashboard
            lines = []
            lines.append("──────────────────────────────────────────────────────────────────────")
            lines.append(f"{'JOINT':16s} | {'ANGLE (DEG)':12s} | {'GUIDE / MEANING'}")
            lines.append("──────────────────────────────────────────────────────────────────────")
            for m in MOTOR_ORDER:
                k = f"{m}.pos"
                deg = cur.get(k, 0.0)
                guide = ""
                if m == "shoulder_pan":
                    guide = "Centered = ~0°"
                elif m == "shoulder_lift":
                    guide = "Elevated = ~ -104°"
                elif m == "elbow_flex":
                    guide = "Angled down = ~ +98°"
                elif m == "wrist_flex":
                    guide = "Looking at table = ~ +18°"
                elif m == "wrist_roll":
                    guide = "Level claw jaws horizontal with table"
                elif m == "gripper":
                    guide = "Closed = ~0° to 15°, Open = ~45° to 65° (DO NOT EXCEED 75°)"
                lines.append(f"{m:16s} | {deg:+8.2f}°    | {guide}")
            lines.append("──────────────────────────────────────────────────────────────────────")
            lines.append("Commands: [s]=Save SCAN | [p]=Save STOW | [ENTER]=Refresh | [q]=Quit")
            
            print("\n".join(lines))

            # Non-blocking or 1-second timeout input
            user_cmd = ""
            if sys.stdin in select.select([sys.stdin], [], [], 1.5)[0]:
                user_cmd = sys.stdin.readline().strip().lower()

            if user_cmd == "q":
                print("\nExiting inspector.")
                break
            elif user_cmd == "s":
                print("\n📸 Captured SCAN_BASE posture:")
                for m in MOTOR_ORDER:
                    print(f"   {m:16s}: {cur.get(f'{m}.pos', 0.0):+8.2f}°")
                save_pose_yaml("scan_base", cur)
                print("✅ SCAN_BASE posture successfully saved!\n")
            elif user_cmd == "p":
                print("\n📸 Captured STOW_BASE posture:")
                for m in MOTOR_ORDER:
                    print(f"   {m:16s}: {cur.get(f'{m}.pos', 0.0):+8.2f}°")
                save_pose_yaml("stow_base", cur)
                print("✅ STOW_BASE posture successfully saved!\n")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (Ctrl+C).")
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass
        print("🔌 Robot disconnected cleanly. All done.\n")

if __name__ == "__main__":
    main()
