#!/usr/bin/env python3
"""
probe_arm_angles.py — Live Joint, Raw Encoder & Rapid Calibration Tool for SO-ARM101

Run this on the Jetson:
    python3 probe_arm_angles.py

What it does:
1. Connects to SO-ARM101 and IMMEDIATELY disables motor torque (free-movement mode).
2. Displays live angles in degrees, RAW 12-bit encoder ticks (0-4095), and calibration ranges.
3. Rapid Gripper Calibration:
   - Press 'c' -> hold claw fully closed -> press ENTER -> hold fully open -> press ENTER.
   - Instantly recalculates range_min and range_max and updates jetson_arm.json.
4. Full Arm Calibration:
   - Press 'a' -> sweeps all 6 joints (min and max limits) and updates jetson_arm.json.
5. Reference Pose Saving:
   - Press 's' to save current posture as SCAN_BASE.
   - Press 'p' to save current posture as STOW_BASE.
6. Press 'q' to quit safely.

ZERO RISK: Motors are completely unpowered (torque disabled), so nothing can twist,
strain, or overload.
"""

import os
import sys
import time
import math
import yaml
import json
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

CALIB_JSON_PATHS = [
    "/root/ros2_ws/calibration/jetson_arm.json",
    "/root/ros2_ws/scripts/jetson_arm.json",
    os.path.join(os.path.dirname(__file__), "..", "calibration", "jetson_arm.json"),
    os.path.join(os.path.dirname(__file__), "jetson_arm.json"),
    os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots/so101_follower/jetson_arm.json"),
    os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots/so_follower/jetson_arm.json"),
    "/data/models/huggingface/lerobot/calibration/robots/so101_follower/jetson_arm.json",
]

MOTOR_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper"
]

def load_calib_json():
    for p in CALIB_JSON_PATHS:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            try:
                with open(p, "r") as f:
                    return json.load(f), p
            except Exception:
                pass
    return {}, None

def save_calib_json(calib_data):
    saved_any = False
    for p in CALIB_JSON_PATHS:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
            with open(p, "w") as f:
                json.dump(calib_data, f, indent=4)
            print(f"   💾 Saved calibration to: {p}")
            saved_any = True
        except Exception:
            pass
    if not saved_any:
        local_p = os.path.join(os.path.dirname(__file__), "jetson_arm.json")
        with open(local_p, "w") as f:
            json.dump(calib_data, f, indent=4)
        print(f"   💾 Saved calibration to: {local_p}")

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
    print("\n" + "═" * 75)
    print(" 🔍 SO-ARM101 LIVE JOINT, ENCODER & RAPID CALIBRATION TOOL")
    print("═" * 75)
    print(" Connecting to robot and immediately cutting motor torque...")

    robot = connect_robot()
    if robot is None:
        print("❌ Could not connect to robot on /dev/arm_controller.")
        return

    # Disable torque immediately
    _set_torque(robot, False)
    print(" 🔓 Motor torque DISABLED! Motors are 100% free to move by hand.")
    print(" 🛡️  Motors are completely safe from straining, twisting, or stalling.\n")

    calib_data, calib_src = load_calib_json()
    if calib_src:
        print(f" 📂 Active calibration profile: {calib_src}\n")

    print(" Instructions:")
    print("   • Move the arm with your hand to inspect live angles & raw ticks.")
    print("   • Press 'c' + ENTER: Rapid Gripper Recalibration (Closed -> Open).")
    print("   • Press 'a' + ENTER: Full Arm Recalibration (All 6 joints min/max).")
    print("   • Press 's' + ENTER: Save current pose as SCAN_BASE in YAML.")
    print("   • Press 'p' + ENTER: Save current pose as STOW_BASE in YAML.")
    print("   • Press 'q' + ENTER: Exit cleanly.\n")

    try:
        import select
        while True:
            cur = get_pos(robot)

            # Read raw encoder ticks from servos
            raw_ticks = {}
            for m in MOTOR_ORDER:
                try:
                    val = robot.bus.read("Present_Position", m)
                    raw_ticks[m] = int(val[0]) if isinstance(val, (list, tuple)) else int(val)
                except Exception:
                    raw_ticks[m] = -1

            # Print live dashboard
            lines = []
            lines.append("───────────────────────────────────────────────────────────────────────────────────")
            lines.append(f"{'JOINT':15s} | {'DEGREE':10s} | {'RAW ENCODER':13s} | {'CALIB RANGE [MIN..MAX]':23s} | {'STATUS'}")
            lines.append("───────────────────────────────────────────────────────────────────────────────────")
            for m in MOTOR_ORDER:
                k = f"{m}.pos"
                deg = cur.get(k, 0.0)
                raw = raw_ticks.get(m, -1)
                raw_str = f"{raw:4d} ticks" if raw >= 0 else "N/A"

                c_info = calib_data.get(m, {})
                rmin = c_info.get("range_min", None)
                rmax = c_info.get("range_max", None)
                if isinstance(rmin, int) and isinstance(rmax, int):
                    range_str = f"[{rmin:4d} .. {rmax:4d}]"
                    if raw >= 0:
                        if raw < rmin:
                            status = "⚠️ BELOW MIN (clamped 0°)"
                        elif raw > rmax:
                            status = "⚠️ ABOVE MAX (clamped max)"
                        else:
                            pct = (raw - rmin) / max(1, (rmax - rmin)) * 100.0
                            status = f"✅ In-range ({pct:.0f}%)"
                    else:
                        status = "OK"
                else:
                    range_str = "No profile"
                    status = "⚠️ Missing"

                lines.append(f"{m:15s} | {deg:+7.2f}°   | {raw_str:13s} | {range_str:23s} | {status}")
            lines.append("───────────────────────────────────────────────────────────────────────────────────")
            lines.append("Commands: [c]=Calib Gripper | [a]=Calib ALL | [s]=Save SCAN | [p]=Save STOW | [q]=Quit")

            print("\n".join(lines))

            # Non-blocking input (1.5 sec refresh)
            user_cmd = ""
            if sys.stdin in select.select([sys.stdin], [], [], 1.5)[0]:
                user_cmd = sys.stdin.readline().strip().lower()

            if user_cmd == "q":
                print("\nExiting inspector.")
                break

            elif user_cmd == "c":
                print("\n" + "=" * 65)
                print(" 🦾 RAPID GRIPPER RE-CALIBRATION (Torque is OFF)")
                print("=" * 65)
                input("👉 1. Gently hold the claw FULLY CLOSED by hand, then press ENTER...")
                time.sleep(0.3)
                v_closed = robot.bus.read("Present_Position", "gripper")
                t_closed = int(v_closed[0]) if isinstance(v_closed, (list, tuple)) else int(v_closed)
                print(f"   🔒 Closed Position Recorded: {t_closed} ticks")

                input("👉 2. Gently hold the claw FULLY OPEN by hand, then press ENTER...")
                time.sleep(0.3)
                v_open = robot.bus.read("Present_Position", "gripper")
                t_open = int(v_open[0]) if isinstance(v_open, (list, tuple)) else int(v_open)
                print(f"   🖐 Open Position Recorded  : {t_open} ticks")

                rmin = min(t_closed, t_open)
                rmax = max(t_closed, t_open)
                span = rmax - rmin
                print(f"\n📊 Measured Physical Span: {span} ticks (Range: [{rmin} .. {rmax}])")

                if span < 50:
                    print("⚠️  Measured span is too small (<50 ticks). Gripper was not moved. Aborting.")
                else:
                    if "gripper" not in calib_data:
                        calib_data["gripper"] = {"id": 6, "drive_mode": 0, "homing_offset": 0}
                    calib_data["gripper"]["range_min"] = rmin
                    calib_data["gripper"]["range_max"] = rmax
                    calib_data["gripper"]["homing_offset"] = 0
                    save_calib_json(calib_data)
                    print("🎉 Gripper calibration successfully updated!")
                    print("   ↳ Restart probe_arm_angles.py or arm_picker.py to apply new limits.\n")
                    time.sleep(1.5)

            elif user_cmd == "a":
                print("\n" + "=" * 65)
                print(" 🦾 FULL ARM HARDWARE CALIBRATION WIZARD (Torque is OFF)")
                print("=" * 65)
                for i, m in enumerate(MOTOR_ORDER, start=1):
                    print(f"\n▶ [{i}/6] Motor: {m.upper()}")
                    input(f"👉 Move '{m}' to its MINIMUM physical travel limit, then press ENTER...")
                    time.sleep(0.2)
                    vm = robot.bus.read("Present_Position", m)
                    t_min = int(vm[0]) if isinstance(vm, (list, tuple)) else int(vm)
                    print(f"   Min Recorded: {t_min} ticks")

                    input(f"👉 Move '{m}' to its MAXIMUM physical travel limit, then press ENTER...")
                    time.sleep(0.2)
                    vx = robot.bus.read("Present_Position", m)
                    t_max = int(vx[0]) if isinstance(vx, (list, tuple)) else int(vx)
                    print(f"   Max Recorded: {t_max} ticks")

                    rmin = min(t_min, t_max)
                    rmax = max(t_min, t_max)
                    if m not in calib_data:
                        calib_data[m] = {"id": i, "drive_mode": 0, "homing_offset": 0}
                    calib_data[m]["range_min"] = rmin
                    calib_data[m]["range_max"] = rmax
                    calib_data[m]["homing_offset"] = 0
                    print(f"   ✅ Saved {m}: [{rmin} .. {rmax}] (span: {rmax - rmin} ticks)")

                save_calib_json(calib_data)
                print("\n🎉 Full arm calibration completed and saved to all locations!\n")
                time.sleep(1.5)

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
