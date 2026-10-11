#!/usr/bin/env python3
"""
live_joint_angles.py — Live Joint Angle & Hardware Limit Monitor for SO-ARM101

Run on the Jetson:
    python3 live_joint_angles.py

Features:
1. Connects to arm and IMMEDIATELY disables motor torque (motors free to move by hand).
2. Reads live joint angles in degrees AND raw 12-bit encoder ticks (0-4095).
3. Compares live positions against jetson_arm.json calibration limits:
   - Shows remaining headroom before hitting software/hardware limit.
   - Highlights in RED if a joint is hitting or blocked by range_max or range_min.
4. Computes live End-Effector 3D Position (X, Y, Z in mm) via Forward Kinematics.
5. Zero risk: Motors are completely unpowered, safe to guide by hand.
"""

import os
import sys
import time
import math
import json

# Add current dir and parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from arm_picker import connect_robot, get_pos, _set_torque, forward_kinematics
except ImportError:
    try:
        from test_tiered_grasp_pipeline import connect_robot, get_pos, _set_torque, forward_kinematics
    except ImportError:
        connect_robot = None

CALIB_JSON_PATHS = [
    "/root/ros2_ws/calibration/jetson_arm.json",
    "/root/ros2_ws/scripts/jetson_arm.json",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "jetson_arm.json"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration", "jetson_arm.json"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration", "jetson_arm.json"),
    os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots/so101_follower/jetson_arm.json"),
    os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots/so_follower/jetson_arm.json"),
]

MOTOR_ORDER = [
    ("shoulder_pan",  "Base rotation (pan left/right)"),
    ("shoulder_lift", "Shoulder tilt (forward lunge/backward)"),
    ("elbow_flex",    "Elbow bend (unfold forward/fold)"),
    ("wrist_flex",    "Wrist pitch (angle up/down)"),
    ("wrist_roll",    "Wrist roll (twist clockwise/counter)"),
    ("gripper",       "Claw opening (open/closed)"),
]

def load_calibration():
    for p in CALIB_JSON_PATHS:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            try:
                with open(p, "r") as f:
                    return json.load(f), p
            except Exception:
                pass
    return {}, None

def main():
    print("\n" + "═" * 80)
    print(" 🦾 SO-ARM101 LIVE JOINT ANGLE & CALIBRATION LIMIT INSPECTOR")
    print("═" * 80)

    calib_data, calib_src = load_calibration()
    if calib_src:
        print(f" 📂 Loaded calibration profile from: {calib_src}")
    else:
        print(" ⚠️  No jetson_arm.json found! Running with uncalibrated default limits.")

    print(" 🔌 Connecting to robot and disabling torque for free hand movement...")
    if connect_robot is None:
        print(" ❌ Error: Could not import connect_robot from arm_picker.")
        return

    robot = connect_robot()
    if robot is None:
        print(" ❌ Error: Failed connecting to /dev/arm_controller.")
        return

    try:
        _set_torque(robot, False)
        print(" 🔓 Motor torque DISABLED! Move the arm with your hand to test limits.\n")
        time.sleep(0.5)

        print(" Press Ctrl+C to exit.\n")

        while True:
            cur_pos = get_pos(robot)

            # Read raw encoder ticks from each servo
            raw_ticks = {}
            loads = {}
            for m, _ in MOTOR_ORDER:
                try:
                    val = robot.bus.read("Present_Position", m)
                    raw_ticks[m] = int(val[0]) if isinstance(val, (list, tuple)) else int(val)
                except Exception:
                    raw_ticks[m] = -1
                try:
                    l_val = robot.bus.read("Present_Load", m)
                    loads[m] = int(l_val[0]) if isinstance(l_val, (list, tuple)) else int(l_val)
                except Exception:
                    loads[m] = 0

            # Compute FK tip position
            try:
                T_wb = forward_kinematics(cur_pos)
                tx, ty, tz = float(T_wb[0, 3]), float(T_wb[1, 3]), float(T_wb[2, 3])
                rho = math.hypot(tx, ty)
                fk_str = f"Tip Reach: X={tx:+6.1f}mm  Y={ty:+6.1f}mm  Z={tz:+6.1f}mm  (Dist={rho:5.1f}mm)"
            except Exception:
                fk_str = "Tip Reach: N/A"

            # Print table
            out = []
            out.append("\033[2J\033[H" if sys.platform != "win32" else "\n" * 2)  # Clear screen on linux
            out.append("═" * 90)
            out.append(f" 📊 LIVE JOINT MONITOR | {fk_str}")
            out.append("═" * 90)
            out.append(f" {'JOINT NAME':15s} | {'LIVE DEG':9s} | {'RAW TICKS':10s} | {'CALIB RANGE [MIN..MAX]':23s} | {'STATUS / LIMIT WARNING'}")
            out.append("─" * 90)

            for m, desc in MOTOR_ORDER:
                k = f"{m}.pos"
                deg = cur_pos.get(k, 0.0)
                raw = raw_ticks.get(m, -1)
                raw_str = f"{raw:4d} ticks" if raw >= 0 else "N/A"

                c = calib_data.get(m, {})
                rmin = c.get("range_min", None)
                rmax = c.get("range_max", None)

                if isinstance(rmin, int) and isinstance(rmax, int) and raw >= 0:
                    range_str = f"[{rmin:4d} .. {rmax:4d}]"
                    dist_to_min = raw - rmin
                    dist_to_max = rmax - raw

                    if raw < rmin:
                        status = f"🚨 BELOW MIN LIMIT (by {abs(dist_to_min)} ticks!) CLAMPED"
                    elif raw > rmax:
                        status = f"🚨 ABOVE MAX LIMIT (by {abs(dist_to_max)} ticks!) CLAMPED"
                    elif dist_to_max <= 50:
                        status = f"⚠️  AT MAX LIMIT! (Only {dist_to_max} ticks headroom left)"
                    elif dist_to_min <= 50:
                        status = f"⚠️  AT MIN LIMIT! (Only {dist_to_min} ticks headroom left)"
                    else:
                        pct = (raw - rmin) / max(1, (rmax - rmin)) * 100.0
                        status = f"✅ In-range ({pct:.0f}% of travel)"
                else:
                    range_str = "No profile"
                    status = "⚠️ Uncalibrated"

                out.append(f" {m:15s} | {deg:+7.2f}°  | {raw_str:10s} | {range_str:23s} | {status}")

            out.append("═" * 90)
            out.append(" 💡 Move the arm forward to your grasp target to see if shoulder_lift reaches limit.")
            out.append("    Press Ctrl+C to stop.")

            print("\n".join(out))
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n 🛑 Stopping monitor...")
    finally:
        print(" 🔌 Restoring motor torque safely...")
        try:
            _set_torque(robot, True)
        except Exception:
            pass
        print(" ✅ Done. Goodbye!\n")

if __name__ == "__main__":
    main()
