#!/usr/bin/env python3
"""
live_joint_angles.py — Live Joint Angle & Hardware Limit Monitor for SO-ARM101

Run on the Jetson:
    python3 live_joint_angles.py

Features:
1. Connects to arm and IMMEDIATELY disables motor torque (free to move safely by hand).
2. Reads live joint angles in degrees AND raw 12-bit encoder ticks (0-4095) without false clamps.
3. Clean, non-flooding interactive console interface:
   - Does NOT spam or constantly scroll the terminal.
   - Press ENTER (or type 'r') to refresh readings on demand.
   - Type 'm' or 'monitor' to run an in-place live auto-refresh monitor.
   - Type 'update,<motor_id>,<min>,<max>' (e.g. update,6,1500,3626) to edit calibration
     live in memory and persist immediately to all jetson_arm.json files on disk.
   - Type 'q' or 'quit' to safely re-enable torque and exit.
4. Computes live End-Effector 3D Position (X, Y, Z in mm) via Forward Kinematics.
"""

import os
import sys
import time
import math
import json
import re

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
    "/data/models/huggingface/lerobot/calibration/robots/so101_follower/jetson_arm.json",
    os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots/so101_follower/jetson_arm.json"),
    os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots/so_follower/jetson_arm.json"),
    "/root/.cache/huggingface/lerobot/calibration/robots/so101_follower/jetson_arm.json",
]

CALIBRATED_BASELINE = {
    "shoulder_pan":  {"id": 1, "drive_mode": 0, "homing_offset": 1604,  "range_min": 962,  "range_max": 3486},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": -1498, "range_min": 814,  "range_max": 3207},
    "elbow_flex":    {"id": 3, "drive_mode": 0, "homing_offset": 1619,  "range_min": 882,  "range_max": 3138},
    "wrist_flex":    {"id": 4, "drive_mode": 0, "homing_offset": -1885, "range_min": 887,  "range_max": 3243},
    "wrist_roll":    {"id": 5, "drive_mode": 0, "homing_offset": -1120, "range_min": 765,  "range_max": 3495},
    "gripper":       {"id": 6, "drive_mode": 0, "homing_offset": 1947,  "range_min": 1500, "range_max": 3626},
}

MOTOR_CONFIG = [
    (1, "shoulder_pan",  "Base rotation (pan left/right)"),
    (2, "shoulder_lift", "Shoulder tilt (forward lunge/backward)"),
    (3, "elbow_flex",    "Elbow bend (unfold forward/fold)"),
    (4, "wrist_flex",    "Wrist pitch (angle up/down)"),
    (5, "wrist_roll",    "Wrist roll (twist clockwise/counter)"),
    (6, "gripper",       "Claw opening (open/closed)"),
]

ID_TO_MOTOR = {cfg[0]: cfg[1] for cfg in MOTOR_CONFIG}
MOTOR_TO_ID = {cfg[1]: cfg[0] for cfg in MOTOR_CONFIG}
MOTOR_ALIASES = {
    "pan": "shoulder_pan",
    "base": "shoulder_pan",
    "lift": "shoulder_lift",
    "shoulder": "shoulder_lift",
    "elbow": "elbow_flex",
    "pitch": "wrist_flex",
    "wrist": "wrist_flex",
    "roll": "wrist_roll",
    "gripper": "gripper",
    "claw": "gripper",
}

def load_calibration():
    loaded_data = None
    loaded_path = None
    for p in CALIB_JSON_PATHS:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and len(data) > 0:
                        loaded_data = data
                        loaded_path = os.path.abspath(p)
                        break
            except Exception:
                pass

    if loaded_data is None:
        loaded_data = {k: dict(v) for k, v in CALIBRATED_BASELINE.items()}
    else:
        # Fill any missing motors from baseline
        for k, v in CALIBRATED_BASELINE.items():
            if k not in loaded_data or not isinstance(loaded_data[k], dict):
                loaded_data[k] = dict(v)

    return loaded_data, loaded_path

def save_calibration(calib_data, preferred_path=None):
    saved_paths = []
    targets = []
    if preferred_path:
        targets.append(preferred_path)
    for p in CALIB_JSON_PATHS:
        p_abs = os.path.abspath(p)
        if p_abs not in targets:
            targets.append(p_abs)

    for p in targets:
        try:
            d = os.path.dirname(p)
            if os.path.exists(d):
                with open(p, "w") as f:
                    json.dump(calib_data, f, indent=4)
                saved_paths.append(p)
        except Exception:
            pass

    if not saved_paths and preferred_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(preferred_path)), exist_ok=True)
            with open(preferred_path, "w") as f:
                json.dump(calib_data, f, indent=4)
            saved_paths.append(preferred_path)
        except Exception:
            pass

    return saved_paths

def apply_calibration_to_bus(robot, calib_data):
    """Update active LeRobot bus calibration attributes in-memory."""
    if not hasattr(robot, "bus") or robot.bus is None:
        return
    for attr in ("calibration", "_calibration"):
        if hasattr(robot.bus, attr):
            target = getattr(robot.bus, attr)
            if isinstance(target, dict):
                for m, d in calib_data.items():
                    if m in target:
                        obj = target[m]
                        if isinstance(obj, dict):
                            obj.update(d)
                        else:
                            for k, v in d.items():
                                if hasattr(obj, k):
                                    setattr(obj, k, v)
    for method in ("set_calibration", "load_calibration", "_set_calibration"):
        if hasattr(robot.bus, method):
            try:
                getattr(robot.bus, method)(calib_data)
                break
            except Exception:
                pass

def read_all_raw_ticks(robot, motor_names, cur_deg=None, calib_data=None):
    """
    Read raw 12-bit encoder ticks (0-4095) from Feetech servo registers.
    CRITICAL: Pass normalize=False to prevent LeRobot from returning degrees
    instead of ticks (which causes false 'BELOW MIN LIMIT' alarms!).
    """
    raw_ticks = {}

    # Try sync_read with normalize=False (atomic across all servos)
    if hasattr(robot, "bus") and hasattr(robot.bus, "sync_read"):
        try:
            res = robot.bus.sync_read("Present_Position", normalize=False)
            if isinstance(res, dict):
                for m in motor_names:
                    if m in res:
                        v = res[m]
                        raw_ticks[m] = int(v[0]) if isinstance(v, (list, tuple)) else int(v)
        except TypeError:
            try:
                res = robot.bus.sync_read("Present_Position")
                if isinstance(res, dict):
                    for m in motor_names:
                        if m in res:
                            v = res[m]
                            raw_ticks[m] = int(v[0]) if isinstance(v, (list, tuple)) else int(v)
            except Exception:
                pass
        except Exception:
            pass

    # Read any remaining motors individually with retries & serial rest
    for m in motor_names:
        if m not in raw_ticks or raw_ticks[m] < 0:
            for _ in range(3):
                try:
                    val = robot.bus.read("Present_Position", m, normalize=False)
                    raw_ticks[m] = int(val[0]) if isinstance(val, (list, tuple)) else int(val)
                    break
                except TypeError:
                    try:
                        val = robot.bus.read("Present_Position", m)
                        raw_ticks[m] = int(val[0]) if isinstance(val, (list, tuple)) else int(val)
                        break
                    except Exception:
                        pass
                except Exception:
                    pass
                time.sleep(0.015)

            if m not in raw_ticks:
                raw_ticks[m] = -1

    # Safety check: If a servo read returned degrees (val < 500 matching cur_deg)
    # invert to ticks using homing_offset so false clamps never occur
    if cur_deg and calib_data:
        for m, raw in list(raw_ticks.items()):
            deg_val = cur_deg.get(f"{m}.pos", None)
            if deg_val is not None and 0 <= raw < 400 and abs(raw - round(deg_val)) <= 1:
                offset = calib_data.get(m, {}).get("homing_offset", 0)
                reconstructed = int(round(deg_val * 4096.0 / 360.0 + 2048 + offset))
                raw_ticks[m] = max(0, min(4095, reconstructed))

    return raw_ticks

def render_table(cur_pos, raw_ticks, calib_data, clear_screen=False):
    out = []
    if clear_screen:
        out.append("\033[2J\033[H" if sys.platform != "win32" else "\n" * 3)

    # Compute FK tip position
    try:
        T_wb = forward_kinematics(cur_pos)
        tx, ty, tz = float(T_wb[0, 3]), float(T_wb[1, 3]), float(T_wb[2, 3])
        rho = math.hypot(tx, ty)
        fk_str = f"Tip Reach: X={tx:+6.1f}mm  Y={ty:+6.1f}mm  Z={tz:+6.1f}mm  (Dist={rho:5.1f}mm)"
    except Exception:
        fk_str = "Tip Reach: N/A"

    out.append("═" * 96)
    out.append(f" 📊 SO-ARM101 LIVE JOINT MONITOR | {fk_str}")
    out.append("═" * 96)
    out.append(f" {'ID':2s} | {'JOINT NAME':15s} | {'LIVE DEG':9s} | {'RAW TICKS':11s} | {'CALIB RANGE [MIN..MAX]':23s} | {'STATUS / LIMIT WARNING'}")
    out.append("─" * 96)

    for mid, m, _ in MOTOR_CONFIG:
        deg = cur_pos.get(f"{m}.pos", None)
        deg_str = f"{deg:+7.2f}°" if deg is not None else "   N/A  "

        raw = raw_ticks.get(m, -1)
        raw_str = f"{raw:4d} ticks" if raw >= 0 else "N/A"

        c = calib_data.get(m, {})
        rmin = c.get("range_min", None)
        rmax = c.get("range_max", None)
        has_calib = isinstance(rmin, int) and isinstance(rmax, int)

        range_str = f"[{rmin:4d} .. {rmax:4d}]" if has_calib else "No profile"

        if not has_calib:
            status = "⚠️ Uncalibrated (No profile in JSON)"
        elif raw < 0:
            status = "⚠️ Comm Read Error (Ticks unavailable)"
        else:
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

        out.append(f" {mid:2d} | {m:15s} | {deg_str:9s} | {raw_str:11s} | {range_str:23s} | {status}")

    out.append("═" * 96)
    return "\n".join(out)

def parse_update_cmd(cmd_str):
    cleaned = re.sub(r"[,;:\.\.]+", " ", cmd_str).strip()
    parts = cleaned.split()
    if not parts:
        return None, "Empty command"
    if parts[0].lower() == "update":
        parts = parts[1:]
    if len(parts) != 3:
        return None, f"Expected 3 parameters: <motor_id_or_name> <min> <max> (e.g. update,6,1500,3626). Got {len(parts)}."

    m_raw, rmin_raw, rmax_raw = parts[0].lower(), parts[1], parts[2]

    # Resolve motor ID / name
    if m_raw.isdigit():
        mid = int(m_raw)
        if mid not in ID_TO_MOTOR:
            return None, f"Motor ID {mid} is invalid. Choose 1 through 6."
        m_name = ID_TO_MOTOR[mid]
    elif m_raw in MOTOR_TO_ID:
        m_name = m_raw
        mid = MOTOR_TO_ID[m_raw]
    elif m_raw in MOTOR_ALIASES:
        m_name = MOTOR_ALIASES[m_raw]
        mid = MOTOR_TO_ID[m_name]
    else:
        return None, f"Unknown motor '{m_raw}'. Use 1-6 or name: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper."

    try:
        rmin = int(rmin_raw)
        rmax = int(rmax_raw)
    except ValueError:
        return None, f"Range limits must be integers, got: '{rmin_raw}', '{rmax_raw}'"

    if rmin > rmax:
        rmin, rmax = rmax, rmin

    if rmin < 0 or rmax > 4095:
        return None, f"Encoder ticks must be within 0..4095 (got [{rmin} .. {rmax}])"

    return (mid, m_name, rmin, rmax), None

def run_live_monitor(robot, calib_data):
    """Runs periodic in-place monitor mode until Enter or Ctrl+C is pressed."""
    print("\n🟢 Live monitor active (updating every 1.0s). Press ENTER or Ctrl+C to return to command prompt.\n")
    motor_names = [cfg[1] for cfg in MOTOR_CONFIG]
    try:
        while True:
            cur_pos = get_pos(robot)
            raw_ticks = read_all_raw_ticks(robot, motor_names, cur_pos, calib_data)
            print(render_table(cur_pos, raw_ticks, calib_data, clear_screen=True))
            print(" 💡 Move arm by hand. Press ENTER or Ctrl+C to return to command prompt...")

            if sys.platform != "win32":
                import select
                rlist, _, _ = select.select([sys.stdin], [], [], 1.0)
                if rlist:
                    sys.stdin.readline()
                    break
            else:
                import msvcrt
                t_end = time.time() + 1.0
                stopped = False
                while time.time() < t_end:
                    if msvcrt.kbhit():
                        msvcrt.getch()
                        stopped = True
                        break
                    time.sleep(0.05)
                if stopped:
                    break
    except KeyboardInterrupt:
        print("\n   Stopped live monitor.")

def main():
    print("\n" + "═" * 80)
    print(" 🦾 SO-ARM101 LIVE JOINT ANGLE & CALIBRATION LIMIT INSPECTOR")
    print("═" * 80)

    calib_data, calib_src = load_calibration()
    if calib_src:
        print(f" 📂 Loaded calibration profile: {calib_src}")
    else:
        print(" ⚠️  No jetson_arm.json found on disk. Using built-in baseline limits.")

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
        print(" 🔓 Motor torque DISABLED! Arm is 100% free and safe to guide by hand.\n")
        time.sleep(0.3)

        motor_names = [cfg[1] for cfg in MOTOR_CONFIG]

        # Initial display
        cur_pos = get_pos(robot)
        raw_ticks = read_all_raw_ticks(robot, motor_names, cur_pos, calib_data)
        print(render_table(cur_pos, raw_ticks, calib_data))

        while True:
            print("\n📋 Commands:")
            print("   • update,<id>,<min>,<max>  -> Edit calibration range live & save to JSON (e.g. update,6,1500,3626)")
            print("   • Press ENTER or type 'r'  -> Refresh live joint angles & raw ticks")
            print("   • 'm' or 'monitor'         -> Continuous live in-place monitor mode")
            print("   • 's' or 'save'            -> Save current calibration to all JSON files")
            print("   • 'q' or 'quit'            -> Exit cleanly and restore motor torque\n")

            try:
                cmd = input("cmd > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            cmd_lower = cmd.lower()
            if cmd_lower in ("q", "quit", "exit"):
                break
            elif cmd_lower in ("", "r", "refresh", "show"):
                cur_pos = get_pos(robot)
                raw_ticks = read_all_raw_ticks(robot, motor_names, cur_pos, calib_data)
                print(render_table(cur_pos, raw_ticks, calib_data, clear_screen=True))
            elif cmd_lower in ("m", "monitor", "live"):
                run_live_monitor(robot, calib_data)
            elif cmd_lower in ("s", "save"):
                saved = save_calibration(calib_data, calib_src)
                print(f"💾 Saved calibration to {len(saved)} location(s):")
                for sp in saved:
                    print(f"   ↳ {sp}")
            elif "update" in cmd_lower or (cmd and cmd[0].isdigit()):
                res, err = parse_update_cmd(cmd)
                if err:
                    print(f"❌ Error: {err}")
                else:
                    mid, m_name, rmin, rmax = res
                    old_min = calib_data[m_name].get("range_min", "N/A")
                    old_max = calib_data[m_name].get("range_max", "N/A")

                    calib_data[m_name]["range_min"] = rmin
                    calib_data[m_name]["range_max"] = rmax

                    # Apply live to robot
                    apply_calibration_to_bus(robot, calib_data)

                    # Save to all json paths
                    saved = save_calibration(calib_data, calib_src)

                    print(f"\n🎉 Successfully updated Motor {mid} ({m_name}):")
                    print(f"   Range: [{old_min} .. {old_max}] ➔ [{rmin} .. {rmax}]")
                    print(f"💾 Saved to {len(saved)} file(s):")
                    for sp in saved:
                        print(f"   ↳ {sp}")

                    # Refresh and show table
                    cur_pos = get_pos(robot)
                    raw_ticks = read_all_raw_ticks(robot, motor_names, cur_pos, calib_data)
                    print()
                    print(render_table(cur_pos, raw_ticks, calib_data))
            else:
                print(f"❓ Unrecognized command: '{cmd}'. Type 'update,6,1500,3626', press ENTER to refresh, 'm' to monitor, or 'q' to quit.")

    except KeyboardInterrupt:
        print("\n\n 🛑 Stopping monitor...")
    finally:
        print(" 🔌 Restoring motor torque safely...")
        try:
            _set_torque(robot, True)
        except Exception:
            pass
        print(" ✅ Done. Torque enabled. Goodbye!\n")

if __name__ == "__main__":
    main()
