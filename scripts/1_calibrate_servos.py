#!/usr/bin/env python3
"""
1_calibrate_servos.py — Safe Range-of-Motion Calibration for SO-ARM101 Servos
=============================================================================
Purpose:
  1. Connects to the Feetech STS3215 bus servos on /dev/arm_controller.
  2. Disables motor torque so you can manually move the arm by hand without resistance.
  3. Real-time tracks the MINIMUM and MAXIMUM encoder ticks (0-4095) for each joint.
  4. Saves clean, verified calibration data with homing_offset=0 directly to:
     - calibration/jetson_arm.json
     - ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/jetson_arm.json
     - /data/models/huggingface/lerobot/calibration/robots/so101_follower/jetson_arm.json

Usage:
  python3 scripts/1_calibrate_servos.py
"""

import os
import sys
import time
import json
import shutil
import select
import termios
import tty

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

PORT = os.environ.get("ARM_PORT", "/dev/arm_controller")
ARM_ID = os.environ.get("ARM_ID", "jetson_arm")

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper"
]

MOTOR_IDS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}

# Known safe baseline fallback ranges (ticks) if user only moves a subset
BASELINE_SAFE_LIMITS = {
    "shoulder_pan":  {"range_min": 866, "range_max": 3187},
    "shoulder_lift": {"range_min": 750, "range_max": 3250},
    "elbow_flex":    {"range_min": 900, "range_max": 3250},
    "wrist_flex":    {"range_min": 764, "range_max": 2815},
    "wrist_roll":    {"range_min": 765, "range_max": 3495},
    "gripper":       {"range_min": 766, "range_max": 2049},
}


def set_torque(robot, enable: bool) -> bool:
    """Safely enable or disable torque across any LeRobot version."""
    if robot is None or not hasattr(robot, "bus"):
        return False
    val = 1 if enable else 0
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
        robot.bus.write("Torque_Enable", [val] * len(MOTOR_NAMES), MOTOR_NAMES)
        return True
    except Exception:
        pass
    try:
        robot.bus.write("Torque_Enable", val, MOTOR_NAMES)
        return True
    except Exception:
        pass
    try:
        for m in MOTOR_NAMES:
            try:
                robot.bus.write("Torque_Enable", val, [m])
            except Exception:
                robot.bus.write("Torque_Enable", [m], val)
        return True
    except Exception:
        pass
    return False


def get_raw_ticks(robot) -> dict:
    """Reads raw integer encoder ticks (0-4095) directly from the Feetech servo bus."""
    try:
        positions = robot.bus.read("Present_Position", MOTOR_NAMES)
        return {name: int(pos) for name, pos in zip(MOTOR_NAMES, positions)}
    except Exception:
        pass
    try:
        obs = robot.get_observation()
        ticks = {}
        for m in MOTOR_NAMES:
            key = f"{m}.pos"
            if key in obs:
                ticks[m] = int(obs[key])
        if len(ticks) == len(MOTOR_NAMES):
            return ticks
    except Exception:
        pass
    return {}


def is_data_waiting():
    """Non-blocking check for keyboard input."""
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


def save_calibration(stats: dict):
    """Saves valid calibration data with homing_offset: 0 to all required target locations."""
    calib_data = {}
    for m in MOTOR_NAMES:
        min_v = stats[m]["min"]
        max_v = stats[m]["max"]
        # If user didn't move this joint, use safe baseline
        if min_v >= max_v or min_v < 0 or max_v > 4095:
            min_v = BASELINE_SAFE_LIMITS[m]["range_min"]
            max_v = BASELINE_SAFE_LIMITS[m]["range_max"]

        calib_data[m] = {
            "id": MOTOR_IDS[m],
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": int(min_v),
            "range_max": int(max_v),
        }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_paths = [
        os.path.join(script_dir, "..", "calibration", "jetson_arm.json"),
        os.path.join(script_dir, "calibration", "jetson_arm.json"),
        "/root/ros2_ws/calibration/jetson_arm.json",
        "/data/models/huggingface/lerobot/calibration/robots/so101_follower/jetson_arm.json",
        os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots/so101_follower/jetson_arm.json"),
    ]

    saved_locations = []
    for p in target_paths:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
            with open(p, "w") as f:
                json.dump(calib_data, f, indent=4)
            saved_locations.append(p)
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("✅ CALIBRATION SAVED SUCCESSFULLY!")
    print("=" * 70)
    for p in saved_locations:
        print(f"  📁 {p}")
    print("=" * 70)


def main():
    print("=" * 70)
    print("🦾 SO-ARM101 SERVO RANGE-OF-MOTION CALIBRATION TOOL")
    print("=" * 70)
    print(f"Connecting to port: {PORT} (ID: {ARM_ID})...")

    if SOFollower is None or SOFollowerRobotConfig is None:
        print("❌ Error: LeRobot SOFollower class could not be loaded!")
        sys.exit(1)

    # Auto-confirm LeRobot prompt if it asks
    import builtins
    _orig_input = builtins.input
    builtins.input = lambda prompt="": ""

    try:
        config = SOFollowerRobotConfig(port=PORT, id=ARM_ID, use_degrees=False)
        robot = SOFollower(config)
        robot.connect()
    except Exception as e:
        print(f"❌ Failed to connect to robot: {e}")
        sys.exit(1)
    finally:
        builtins.input = _orig_input

    print("✅ Connected to Feetech servo bus.")
    print("🔌 Disabling motor torque... You can now move each joint by hand!")
    set_torque(robot, False)

    # Read starting positions
    initial_ticks = {}
    for _ in range(10):
        initial_ticks = get_raw_ticks(robot)
        if len(initial_ticks) == len(MOTOR_NAMES):
            break
        time.sleep(0.05)

    if not initial_ticks:
        print("⚠️ Warning: Could not read initial ticks directly. Setting defaults.")
        initial_ticks = {m: 2048 for m in MOTOR_NAMES}

    stats = {}
    for m in MOTOR_NAMES:
        curr = initial_ticks.get(m, 2048)
        stats[m] = {
            "min": curr,
            "max": curr,
            "current": curr,
        }

    print("\n" + "─" * 70)
    print("  INSTRUCTIONS:")
    print("  1. Slowly move each servo by hand from its MINIMUM to its MAXIMUM limit.")
    print("  2. Watch the live readings update in the table below.")
    print("  3. Press [s] or [ENTER] to save calibration, [r] to reset min/max, [q] to quit.")
    print("─" * 70 + "\n")

    # Save original terminal settings for raw keypress detection
    old_term = None
    if sys.stdin.isatty():
        try:
            old_term = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except Exception:
            old_term = None

    try:
        while True:
            # Check for key presses
            if is_data_waiting():
                ch = sys.stdin.read(1)
                if ch in ['q', 'Q', '\x03']: # q or Ctrl+C
                    print("\nCalibration cancelled by user.")
                    break
                elif ch in ['s', 'S', '\r', '\n']:
                    save_calibration(stats)
                    break
                elif ch in ['r', 'R']:
                    for m in MOTOR_NAMES:
                        curr = stats[m]["current"]
                        stats[m]["min"] = curr
                        stats[m]["max"] = curr
                    print("\n🔄 Min/Max limits reset to current position.")

            ticks = get_raw_ticks(robot)
            if ticks:
                for m, val in ticks.items():
                    stats[m]["current"] = val
                    stats[m]["min"] = min(stats[m]["min"], val)
                    stats[m]["max"] = max(stats[m]["max"], val)

            # Draw table
            sys.stdout.write("\033[H\033[J") # Clear screen
            print("=" * 72)
            print(f"  {'JOINT NAME':16s} | {'MIN':6s} | {'LIVE':6s} | {'MAX':6s} | {'TRAVEL':6s} | {'LIVE DEG':8s}")
            print("-" * 72)
            for m in MOTOR_NAMES:
                s = stats[m]
                rng = s["max"] - s["min"]
                deg = (s["current"] - 2048) * (360.0 / 4096.0)
                print(f"  {m:16s} | {s['min']:6d} | {s['current']:6d} | {s['max']:6d} | {rng:6d} | {deg:+7.1f}°")
            print("=" * 72)
            print("  [s / ENTER] Save Calibration  |  [r] Reset Min/Max  |  [q] Quit")
            print("=" * 72)
            sys.stdout.flush()

            time.sleep(0.06)

    except KeyboardInterrupt:
        print("\n\n⏹️ Interrupted.")
        choice = input("Do you want to save the recorded limits before exiting? (y/n): ").strip().lower()
        if choice.startswith('y'):
            save_calibration(stats)
    finally:
        if old_term is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
            except Exception:
                pass
        print("\n🔌 Disconnecting robot cleanly...")
        try:
            robot.disconnect()
        except Exception:
            pass
        print("Done.\n")


if __name__ == "__main__":
    main()
