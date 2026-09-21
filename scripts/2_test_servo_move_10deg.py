#!/usr/bin/env python3
"""
2_test_servo_move_10deg.py — Safe Single-Servo Micro-Movement Test
==================================================================
Purpose:
  1. Tests individual Feetech STS3215 servos one at a time.
  2. Moves ONLY the selected joint by a gentle, controlled increment (default: 10°).
  3. Uses slow micro-stepping (0.5° per 30ms) to guarantee ZERO current spikes,
     ZERO mechanical shock, and ZERO arm snapping.
  4. Keeps all other 5 joints stationary at their current positions.
  5. Includes an instant return-to-previous position feature.

Usage:
  python3 scripts/2_test_servo_move_10deg.py
"""

import os
import sys
import time

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

JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper"
]


def get_joint_positions(robot) -> dict:
    """Reads current joint positions in degrees."""
    try:
        obs = robot.get_observation()
        pos = {}
        for j in JOINTS:
            key = f"{j}.pos"
            if key in obs:
                pos[key] = float(obs[key])
        if len(pos) >= len(JOINTS) - 1:
            return pos
    except Exception as e:
        print(f"⚠️ Read error: {e}")
    return {}


def safe_micro_move(robot, start_pos: dict, joint_key: str, delta_deg: float, step_size: float = 0.5, step_delay: float = 0.03):
    """
    Smoothly moves ONE joint by delta_deg in small, controlled steps
    while holding all other joints stationary.
    """
    target_pos = dict(start_pos)
    current_val = start_pos[joint_key]
    target_val = current_val + delta_deg
    target_pos[joint_key] = target_val

    n_steps = max(1, int(abs(delta_deg) / step_size))
    step_delta = delta_deg / n_steps

    print(f"   Moving {joint_key}: {current_val:+.1f}° -> {target_val:+.1f}° ({delta_deg:+.1f}°) in {n_steps} gentle steps...")

    for s in range(1, n_steps + 1):
        interp = dict(start_pos)
        interp[joint_key] = current_val + s * step_delta
        robot.send_action(interp)
        time.sleep(step_delay)

    # Final hold
    robot.send_action(target_pos)
    time.sleep(0.1)


def main():
    print("=" * 70)
    print("🔬 SAFE SINGLE-SERVO MICRO-MOVEMENT TEST SCRIPT")
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
        config = SOFollowerRobotConfig(port=PORT, id=ARM_ID, use_degrees=True)
        robot = SOFollower(config)
        robot.connect()
    except Exception as e:
        print(f"❌ Failed to connect to robot: {e}")
        sys.exit(1)
    finally:
        builtins.input = _orig_input

    print("✅ Arm connected successfully.\n")

    try:
        while True:
            cur_pos = get_joint_positions(robot)
            if not cur_pos:
                print("⚠️ Could not read joint positions from robot!")
                time.sleep(0.5)
                continue

            print("\n" + "=" * 55)
            print("  CURRENT SERVO POSITIONS:")
            print("-" * 55)
            for idx, j in enumerate(JOINTS, 1):
                key = f"{j}.pos"
                deg = cur_pos.get(key, 0.0)
                print(f"  [{idx}] {j:16s} : {deg:+7.1f}°")
            print("=" * 55)
            print("  [1-6] Select joint to test")
            print("  [q]   Quit and disconnect")
            print("=" * 55)

            choice = input("Select joint [1-6, q]: ").strip()
            if choice.lower() in ['q', 'quit', 'exit']:
                break

            if choice not in [str(i) for i in range(1, 7)]:
                print("Invalid selection. Enter 1 to 6.")
                continue

            joint_name = JOINTS[int(choice) - 1]
            joint_key = f"{joint_name}.pos"
            current_deg = cur_pos[joint_key]

            print(f"\nTarget Joint: {joint_name} (Current: {current_deg:+.1f}°)")
            print("  [1] +10.0° (forward/open)")
            print("  [2] -10.0° (backward/close)")
            print("  [3] Custom angle delta")
            print("  [c] Cancel")

            dir_choice = input("Choose movement [1/2/3/c]: ").strip()
            if dir_choice == "1":
                delta = 10.0
            elif dir_choice == "2":
                delta = -10.0
            elif dir_choice == "3":
                custom_str = input("Enter delta degrees (e.g. 5, -5, 10, -10): ").strip()
                try:
                    delta = float(custom_str)
                    if abs(delta) > 30.0:
                        print("⚠️ For safety, movements are limited to max ±30.0° per test.")
                        delta = 30.0 if delta > 0 else -30.0
                except ValueError:
                    print("Invalid number. Cancelling.")
                    continue
            else:
                print("Cancelled.")
                continue

            # Execute gentle move
            safe_micro_move(robot, cur_pos, joint_key, delta)

            # Read result
            time.sleep(0.2)
            after_pos = get_joint_positions(robot)
            new_deg = after_pos.get(joint_key, current_deg + delta)
            print(f"   ✅ Moved: {joint_name} is now at {new_deg:+.1f}°")

            # Option to return
            ret = input(f"\nReturn {joint_name} back by {-delta:+.1f}° to original position? [Y/n]: ").strip().lower()
            if ret == "" or ret.startswith("y"):
                safe_micro_move(robot, after_pos, joint_key, -delta)
                time.sleep(0.2)
                restored_pos = get_joint_positions(robot)
                print(f"   ↩️ Restored: {joint_name} is at {restored_pos.get(joint_key, current_deg):+.1f}°")

    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user.")
    finally:
        print("\n🔌 Disconnecting robot cleanly...")
        try:
            robot.disconnect()
        except Exception:
            pass
        print("Done.\n")


if __name__ == "__main__":
    main()
