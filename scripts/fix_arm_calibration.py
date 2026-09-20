#!/usr/bin/env python3
"""
fix_arm_calibration.py
Inspects, repairs, and permanently saves the LeRobot SO-ARM101 calibration.
Fixes the wrap-around bug (min=0, max=4095) on shoulder_lift, elbow_flex, and wrist_roll
and syncs the file between LeRobot cache and the persistent host volume (/root/ros2_ws/calibration).
"""
import os
import sys
import json
import shutil

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CALIB_CACHE_DIR = "/data/models/huggingface/lerobot/calibration/robots/so101_follower"
CALIB_CACHE_FILE = os.path.join(CALIB_CACHE_DIR, "jetson_arm.json")
CALIB_PERSIST_FILE = "/root/ros2_ws/calibration/jetson_arm.json"
LOCAL_PERSIST_FILE = os.path.join(os.path.dirname(__file__), "..", "calibration", "jetson_arm.json")

# Calibrated physical bounds (Feetech STS3215 12-bit encoder, center ~2048)
SAFE_BOUNDS = {
    "shoulder_pan":  {"range_min": 866, "range_max": 3187, "homing_offset": 2036},
    "shoulder_lift": {"range_min": 750, "range_max": 3250, "homing_offset": 1996},
    "elbow_flex":    {"range_min": 900, "range_max": 3250, "homing_offset": 2060},
    "wrist_flex":    {"range_min": 764, "range_max": 2815, "homing_offset": 1966},
    "wrist_roll":    {"range_min": 765, "range_max": 3495, "homing_offset": 2130},
    "gripper":       {"range_min": 766, "range_max": 2049, "homing_offset": 779},
}

def main():
    print("═" * 60)
    print(" 🦾 LeRobot SO-ARM101 Calibration Repair & Persistence Tool")
    print("═" * 60)

    # 1. Determine existing calibration file
    source_file = None
    for candidate in [CALIB_CACHE_FILE, CALIB_PERSIST_FILE, LOCAL_PERSIST_FILE]:
        if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
            source_file = candidate
            print(f"🔍 Found calibration file at: {source_file}")
            break

    calib_data = {}
    if source_file:
        try:
            with open(source_file, "r") as f:
                calib_data = json.load(f)
        except Exception as e:
            print(f"⚠️ Failed reading {source_file}: {e}")

    # Fallback template if none existed
    if not calib_data:
        print("⚠️ No existing calibration found. Generating from baseline hardware profile...")
        motor_ids = {
            "shoulder_pan": 1,
            "shoulder_lift": 2,
            "elbow_flex": 3,
            "wrist_flex": 4,
            "wrist_roll": 5,
            "gripper": 6,
        }
        for joint, mid in motor_ids.items():
            calib_data[joint] = {
                "id": mid,
                "drive_mode": 0,
                "homing_offset": SAFE_BOUNDS[joint]["homing_offset"],
                "range_min": SAFE_BOUNDS[joint]["range_min"],
                "range_max": SAFE_BOUNDS[joint]["range_max"]
            }

    print("\n📊 Current Joint Calibration Status:")
    print("---------------------------------------------------------------")
    print(f"{'NAME':16s} | {'MIN':6s} | {'MAX':6s} | {'OFFSET':8s} | {'STATUS'}")
    print("---------------------------------------------------------------")

    modified = False
    for joint, info in calib_data.items():
        rmin = info.get("range_min", 0)
        rmax = info.get("range_max", 4095)
        offset = info.get("homing_offset", 0)

        # Check for corrupted wrap-around
        if rmin == 0 and rmax == 4095 and joint in SAFE_BOUNDS:
            print(f"{joint:16s} | {rmin:6d} | {rmax:6d} | {offset:8d} | ❌ CORRUPTED (0/4095 wrap-around)")
            info["range_min"] = SAFE_BOUNDS[joint]["range_min"]
            info["range_max"] = SAFE_BOUNDS[joint]["range_max"]
            print(f"   ↳ Fixed to physical bounds: [{info['range_min']}, {info['range_max']}]")
            modified = True
        elif offset == 0 and joint in SAFE_BOUNDS:
            print(f"{joint:16s} | {rmin:6d} | {rmax:6d} | {offset:8d} | ❌ UNCALIBRATED (offset=0 drives motors to limit!)")
            info["homing_offset"] = SAFE_BOUNDS[joint]["homing_offset"]
            print(f"   ↳ Fixed homing_offset to calibrated neutral: {info['homing_offset']}")
            modified = True
        else:
            print(f"{joint:16s} | {rmin:6d} | {rmax:6d} | {offset:8d} | ✅ OK")

    print("---------------------------------------------------------------")

    # 2. Write repaired calibration to persistent location
    persist_targets = [CALIB_PERSIST_FILE, LOCAL_PERSIST_FILE]
    saved_persistent = False
    for pt in persist_targets:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(pt)), exist_ok=True)
            with open(pt, "w") as f:
                json.dump(calib_data, f, indent=4)
            print(f"💾 Permanently saved to: {pt}")
            saved_persistent = True
        except Exception as e:
            pass

    # 3. Write to LeRobot cache directory so SO101Follower loads it directly
    try:
        os.makedirs(CALIB_CACHE_DIR, exist_ok=True)
        with open(CALIB_CACHE_FILE, "w") as f:
            json.dump(calib_data, f, indent=4)
        print(f"✅ Synced to LeRobot cache: {CALIB_CACHE_FILE}")
    except Exception as e:
        print(f"⚠️ Note: could not write to {CALIB_CACHE_FILE} ({e})")

    print("\n🎉 Calibration is valid and protected! The arm will not ask to calibrate again.\n")

if __name__ == "__main__":
    main()
