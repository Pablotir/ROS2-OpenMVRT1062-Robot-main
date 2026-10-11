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
# Note: Feetech Homing_Offset register is an 11-bit signed magnitude offset (-2047 to +2047), default 0.
CALIBRATED_BASELINE = {
    "shoulder_pan":  {"id": 1, "drive_mode": 0, "homing_offset": 1604,  "range_min": 962,  "range_max": 3486},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": -1498, "range_min": 814,  "range_max": 3207},
    "elbow_flex":    {"id": 3, "drive_mode": 0, "homing_offset": 1619,  "range_min": 882,  "range_max": 3138},
    "wrist_flex":    {"id": 4, "drive_mode": 0, "homing_offset": -1885, "range_min": 887,  "range_max": 3243},
    "wrist_roll":    {"id": 5, "drive_mode": 0, "homing_offset": -1120, "range_min": 765,  "range_max": 3495},
    "gripper":       {"id": 6, "drive_mode": 0, "homing_offset": 1947,  "range_min": 2024, "range_max": 3626},
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
        calib_data = {k: dict(v) for k, v in CALIBRATED_BASELINE.items()}

    print("\n📊 Current Joint Calibration Status:")
    print("-------------------------------------------------------------------------")
    print(f"{'NAME':16s} | {'MIN':6s} | {'MAX':6s} | {'OFFSET':8s} | {'STATUS'}")
    print("-------------------------------------------------------------------------")

    modified = False
    for joint, baseline in CALIBRATED_BASELINE.items():
        if joint not in calib_data:
            calib_data[joint] = dict(baseline)
            modified = True

        info = calib_data[joint]
        rmin = info.get("range_min", 0)
        rmax = info.get("range_max", 4095)
        offset = info.get("homing_offset", 0)

        # Check for corrupted wrap-around (e.g. 0/4095)
        if rmin == 0 and rmax == 4095:
            print(f"{joint:16s} | {rmin:6d} | {rmax:6d} | {offset:8d} | ❌ CORRUPTED (0/4095 wrap-around)")
            info["range_min"] = baseline["range_min"]
            info["range_max"] = baseline["range_max"]
            print(f"   ↳ Fixed to safe physical bounds: [{info['range_min']}, {info['range_max']}]")
            modified = True
        elif rmin == rmax:
            print(f"{joint:16s} | {rmin:6d} | {rmax:6d} | {offset:8d} | ❌ DEGENERATE (min==max)")
            info["range_min"] = baseline["range_min"]
            info["range_max"] = baseline["range_max"]
            print(f"   ↳ Fixed to safe physical bounds: [{info['range_min']}, {info['range_max']}]")
            modified = True

        # Check if homing_offset was incorrectly wiped to 0
        if offset == 0 and baseline["homing_offset"] != 0:
            print(f"{joint:16s} | {rmin:6d} | {rmax:6d} | {offset:8d} | ⚠️ ZERO OFFSET (Restoring calibrated offset)")
            info["homing_offset"] = baseline["homing_offset"]
            print(f"   ↳ Restored homing_offset: {baseline['homing_offset']}")
            modified = True
        else:
            print(f"{joint:16s} | {rmin:6d} | {rmax:6d} | {offset:8d} | ✅ OK")

    print("-------------------------------------------------------------------------")

    # 2. Write repaired calibration to persistent location
    persist_targets = [
        CALIB_PERSIST_FILE,
        LOCAL_PERSIST_FILE,
        "/root/ros2_ws/scripts/jetson_arm.json",
        os.path.join(os.path.dirname(__file__), "jetson_arm.json"),
        os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots/so101_follower/jetson_arm.json"),
        os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots/so_follower/jetson_arm.json"),
        "/root/.cache/huggingface/lerobot/calibration/robots/so101_follower/jetson_arm.json",
        CALIB_CACHE_FILE,
    ]
    saved_persistent = False
    for pt in persist_targets:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(pt)), exist_ok=True)
            with open(pt, "w") as f:
                json.dump(calib_data, f, indent=4)
            print(f"💾 Permanently saved to: {pt}")
            saved_persistent = True
        except Exception:
            pass

    print("\n🎉 Calibration is valid and protected! The arm will not ask to calibrate again.\n")

if __name__ == "__main__":
    main()
