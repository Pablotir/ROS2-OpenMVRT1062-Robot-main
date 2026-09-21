#!/usr/bin/env python3
"""
3_charuco_calibration.py — Precision ChArUco Hand-Eye Calibration
=================================================================
Purpose:
  1. Detects the 5x7 ChArUco board using Intel RealSense D405.
  2. Connects to SO-ARM101 arm and disables motor torque so you can freely guide the arm.
  3. Interactive capture: Press [ENTER] to record poses when the board is visible.
  4. Solves eye-in-hand matrix T_cam_wrist using multi-algorithm OpenCV solver:
     (Daniilidis, Park, Horaud, Tsai, Andreff).
  5. Saves the verified calibration to:
     - calibration/hand_eye_calibration.yaml
     - /root/ros2_ws/calibration/hand_eye_calibration.yaml

Usage:
  python3 scripts/3_charuco_calibration.py
"""

import os
import sys
import time
import math
import yaml
import json
import numpy as np
import cv2
import pyrealsense2 as rs
from datetime import datetime

PORT = os.environ.get("ARM_PORT", "/dev/arm_controller")
ARM_ID = os.environ.get("ARM_ID", "jetson_arm")

IK_L1 = 115.0  # mm (shoulder -> elbow)
IK_L2 = 137.5  # mm (elbow -> wrist)
IK_L3 = 90.0   # mm (wrist -> gripper tip)
PAN_ZERO_OFFSET_DEG = -4.6

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


def get_pos(robot) -> dict:
    if robot is None:
        return {}
    try:
        obs = robot.get_observation()
        joints = {"shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                  "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"}
        return {k: float(v) for k, v in obs.items() if k in joints}
    except Exception:
        return {}


def set_torque(robot, enable: bool) -> bool:
    if robot is None or not hasattr(robot, "bus"):
        return False
    val = 1 if enable else 0
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll", "gripper"]
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
        robot.bus.write("Torque_Enable", [val] * len(motor_names), motor_names)
        return True
    except Exception:
        pass
    try:
        robot.bus.write("Torque_Enable", val, motor_names)
        return True
    except Exception:
        pass
    try:
        for m in motor_names:
            try:
                robot.bus.write("Torque_Enable", val, [m])
            except Exception:
                robot.bus.write("Torque_Enable", [m], val)
        return True
    except Exception:
        pass
    return False


def forward_kinematics(q: dict) -> np.ndarray:
    """Standard forward kinematics matching verified geometry."""
    pan  = math.radians(q.get("shoulder_pan.pos",  0.0) - PAN_ZERO_OFFSET_DEG)
    lift = q.get("shoulder_lift.pos", 0.0)
    elb  = q.get("elbow_flex.pos",    0.0)
    wst  = q.get("wrist_flex.pos",    0.0)

    t1 = math.radians(90.0 - lift)
    t2 = t1 - math.radians(elb + 81.0)
    t3 = t2 - math.radians(wst + 5.0)

    rho_w = IK_L1 * math.cos(t1) + IK_L2 * math.cos(t2)
    z_w   = IK_L1 * math.sin(t1) + IK_L2 * math.sin(t2)

    wx = rho_w * math.cos(pan)
    wy = rho_w * math.sin(pan)
    wz = z_w

    ax = math.cos(t3) * math.cos(pan)
    ay = math.cos(t3) * math.sin(pan)
    az = math.sin(t3)

    zx = -math.sin(pan)
    zy =  math.cos(pan)
    zz = 0.0

    yx = zy * az - zz * ay
    yy = zz * ax - zx * az
    yz = zx * ay - zy * ax

    T = np.array([
        [ax, yx, zx, wx],
        [ay, yy, zy, wy],
        [az, yz, zz, wz],
        [0., 0., 0., 1.],
    ])
    return T


class D405Camera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 848, 480, rs.format.yuyv, 15)
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 15)

        try:
            self.profile = self.pipeline.start(self.config)
        except RuntimeError:
            self.config.disable_all_streams()
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.yuyv, 15)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
            self.profile = self.pipeline.start(self.config)

        color_stream = self.profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array(intrinsics.coeffs)

    def get_frame(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            raw = np.asanyarray(color_frame.get_data())
            h, w = color_frame.get_height(), color_frame.get_width()
            yuyv = raw.view(np.uint8).reshape(h, w, 2)
            return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
        except Exception:
            return None

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


def estimate_pose_charuco(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs):
    if charuco_ids is None or len(charuco_ids) < 4:
        return False, None, None
    obj_points, img_points = None, None
    if hasattr(board, "matchImagePoints"):
        try:
            obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
        except Exception:
            pass
    if obj_points is None or len(obj_points) < 4:
        try:
            corners = board.getChessboardCorners() if hasattr(board, "getChessboardCorners") else getattr(board, "chessboardCorners", None)
            if corners is not None:
                obj_points = np.array([corners[i[0]] for i in charuco_ids], dtype=np.float32)
                img_points = np.array(charuco_corners, dtype=np.float32)
        except Exception:
            pass
    if obj_points is not None and len(obj_points) >= 4:
        try:
            flag = getattr(cv2, "SOLVEPNP_IPPE", cv2.SOLVEPNP_ITERATIVE)
            success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs, flags=flag)
            if success:
                return True, rvec, tvec
        except Exception:
            success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
            if success:
                return True, rvec, tvec
    return False, None, None


def evaluate_calibration_error(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, R_cam2gripper, t_cam2gripper):
    T_gc = np.eye(4)
    T_gc[:3, :3] = R_cam2gripper
    T_gc[:3, 3] = t_cam2gripper.flatten()

    target_positions_base = []
    for i in range(len(R_gripper2base)):
        T_bg = np.eye(4)
        T_bg[:3, :3] = R_gripper2base[i]
        T_bg[:3, 3] = t_gripper2base[i].flatten()

        T_ct = np.eye(4)
        T_ct[:3, :3] = R_target2cam[i]
        T_ct[:3, 3] = t_target2cam[i].flatten()

        T_bt = T_bg @ T_gc @ T_ct
        target_positions_base.append(T_bt[:3, 3])

    target_positions_base = np.array(target_positions_base)
    mean_target = np.mean(target_positions_base, axis=0)
    errors_mm = np.linalg.norm(target_positions_base - mean_target, axis=1) * 1000.0
    return float(np.mean(errors_mm)), errors_mm


def main():
    print("=" * 70)
    print("📐 CHARUCO EYE-IN-HAND CALIBRATION (SO-ARM101 + D405)")
    print("=" * 70)

    # Resolve board parameters
    script_dir = os.path.dirname(os.path.abspath(__file__))
    params_candidates = [
        os.path.join(script_dir, "..", "calibration", "charuco_board_params.yaml"),
        os.path.join(script_dir, "calibration", "charuco_board_params.yaml"),
        "calibration/charuco_board_params.yaml",
        "/root/ros2_ws/calibration/charuco_board_params.yaml",
    ]
    board_params = {"columns": 5, "rows": 7, "square_size_mm": 35.0, "marker_size_mm": 25.0}
    for p in params_candidates:
        if os.path.exists(p):
            with open(p, "r") as f:
                board_params = yaml.safe_load(f)
            print(f"Loaded board params from: {p}")
            break

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(
        (board_params["columns"], board_params["rows"]),
        board_params["square_size_mm"] / 1000.0,
        board_params["marker_size_mm"] / 1000.0,
        dictionary
    )
    charuco_detector = cv2.aruco.CharucoDetector(board)

    print("\n📷 Connecting to RealSense D405...")
    cam = D405Camera()

    print(f"🔌 Connecting to SO-ARM101 on {PORT}...")
    robot = None
    if SOFollower is not None and SOFollowerRobotConfig is not None:
        import builtins
        _orig_input = builtins.input
        builtins.input = lambda prompt="": ""
        try:
            config = SOFollowerRobotConfig(port=PORT, id=ARM_ID, use_degrees=True)
            robot = SOFollower(config)
            robot.connect()
            print("   ✅ Arm connected successfully!")
            print("   🔌 Disabling motor torque for manual positioning...")
            set_torque(robot, False)
        except Exception as e:
            print(f"   ⚠️ Could not connect to arm: {e}")
        finally:
            builtins.input = _orig_input

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []
    pose_count = 0

    print("\n" + "─" * 65)
    print(" INSTRUCTIONS:")
    print(" 1. Move the arm by hand so the D405 camera sees the ChArUco board.")
    print(" 2. When the green axes appear on the board, press [ENTER] to capture.")
    print(" 3. Capture 10 to 15 different viewpoints (various angles & distances).")
    print(" 4. Press [c] to compute calibration, [t] to toggle torque, [q] to quit.")
    print("─" * 65 + "\n")

    try:
        while True:
            color_img = cam.get_frame()
            if color_img is None:
                time.sleep(0.02)
                continue

            joints = get_pos(robot)
            T_base_wrist = forward_kinematics(joints)
            x_mm, y_mm, z_mm = T_base_wrist[0, 3], T_base_wrist[1, 3], T_base_wrist[2, 3]

            display = color_img.copy()
            charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(color_img)

            can_capture = False
            rvec_cam, tvec_cam = None, None

            if charuco_ids is not None and len(charuco_ids) >= 6:
                cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids, (0, 255, 0))
                success, rvec_cam, tvec_cam = estimate_pose_charuco(
                    charuco_corners, charuco_ids, board, cam.camera_matrix, cam.dist_coeffs
                )
                if success:
                    if hasattr(cv2, "drawFrameAxes"):
                        cv2.drawFrameAxes(display, cam.camera_matrix, cam.dist_coeffs, rvec_cam, tvec_cam, 0.05)
                    can_capture = True

            # HUD
            cv2.rectangle(display, (0, 0), (display.shape[1], 100), (20, 20, 20), -1)
            cv2.putText(display, f"Poses Captured: {pose_count} (Need 8-15) | [ENTER] Capture | [c] Solve | [q] Quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            status_color = (0, 255, 0) if can_capture else (0, 0, 255)
            status_text = "Board: DETECTED (Ready)" if can_capture else "Board: NOT VISIBLE"
            cv2.putText(display, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(display, f"Wrist Position: X={x_mm:+.1f}mm Y={y_mm:+.1f}mm Z={z_mm:+.1f}mm",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("ChArUco Hand-Eye Calibration", display)
            key = cv2.waitKey(20) & 0xFF

            if key in [ord('q'), 27]: # q or Esc
                break
            elif key == ord('t') and robot:
                # Toggle torque
                pass
            elif key == 13 and can_capture: # Enter
                T_base_wrist_m = T_base_wrist.copy()
                T_base_wrist_m[:3, 3] /= 1000.0  # to meters
                R_gb = T_base_wrist_m[:3, :3]
                t_gb = T_base_wrist_m[:3, 3]

                R_tc, _ = cv2.Rodrigues(rvec_cam)
                t_tc = tvec_cam.flatten()

                R_gripper2base.append(R_gb)
                t_gripper2base.append(t_gb)
                R_target2cam.append(R_tc)
                t_target2cam.append(t_tc)

                pose_count += 1
                dist_mm = np.linalg.norm(t_tc) * 1000.0
                print(f"📸 Pose #{pose_count:2d} recorded! (Cam distance: {dist_mm:.0f}mm | Wrist: X={x_mm:+.0f}, Y={y_mm:+.0f}, Z={z_mm:+.0f})")

            elif key == ord('c'):
                if pose_count < 5:
                    print(f"⚠️ Need at least 5 poses to solve (currently have {pose_count}). Capture more!")
                    continue

                print(f"\n⚙️ Solving Multi-Algorithm Hand-Eye Optimization from {pose_count} poses...")
                methods = {
                    "CALIB_HAND_EYE_TSAI": cv2.CALIB_HAND_EYE_TSAI,
                    "CALIB_HAND_EYE_PARK": cv2.CALIB_HAND_EYE_PARK,
                    "CALIB_HAND_EYE_HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
                    "CALIB_HAND_EYE_DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
                    "CALIB_HAND_EYE_ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
                }

                best_method = None
                best_error = float("inf")
                best_R = None
                best_t = None

                for name, flag in methods.items():
                    try:
                        R_c2g, t_c2g = cv2.calibrateHandEye(
                            R_gripper2base, t_gripper2base,
                            R_target2cam, t_target2cam,
                            method=flag
                        )
                        err, _ = evaluate_calibration_error(
                            R_gripper2base, t_gripper2base,
                            R_target2cam, t_target2cam,
                            R_c2g, t_c2g
                        )
                        print(f"   [{name:28s}] Mean error: {err:6.2f} mm")
                        if err < best_error:
                            best_error = err
                            best_method = name
                            best_R = R_c2g
                            best_t = t_c2g
                    except Exception as e:
                        print(f"   [{name:28s}] Failed: {e}")

                if best_R is not None:
                    trans_mm = (best_t.flatten() * 1000.0).tolist()
                    print("\n" + "=" * 65)
                    print(f"🏆 BEST METHOD: {best_method}")
                    print(f"   Mean 3D Consistency Error: {best_error:.2f} mm")
                    print(f"   Translation (mm): X={trans_mm[0]:+.1f}, Y={trans_mm[1]:+.1f}, Z={trans_mm[2]:+.1f}")
                    print("=" * 65)

                    out_data = {
                        "method": best_method,
                        "num_poses_used": pose_count,
                        "reprojection_error": float(best_error),
                        "rotation_matrix": best_R.tolist(),
                        "translation_mm": [[trans_mm[0]], [trans_mm[1]], [trans_mm[2]]],
                    }

                    save_targets = [
                        os.path.join(script_dir, "..", "calibration", "hand_eye_calibration.yaml"),
                        os.path.join(script_dir, "calibration", "hand_eye_calibration.yaml"),
                        "/root/ros2_ws/calibration/hand_eye_calibration.yaml",
                    ]
                    for st in save_targets:
                        try:
                            os.makedirs(os.path.dirname(os.path.abspath(st)), exist_ok=True)
                            with open(st, "w") as f:
                                yaml.dump(out_data, f, sort_keys=False)
                            print(f"   💾 Saved calibration to: {st}")
                        except Exception:
                            pass
                    print("=" * 65)
                    break

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user.")
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        if robot:
            robot.disconnect()
        print("Done.\n")


if __name__ == "__main__":
    main()
