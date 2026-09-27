#!/usr/bin/env python3
"""
calibrate_hand_eye.py — High-Precision ChArUco Hand-Eye Calibration for SO-ARM101 + D405.

Calculates the exact 4x4 spatial transformation matrix (T_cam_wrist) between the
wrist-mounted RealSense D405 camera and the robot arm's end-effector.

Workflow:
1. Connects to the SO-ARM101 using calibrated zero offsets and servo health checks.
2. Starts the RealSense D405 at maximum hardware framerate (30/60/90 FPS).
3. Opens the live video window directly on the Jetson desktop.
4. Press 't' to cut motor torque and enter Free-Move Mode.
5. Move the arm to 10–15 different poses (different angles, tilts, and heights)
   where the 5x7 ChArUco board is visible with green coordinate axes.
6. Press [Enter] to capture each valid pose.
7. Press 'c' to solve with 5 distinct Hand-Eye algorithms (Tsai, Park, Horaud,
   Daniilidis, Andreff), pick the one with lowest 3D error, and automatically
   save hand_eye_calibration.yaml!
"""

import os
import sys
import time
import math
import yaml
import json
import pathlib
import select
from datetime import datetime
import cv2
import numpy as np
import pyrealsense2 as rs

# Import arm helpers, kinematics, camera stream, and display manager from arm_picker
from arm_picker import (
    connect_robot, get_pos, _set_torque, forward_kinematics,
    RealSenseStream, _init_display_mode, _show_frame, _destroy_windows
)

BOARD_COLS = 5
BOARD_ROWS = 7
SQUARE_SIZE_MM = 35.0
MARKER_SIZE_MM = 25.0


def setup_charuco():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(
        (BOARD_COLS, BOARD_ROWS),
        SQUARE_SIZE_MM / 1000.0,
        MARKER_SIZE_MM / 1000.0,
        dictionary
    )
    try:
        detector = cv2.aruco.CharucoDetector(board)
    except AttributeError:
        detector = None
    return board, dictionary, detector


def estimate_pose_charuco(corners, ids, board, K, dist_coeffs):
    if ids is None or len(ids) < 4:
        return False, None, None

    obj_pts, img_pts = None, None
    if hasattr(board, "matchImagePoints"):
        try:
            obj_pts, img_pts = board.matchImagePoints(corners, ids)
        except Exception:
            pass

    if obj_pts is None or len(obj_pts) < 4:
        try:
            all_corners = board.getChessboardCorners() if hasattr(board, "getChessboardCorners") else getattr(board, "chessboardCorners", None)
            if all_corners is not None:
                obj_pts = np.array([all_corners[i[0]] for i in ids], dtype=np.float32)
                img_pts = np.array(corners, dtype=np.float32)
        except Exception:
            pass

    if obj_pts is not None and len(obj_pts) >= 4:
        try:
            flag = getattr(cv2, "SOLVEPNP_IPPE", cv2.SOLVEPNP_ITERATIVE)
            success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_coeffs, flags=flag)
            if success:
                return True, rvec, tvec
        except Exception:
            try:
                success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_coeffs)
                if success:
                    return True, rvec, tvec
            except Exception:
                pass

    return False, None, None


def evaluate_calibration_error(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, R_cam2gripper, t_cam2gripper):
    """
    Computes consistency of the board position in the robot base frame across all poses:
        P_base = T_base_gripper @ T_gripper_cam @ P_target_cam
    A perfect calibration produces identical base coordinates for the stationary board.
    """
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


def check_terminal_input():
    """Non-blocking check for keyboard input from the terminal."""
    if sys.platform != "win32":
        try:
            dr, _, _ = select.select([sys.stdin], [], [], 0)
            if dr:
                line = sys.stdin.readline().strip()
                if line == "":
                    return 13  # Enter key
                elif line.lower() == "t":
                    return ord('t')
                elif line.lower() == "c":
                    return ord('c')
                elif line.lower() == "q":
                    return ord('q')
        except Exception:
            pass
    return None


def main():
    print("\n" + "═"*65)
    print(" 🏁 HIGH-PRECISION CHARUCO HAND-EYE CALIBRATION")
    print("═"*65)
    print(f" Board specs: {BOARD_COLS}x{BOARD_ROWS} grid | Square: {SQUARE_SIZE_MM}mm | Marker: {MARKER_SIZE_MM}mm")
    print(" Controls (usable in popup window OR terminal):")
    print("   [t]      Toggle motor TORQUE ON / OFF (Free-Move Mode)")
    print("   [Enter]  Capture pose for calibration (when green axes visible)")
    print("   [c]      Compute Multi-Algorithm Calibration")
    print("   [q]      Quit")
    print("═"*65 + "\n")

    _init_display_mode()
    board, dictionary, detector = setup_charuco()

    robot = None
    cap = None
    torque_enabled = True

    try:
        robot = connect_robot()

        print("\n📷 Starting RealSense D405 (uncapped / max hardware FPS)...")
        cap = RealSenseStream(width=848, height=480, fps=0)
        time.sleep(1.0)

        # Build camera intrinsics matrix
        intrinsics = getattr(cap, "_intrinsics", None)
        if intrinsics is not None:
            K = np.array([
                [intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64)
        else:
            K = np.array([[600.0, 0, 424.0], [0, 600.0, 240.0], [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.zeros(5, dtype=np.float64)

        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []
        pose_count = 0

        _last_fps_t = time.time()
        _fps_smooth = 60.0

        while True:
            color_img, _, _ = cap.read(wait_new=True, timeout=0.05)
            if color_img is None:
                continue

            display_img = color_img.copy()
            h, w = display_img.shape[:2]

            # ── Live FPS Calculation ──────────────────────────────────────────
            t_now = time.time()
            dt = t_now - _last_fps_t
            _last_fps_t = t_now
            if dt > 0:
                _fps_smooth = 0.9 * _fps_smooth + 0.1 * (1.0 / dt)

            # Get current arm forward kinematics
            joints = get_pos(robot)
            T_base_wrist = forward_kinematics(joints)
            x_mm = T_base_wrist[0, 3]
            y_mm = T_base_wrist[1, 3]
            z_mm = T_base_wrist[2, 3]

            # Detect ChArUco
            corners, ids = None, None
            if detector is not None:
                charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(color_img)
                corners, ids = charuco_corners, charuco_ids
            else:
                m_corners, m_ids, _ = cv2.aruco.detectMarkers(color_img, dictionary)
                if m_ids is not None and len(m_ids) > 0:
                    _, corners, ids = cv2.aruco.interpolateCornersCharuco(m_corners, m_ids, color_img, board)

            can_capture = False
            rvec_cam, tvec_cam = None, None

            if ids is not None and len(ids) >= 6:
                cv2.aruco.drawDetectedCornersCharuco(display_img, corners, ids, (0, 255, 0))
                success, rvec_cam, tvec_cam = estimate_pose_charuco(corners, ids, board, K, dist_coeffs)
                if success:
                    try:
                        cv2.drawFrameAxes(display_img, K, dist_coeffs, rvec_cam, tvec_cam, 0.05)
                    except Exception:
                        pass
                    can_capture = True

            # ── HUD Overlay ───────────────────────────────────────────────────
            cv2.rectangle(display_img, (0, 0), (w, 130), (20, 20, 20), -1)

            cv2.putText(display_img, f"FPS: {_fps_smooth:.1f} | Poses Captured: {pose_count} (Goal: 10-15)",
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            coord_str = f"Wrist FK: X={x_mm:5.1f} Y={y_mm:5.1f} Z={z_mm:5.1f} mm"
            cv2.putText(display_img, coord_str, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

            status_color = (0, 255, 0) if can_capture else (0, 0, 255)
            status_str = f"Board: {'DETECTED (Ready to capture)' if can_capture else 'SEARCHING (Align board in view)'}"
            cv2.putText(display_img, status_str, (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

            torque_str = f"Torque: {'ON' if torque_enabled else 'OFF (Free-move)'} [t] | Capture [Enter] | Calibrate [c] | Quit [q]"
            cv2.putText(display_img, torque_str, (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 200, 100), 1)

            _show_frame("SO-ARM101 ChArUco Calibration", display_img)

            # Check keys from both OpenCV window and Terminal
            gui_key = cv2.waitKey(1) & 0xFF
            term_key = check_terminal_input()
            key = term_key if term_key is not None else gui_key

            if key == ord('q'):
                print("\n⏹️  Exiting calibration.")
                break

            elif key == ord('t'):
                torque_enabled = not torque_enabled
                _set_torque(robot, torque_enabled)
                print(f"\n🔧 Motor Torque {'ENABLED' if torque_enabled else 'DISABLED (Free-move mode active: move arm by hand!)'}")

            elif key in [13, 10, ord(' ')] and can_capture:  # Enter or Space
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
                cam_dist_mm = float(np.linalg.norm(t_tc) * 1000.0)
                print(f"📸 Captured pose #{pose_count:2d} | Wrist: X={x_mm:5.1f} Y={y_mm:5.1f} Z={z_mm:5.1f}mm | Cam Dist: {cam_dist_mm:4.0f}mm")

                if pose_count < 10:
                    print(f"   👉 Move arm to another angle/tilt and capture again ({10 - pose_count} more recommended)")
                else:
                    print(f"   ✅ {pose_count} poses captured! You can capture more or press 'c' to compute calibration.")

            elif key == ord('c'):
                if pose_count < 5:
                    print(f"\n⚠️  Need at least 5 captured poses (currently have {pose_count}). Move arm and press Enter to capture.")
                    continue

                print("\n" + "═"*65)
                print(f"⚙️  Solving Multi-Algorithm Hand-Eye Optimization from {pose_count} poses...")
                print("═"*65)

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
                        R_cg, t_cg = cv2.calibrateHandEye(
                            R_gripper2base, t_gripper2base,
                            R_target2cam, t_target2cam,
                            method=flag
                        )
                        mean_err, _ = evaluate_calibration_error(
                            R_gripper2base, t_gripper2base,
                            R_target2cam, t_target2cam,
                            R_cg, t_cg
                        )
                        print(f"   • {name:28s}: Mean 3D Error = {mean_err:5.2f} mm")
                        if mean_err < best_error:
                            best_error = mean_err
                            best_method = name
                            best_R = R_cg
                            best_t = t_cg
                    except Exception as e:
                        print(f"   • {name:28s}: Failed ({e})")

                print("\n" + "═"*65)
                print(f"🏆 BEST METHOD: {best_method} with Mean Error = {best_error:.2f} mm")
                print("═"*65)

                calib_payload = {
                    "rotation_matrix": best_R.tolist(),
                    "translation_mm": (best_t.flatten() * 1000.0).tolist(),
                    "num_poses_used": pose_count,
                    "mean_error_mm": float(best_error),
                    "reprojection_error": float(best_error),
                    "timestamp": datetime.now().isoformat(),
                    "method": best_method
                }

                # Save to multiple search paths so all scripts find it automatically
                save_paths = [
                    pathlib.Path("hand_eye_calibration.yaml"),
                    pathlib.Path("/root/ros2_ws/calibration/hand_eye_calibration.yaml"),
                    pathlib.Path(__file__).parent / "hand_eye_calibration.yaml",
                    pathlib.Path(__file__).parent.parent / "calibration" / "hand_eye_calibration.yaml",
                ]

                saved_any = False
                for sp in save_paths:
                    try:
                        sp.parent.mkdir(parents=True, exist_ok=True)
                        with open(sp, "w") as f:
                            yaml.dump(calib_payload, f, sort_keys=False)
                        print(f"   💾 Saved calibration to: {sp}")
                        saved_any = True
                    except Exception:
                        pass

                t_mm = best_t.flatten() * 1000.0
                print(f"\n✅ Calibrated Camera-to-Wrist Translation (mm): X={t_mm[0]:.1f}, Y={t_mm[1]:.1f}, Z={t_mm[2]:.1f}")
                print("🎉 Hand-Eye Calibration Complete! Now ready to grab objects with arm_picker.py.\n")
                break

    except KeyboardInterrupt:
        print("\n⏹️  Stopping calibration...")
    finally:
        if cap is not None:
            try:
                cap.stop()
            except Exception:
                pass
        if robot is not None:
            try:
                _set_torque(robot, True)
                robot.disconnect()
            except Exception:
                pass
        try:
            _destroy_windows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
