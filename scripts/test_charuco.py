#!/usr/bin/env python3
"""
test_charuco.py — Live ChArUco Board & Hand-Eye Calibration Verifier

Tests if the RealSense D405 can see your 5x7 ChArUco board, and verifies that
the hand-eye calibration matrix correctly computes the 3D position of the board
in the robot's base coordinate frame.

Workflow:
1. Connects to RealSense D405 and SO-ARM101.
2. Loads hand_eye_calibration.yaml.
3. Moves arm smoothly to START_POS (scan posture).
4. Continuously detects the ChArUco board:
   - Shows detected corner count and camera distance.
   - Projects board origin into the Arm Base Frame (X, Y, Z in mm).
   - Saves live visualization to /tmp/charuco_live.jpg.
5. You can press 't' in terminal to disable torque and move the arm by hand:
   If calibration is correct, the reported 3D board coordinates in the base
   frame will remain nearly CONSTANT even as the arm moves!
"""

import os
import sys
import time
import math
import yaml
import json
import pathlib
import cv2
import numpy as np
import pyrealsense2 as rs

# Import arm helpers from arm_picker
from arm_picker import (
    connect_robot, get_pos, smooth_move, _set_torque,
    forward_kinematics, _build_T_cam_wrist, _load_reference_poses,
    _BASE, _STOW_BASE, HEADLESS, RealSenseStream, _show_frame, _destroy_windows,
    _init_display_mode
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
    # Compatible detector setup
    try:
        detector = cv2.aruco.CharucoDetector(board)
    except AttributeError:
        detector = None
    return board, dictionary, detector


def main():
    print("\n" + "═"*65)
    print(" 🏁 CHARUCO BOARD DETECTION & HAND-EYE CALIBRATION TEST")
    print("═"*65)

    _load_reference_poses()
    START_POS = dict(_BASE)
    STOW = dict(_STOW_BASE)

    T_cam_wrist = _build_T_cam_wrist()
    print("📐 Loaded T_cam_wrist matrix:")
    print(np.array2string(T_cam_wrist, precision=3, suppress_small=True))

    _init_display_mode()

    board, dictionary, detector = setup_charuco()

    print("\n📷 Starting RealSense D405...")
    cap = RealSenseStream(width=848, height=480, fps=15)
    time.sleep(1.5)

    robot = connect_robot()

    # Move to start position smoothly
    print("\n▶ Moving to Scan Position...")
    smooth_move(robot, START_POS, step_size=1.0, step_delay=0.03)
    time.sleep(0.5)

    print("\n" + "─"*65)
    print(" 🎯 POINT CAMERA AT YOUR 5x7 CHARUCO BOARD")
    print(" 🖥️  Live feed displayed directly on Jetson monitor!")
    print(" Press Ctrl+C in terminal to finish and stow arm.")
    print("─"*65 + "\n")

    last_print = 0.0
    frame_count = 0
    _last_fps_t = time.time()
    _fps_smooth = 15.0

    try:
        while True:
            color, has_depth, depth_colormap = cap.read()
            if color is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            display = color.copy()
            h, w = color.shape[:2]

            # ── Live FPS calculation and overlay ──────────────────────────────
            t_now = time.time()
            dt = t_now - _last_fps_t
            _last_fps_t = t_now
            if dt > 0:
                _fps_smooth = 0.9 * _fps_smooth + 0.1 * (1.0 / dt)
            cv2.putText(display, f"FPS: {_fps_smooth:.1f}", (w - 140, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            # Detect ChArUco
            corners, ids = None, None
            if detector is not None:
                charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(color)
                corners, ids = charuco_corners, charuco_ids
            else:
                # OpenCV legacy aruco fallback
                m_corners, m_ids, _ = cv2.aruco.detectMarkers(color, dictionary)
                if m_ids is not None and len(m_ids) > 0:
                    cv2.aruco.drawDetectedMarkers(display, m_corners, m_ids)
                    _, corners, ids = cv2.aruco.interpolateCornersCharuco(
                        m_corners, m_ids, color, board
                    )

            board_detected = False
            cam_dist_mm = 0.0
            p_base_mm = None

            if ids is not None and len(ids) >= 6:
                board_detected = True
                cv2.aruco.drawDetectedCornersCharuco(display, corners, ids, (0, 255, 0))

                # Camera matrix from intrinsics if available, else approximate
                intrinsics = getattr(cap, "intrinsics", None)
                if intrinsics is not None:
                    K = np.array([
                        [intrinsics.fx, 0, intrinsics.ppx],
                        [0, intrinsics.fy, intrinsics.ppy],
                        [0, 0, 1]
                    ], dtype=np.float64)
                    dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64)
                else:
                    K = np.array([[600.0, 0, w/2], [0, 600.0, h/2], [0, 0, 1]], dtype=np.float64)
                    dist_coeffs = np.zeros(5, dtype=np.float64)

                # Solve PnP for board pose
                try:
                    obj_pts, img_pts = board.matchImagePoints(corners, ids)
                    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
                except Exception:
                    try:
                        success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(corners, ids, board, K, dist_coeffs, None, None)
                    except Exception:
                        success = False

                if success:
                    # Draw axes on board origin (50mm length)
                    try:
                        cv2.drawFrameAxes(display, K, dist_coeffs, rvec, tvec, 0.05)
                    except Exception:
                        pass

                    # tvec is in meters in camera frame: convert to mm
                    tvec_mm = tvec.flatten() * 1000.0
                    cam_dist_mm = float(np.linalg.norm(tvec_mm))

                    # Homogeneous coordinate of board origin in camera frame
                    p_cam = np.array([tvec_mm[0], tvec_mm[1], tvec_mm[2], 1.0], dtype=np.float64)

                    # Chain: P_base = T_base_wrist @ T_cam_wrist @ P_cam
                    cur_q = get_pos(robot)
                    T_base_wrist = forward_kinematics(cur_q)
                    p_wrist = T_cam_wrist @ p_cam
                    p_base = T_base_wrist @ p_wrist
                    p_base_mm = p_base[:3]

                    # Overlay on image
                    cv2.putText(display, f"Board Found! ({len(ids)} corners)", (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display, f"Cam Dist: {cam_dist_mm:.0f} mm", (20, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(display, f"Base XYZ: X={p_base_mm[0]:.1f} Y={p_base_mm[1]:.1f} Z={p_base_mm[2]:.1f} mm",
                                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            else:
                cv2.putText(display, "Searching for 5x7 ChArUco Board...", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show live popup window and feed web stream
            _show_frame("ChArUco Hand-Eye Verification", display)


            # Throttle terminal logging to ~1 Hz
            now = time.time()
            if now - last_print >= 1.0:
                last_print = now
                if board_detected and p_base_mm is not None:
                    print(f"✅ Board Detected! ({len(ids):2d} corners) | "
                          f"Cam Dist: {cam_dist_mm:4.0f}mm | "
                          f"Robot Base Frame: X={p_base_mm[0]:+6.1f}mm  Y={p_base_mm[1]:+6.1f}mm  Z={p_base_mm[2]:+6.1f}mm")
                else:
                    corners_seen = len(ids) if ids is not None else 0
                    print(f"⏳ Searching... (saw {corners_seen}/6 minimum corners). Ensure board is in camera view.")

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping test...")
    finally:
        print("▶ Stowing arm smoothly...")
        try:
            smooth_move(robot, STOW, step_size=0.8, step_delay=0.025)
            print("   ✅ Arm stowed.")
        except Exception as e:
            print(f"   ⚠️ Stow failed: {e}")
        try:
            robot.disconnect()
            cap.stop()
            _destroy_windows()
        except Exception:
            pass
        print("═"*65 + "\n")



if __name__ == "__main__":
    main()
