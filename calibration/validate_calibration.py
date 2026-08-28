#!/usr/bin/env python3
"""
validate_calibration.py — Live validation of ChArUco Hand-Eye Calibration.
Calculates and overlays real-time millimeter error between Forward Kinematics prediction
and live camera detection.
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import time
import yaml
import os
import math

try:
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
except ImportError:
    SOFollower = None

from calibrate_hand_eye import (
    D405Camera, get_fk_transform, get_pos, set_torque,
    estimate_pose_charuco, draw_frame_axes_compat
)

PORT = "/dev/arm_controller"
ARM_ID = "jetson_arm"

def validate_calibration():
    calib_path = os.path.join(os.path.dirname(__file__), "hand_eye_calibration.yaml")
    if not os.path.exists(calib_path):
        print("❌ No calibration found. Run calibrate_hand_eye.py first.")
        return
        
    with open(calib_path, 'r') as f:
        calib = yaml.safe_load(f)
        
    # R_cam2gripper is Rotation from Camera to Gripper (T_gripper_cam)
    # t_cam2gripper is Translation from Camera to Gripper in meters
    R_cam2gripper = np.array(calib['rotation_matrix'])
    t_cam2gripper = np.array(calib['translation_mm']) / 1000.0  # Convert mm to meters
    
    T_gripper_cam = np.eye(4)
    T_gripper_cam[:3, :3] = R_cam2gripper
    T_gripper_cam[:3, 3] = t_cam2gripper.flatten()

    # Load Board Params
    params_path = os.path.join(os.path.dirname(__file__), "charuco_board_params.yaml")
    if not os.path.exists(params_path):
        board_params = {"columns": 5, "rows": 7, "square_size_mm": 30.0, "marker_size_mm": 22.0}
    else:
        with open(params_path, 'r') as f:
            board_params = yaml.safe_load(f)
        
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(
        (board_params['columns'], board_params['rows']),
        board_params['square_size_mm'] / 1000.0,
        board_params['marker_size_mm'] / 1000.0,
        dictionary
    )
    charuco_detector = cv2.aruco.CharucoDetector(board)

    print("\n📷 Connecting to RealSense D405...")
    cam = D405Camera()
    
    print("🔌 Connecting to SO-ARM101 on", PORT, "...")
    robot = None
    torque_enabled = True
    if SOFollower is not None:
        try:
            config = SOFollowerRobotConfig(port=PORT, id=ARM_ID, use_degrees=True)
            robot = SOFollower(config)
            robot.connect()
            print("   ✅ Robot connected successfully!")
        except Exception as e:
            print(f"   ⚠️ Robot connect failed: {e}")

    T_base_target = None
    print("\n" + "═"*65)
    print(" HAND-EYE CALIBRATION VALIDATOR")
    print("═"*65)
    print(" Controls:")
    print("   [Space]  Lock the board position in the robot base frame")
    print("   [t]      Toggle arm motor TORQUE ON / OFF (for free hand moving)")
    print("   [q]      Quit")
    print("═"*65 + "\n")
    
    try:
        while True:
            color_img, depth_img = cam.get_frames()
            if color_img is None:
                continue
            
            display_img = color_img.copy()
            h, w = display_img.shape[:2]
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(color_img)
            
            T_cam_target_actual = None
            rvec_cam, tvec_cam = None, None
            if charuco_ids is not None and len(charuco_ids) >= 6:
                cv2.aruco.drawDetectedCornersCharuco(display_img, charuco_corners, charuco_ids, (0, 255, 0))
                success, rvec_cam, tvec_cam = estimate_pose_charuco(
                    charuco_corners, charuco_ids, board, cam.camera_matrix, cam.dist_coeffs
                )
                if success:
                    draw_frame_axes_compat(display_img, cam.camera_matrix, cam.dist_coeffs, rvec_cam, tvec_cam, 0.05)
                    
                    R_tc, _ = cv2.Rodrigues(rvec_cam)
                    T_cam_target_actual = np.eye(4)
                    T_cam_target_actual[:3, :3] = R_tc
                    T_cam_target_actual[:3, 3] = tvec_cam.flatten()

            key = cv2.waitKey(20) & 0xFF
            
            # Read live robot telemetry
            joints = get_pos(robot)
            T_base_gripper = get_fk_transform(joints)
            T_base_gripper_m = T_base_gripper.copy()
            T_base_gripper_m[:3, 3] /= 1000.0 # to meters

            x_mm = T_base_gripper[0, 3]
            y_mm = T_base_gripper[1, 3]
            z_mm = T_base_gripper[2, 3]

            if key == ord('t') and robot:
                new_state = not torque_enabled
                if set_torque(robot, new_state):
                    torque_enabled = new_state
                    print(f"🔧 Motor Torque {'ENABLED' if torque_enabled else 'DISABLED (Free-move mode)'}")

            if key == 32 and T_cam_target_actual is not None:  # Space key
                # Forward chain: T_base_target = T_base_gripper * T_gripper_cam * T_cam_target
                T_base_cam_0 = T_base_gripper_m @ T_gripper_cam
                T_base_target = T_base_cam_0 @ T_cam_target_actual
                print("🎯 Board reference position locked in base frame! Move arm to test accuracy.")
            
            # HUD Overlay Header
            cv2.rectangle(display_img, (0, 0), (w, 105), (20, 20, 20), -1)
            
            # Line 1: Live Coordinates & Torque
            torque_status = "TORQUE: ON" if torque_enabled else "TORQUE: OFF (Free-hand)"
            hud_line1 = f"Arm FK: X={x_mm:5.1f}mm Y={y_mm:5.1f}mm Z={z_mm:5.1f}mm | {torque_status} [t]"
            cv2.putText(display_img, hud_line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 255, 200), 1)

            # Line 2: Joints
            joint_str = (f"Pan:{joints.get('shoulder_pan.pos',0):4.1f}°  "
                         f"Lift:{joints.get('shoulder_lift.pos',0):4.1f}°  "
                         f"Elbow:{joints.get('elbow_flex.pos',0):4.1f}°  "
                         f"Wrist:{joints.get('wrist_flex.pos',0):4.1f}°")
            cv2.putText(display_img, joint_str, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

            # Line 3: Validation Error
            if T_base_target is not None:
                # Predict where the board should appear in current camera frame
                # T_cam_target_pred = (T_base_gripper_m * T_gripper_cam)^-1 * T_base_target
                T_base_cam_k = T_base_gripper_m @ T_gripper_cam
                T_cam_base_k = np.linalg.inv(T_base_cam_k)
                T_cam_target_pred = T_cam_base_k @ T_base_target
                
                rvec_pred, _ = cv2.Rodrigues(T_cam_target_pred[:3, :3])
                tvec_pred = T_cam_target_pred[:3, 3]
                
                # Project predicted origin onto camera image (Green Circle)
                imgpts, _ = cv2.projectPoints(np.float32([[0,0,0]]), rvec_pred, tvec_pred, cam.camera_matrix, cam.dist_coeffs)
                pt = tuple(np.int32(imgpts[0].ravel()))
                if 0 <= pt[0] < w and 0 <= pt[1] < h:
                    cv2.circle(display_img, pt, 8, (0, 255, 0), -1)  # Green = predicted
                
                if T_cam_target_actual is not None:
                    actual_pts, _ = cv2.projectPoints(np.float32([[0,0,0]]), rvec_cam, tvec_cam, cam.camera_matrix, cam.dist_coeffs)
                    actual_pt = tuple(np.int32(actual_pts[0].ravel()))
                    if 0 <= actual_pt[0] < w and 0 <= actual_pt[1] < h:
                        cv2.circle(display_img, actual_pt, 5, (0, 0, 255), -1)  # Red = actual camera vision
                    
                    dist_mm = np.linalg.norm(T_cam_target_pred[:3, 3] - T_cam_target_actual[:3, 3]) * 1000.0
                    err_color = (0, 255, 0) if dist_mm < 5.0 else ((0, 165, 255) if dist_mm < 10.0 else (0, 0, 255))
                    cv2.putText(display_img, f"3D Error: {dist_mm:5.2f} mm (Green=Predicted, Red=Actual)", 
                                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.62, err_color, 2)
                else:
                    cv2.putText(display_img, "Target Board not detected (keep board in camera FOV)", 
                                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 165, 255), 1)
            else:
                cv2.putText(display_img, "Press [SPACE] when board is visible to lock reference position", 
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
            
            cv2.imshow("Validate Hand-Eye Calibration", display_img)
            
            if key == ord('q'):
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        if robot:
            set_torque(robot, True)
            robot.disconnect()

if __name__ == "__main__":
    validate_calibration()
