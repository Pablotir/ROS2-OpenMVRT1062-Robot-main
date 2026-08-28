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

from calibrate_hand_eye import D405Camera, get_fk_transform

def validate_calibration():
    # Load Calibration
    calib_path = os.path.join(os.path.dirname(__file__), "hand_eye_calibration.yaml")
    if not os.path.exists(calib_path):
        print("No calibration found. Run calibrate_hand_eye.py first.")
        return
        
    with open(calib_path, 'r') as f:
        calib = yaml.safe_load(f)
        
    R_cg = np.array(calib['rotation_matrix'])
    t_cg = np.array(calib['translation_mm']) / 1000.0  # Convert to meters
    T_cam_gripper = np.eye(4)
    T_cam_gripper[:3, :3] = R_cg
    T_cam_gripper[:3, 3] = t_cg.flatten()
    T_gripper_cam = np.linalg.inv(T_cam_gripper)

    # Load Board Params
    params_path = os.path.join(os.path.dirname(__file__), "charuco_board_params.yaml")
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

    print("Connecting to camera...")
    cam = D405Camera()
    
    print("Connecting to arm...")
    robot = None
    if SOFollower is not None:
        config = SOFollowerRobotConfig()
        robot = SOFollower(config)
        robot.connect()

    T_base_target = None
    print("Press [Space] to lock the board position in the base frame.")
    print("Press [q] to quit.")
    
    try:
        while True:
            color_img, depth_img = cam.get_frames()
            if color_img is None: continue
            
            display_img = color_img.copy()
            
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(color_img)
            
            T_cam_target_actual = None
            if charuco_ids is not None and len(charuco_ids) >= 6:
                success, rvec_cam, tvec_cam = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, board, cam.camera_matrix, cam.dist_coeffs, None, None
                )
                if success:
                    cv2.drawFrameAxes(display_img, cam.camera_matrix, cam.dist_coeffs, rvec_cam, tvec_cam, 0.1)
                    
                    R_tc, _ = cv2.Rodrigues(rvec_cam)
                    T_cam_target_actual = np.eye(4)
                    T_cam_target_actual[:3, :3] = R_tc
                    T_cam_target_actual[:3, 3] = tvec_cam.flatten()

            key = cv2.waitKey(1) & 0xFF
            
            if robot:
                joints = robot.get_joint_positions()
                T_base_gripper = get_fk_transform(joints)
                T_base_gripper[:3, 3] /= 1000.0
                
                if key == 32 and T_cam_target_actual is not None:  # Space
                    T_base_cam = T_base_gripper @ T_gripper_cam
                    T_base_target = T_base_cam @ T_cam_target_actual
                    print("Board position locked in base frame!")
                
                if T_base_target is not None:
                    # Predict cam target
                    T_cam_base = np.linalg.inv(T_base_gripper @ T_gripper_cam)
                    T_cam_target_pred = T_cam_base @ T_base_target
                    
                    rvec_pred, _ = cv2.Rodrigues(T_cam_target_pred[:3, :3])
                    tvec_pred = T_cam_target_pred[:3, 3]
                    
                    # Project board origin
                    imgpts, _ = cv2.projectPoints(np.float32([[0,0,0]]), rvec_pred, tvec_pred, cam.camera_matrix, cam.dist_coeffs)
                    pt = tuple(np.int32(imgpts[0].ravel()))
                    cv2.circle(display_img, pt, 8, (0, 255, 0), -1)  # Green = predicted
                    
                    if T_cam_target_actual is not None:
                        actual_pts, _ = cv2.projectPoints(np.float32([[0,0,0]]), rvec_cam, tvec_cam, cam.camera_matrix, cam.dist_coeffs)
                        actual_pt = tuple(np.int32(actual_pts[0].ravel()))
                        cv2.circle(display_img, actual_pt, 5, (0, 0, 255), -1)  # Red = actual
                        
                        dist_px = np.linalg.norm(np.array(pt) - np.array(actual_pt))
                        dist_mm = np.linalg.norm(T_cam_target_pred[:3, 3] - T_cam_target_actual[:3, 3]) * 1000
                        cv2.putText(display_img, f"Error: {dist_mm:.1f} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Validate Calibration", display_img)
            
            if key == ord('q'):
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        if robot:
            robot.disconnect()

if __name__ == "__main__":
    validate_calibration()
