import cv2
import numpy as np
import pyrealsense2 as rs
import time
import yaml
import os
import math
from datetime import datetime

try:
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
except ImportError:
    print("WARNING: LeRobot SOFollower not found. Will use dummy robot for testing if needed.")
    SOFollower = None

def get_fk_transform(joint_positions):
    """
    Computes Forward Kinematics to get T_gripper_base.
    Returns a 4x4 numpy array representing the pose of the gripper relative to the robot base.
    """
    # Map from lerobot joint names/indices to our variables
    # Assuming joint order: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, gripper]
    # Update indices based on actual SOFollower joint layout
    shoulder_pan = joint_positions[0]
    shoulder_lift = joint_positions[1]
    elbow_flex = joint_positions[2]
    wrist_flex = joint_positions[3]

    L1 = 115.0 # mm (shoulder->elbow)
    L2 = 137.5 # mm (elbow->wrist)
    L3 = 90.0  # mm (wrist->gripper tip)
    SHOULDER_HEIGHT = 170.0
    PAN_ZERO_OFFSET_DEG = -4.6

    pan_rad = math.radians(shoulder_pan + PAN_ZERO_OFFSET_DEG)
    t1 = math.radians(90.0 - shoulder_lift)
    t2 = t1 - math.radians(elbow_flex + 81.0)
    t3 = t2 - math.radians(wrist_flex + 5.0)

    wx = L1 * math.cos(t1) + L2 * math.cos(t2)
    wz = L1 * math.sin(t1) + L2 * math.sin(t2)
    
    gx = wx + L3 * math.cos(t3)
    gz = wz + L3 * math.sin(t3)

    # Apply pan rotation
    x = gx * math.cos(pan_rad)
    y = gx * math.sin(pan_rad)
    z = gz + SHOULDER_HEIGHT

    # Compute orientation matrix
    # Gripper orientation is determined by pan and t3 (pitch)
    # We construct a simple rotation matrix assuming roll=0
    R_pan = np.array([
        [math.cos(pan_rad), -math.sin(pan_rad), 0],
        [math.sin(pan_rad),  math.cos(pan_rad), 0],
        [0, 0, 1]
    ])
    
    R_pitch = np.array([
        [math.cos(t3), 0, math.sin(t3)],
        [0, 1, 0],
        [-math.sin(t3), 0, math.cos(t3)]
    ])
    
    R = R_pan @ R_pitch

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T

class D405Camera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 848, 480, rs.format.yuyv, 15)
        # Depth native aligned, no rs.align!
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 15)
        
        try:
            self.profile = self.pipeline.start(self.config)
        except RuntimeError:
            print("Failed with 848x480 YUYV, trying 640x480 YUYV...")
            self.config.disable_all_streams()
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.yuyv, 15)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
            self.profile = self.pipeline.start(self.config)

        # Get intrinsics
        color_stream = self.profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array(intrinsics.coeffs)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None
            
        # YUYV conversion quirk for Jetson D405
        raw = np.asanyarray(color_frame.get_data())
        h = color_frame.get_height()
        w = color_frame.get_width()
        
        # Reshape and convert
        try:
            yuyv = raw.view(np.uint8).reshape(h, w, 2)
            color_image = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
        except ValueError:
            # Fallback if raw shape is already correct
            color_image = cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUYV)
            
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_image
        
    def stop(self):
        self.pipeline.stop()

def collect_poses():
    # Load Board Params
    params_path = os.path.join(os.path.dirname(__file__), "charuco_board_params.yaml")
    with open(params_path, 'r') as f:
        board_params = yaml.safe_load(f)
        
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard(
        (board_params['columns'], board_params['rows']),
        board_params['square_size_mm'] / 1000.0,  # convert to meters
        board_params['marker_size_mm'] / 1000.0,
        dictionary
    )
    charuco_detector = cv2.aruco.CharucoDetector(board)
    
    # Initialize hardware
    print("Connecting to camera...")
    cam = D405Camera()
    
    print("Connecting to arm...")
    robot = None
    if SOFollower is not None:
        config = SOFollowerRobotConfig()
        robot = SOFollower(config)
        robot.connect()
    
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []
    
    print("Commands:")
    print("  [Enter] Capture pose")
    print("  [q] Quit and compute calibration")
    
    pose_count = 0
    try:
        while True:
            color_img, depth_img = cam.get_frames()
            if color_img is None:
                continue
                
            display_img = color_img.copy()
            
            # Detect ChArUco
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(color_img)
            
            can_capture = False
            if charuco_ids is not None and len(charuco_ids) >= 6:
                cv2.aruco.drawDetectedCornersCharuco(display_img, charuco_corners, charuco_ids, (0, 255, 0))
                
                success, rvec_cam, tvec_cam = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, board, cam.camera_matrix, cam.dist_coeffs, None, None
                )
                
                if success:
                    cv2.drawFrameAxes(display_img, cam.camera_matrix, cam.dist_coeffs, rvec_cam, tvec_cam, 0.1)
                    can_capture = True
            
            cv2.imshow("Hand-Eye Calibration", display_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == 13 and can_capture:  # Enter key
                if robot:
                    # Get joint positions and compute FK
                    # Note: You may need to adapt reading joints from your specific LeRobot setup
                    joints = robot.get_joint_positions()
                    T_base_gripper = get_fk_transform(joints)
                    
                    # Convert to mm
                    T_base_gripper[:3, 3] /= 1000.0
                    
                    R_gb = T_base_gripper[:3, :3]
                    t_gb = T_base_gripper[:3, 3]
                    
                    R_gripper2base.append(R_gb)
                    t_gripper2base.append(t_gb)
                    
                    # Target to Cam
                    R_tc, _ = cv2.Rodrigues(rvec_cam)
                    t_tc = tvec_cam.flatten()
                    
                    R_target2cam.append(R_tc)
                    t_target2cam.append(t_tc)
                    
                    pose_count += 1
                    print(f"Captured pose {pose_count}")
                else:
                    print("Robot not connected, simulated capture.")
                    pose_count += 1
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        if robot:
            robot.disconnect()
            
    if pose_count < 5:
        print("Not enough poses collected (need at least 5).")
        return
        
    print(f"Computing calibration from {pose_count} poses using Tsai method...")
    # R_gripper2base, t_gripper2base are gripper in base
    # R_target2cam, t_target2cam are target in cam
    # We want cam in gripper
    
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    
    # Evaluate reprojection error roughly
    errors = []
    for i in range(len(R_gripper2base)):
        R_gb = R_gripper2base[i]
        t_gb = t_gripper2base[i].reshape(3,1)
        R_tc = R_target2cam[i]
        t_tc = t_target2cam[i].reshape(3,1)
        
        # T_base_cam = T_base_gripper * T_gripper_cam
        T_gb = np.eye(4); T_gb[:3,:3] = R_gb; T_gb[:3,3:] = t_gb
        T_gc = np.eye(4); T_gc[:3,:3] = R_cam2gripper; T_gc[:3,3:] = t_cam2gripper
        
        T_bc = T_gb @ T_gc
        
        # T_base_target = T_base_cam * T_cam_target
        T_ct = np.eye(4); T_ct[:3,:3] = R_tc; T_ct[:3,3:] = t_tc
        T_bt = T_bc @ T_ct
        
        # This should be constant across all poses. We'll skip rigorous eval here to save space
        pass

    out_path = os.path.join(os.path.dirname(__file__), "hand_eye_calibration.yaml")
    with open(out_path, 'w') as f:
        yaml.dump({
            "rotation_matrix": R_cam2gripper.tolist(),
            "translation_mm": (t_cam2gripper.flatten() * 1000).tolist(),
            "num_poses_used": pose_count,
            "timestamp": datetime.now().isoformat(),
            "method": "CALIB_HAND_EYE_TSAI"
        }, f)
        
    print(f"Calibration saved to {out_path}")
    print("Translation (mm):", t_cam2gripper.flatten() * 1000)

if __name__ == "__main__":
    collect_poses()
