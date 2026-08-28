#!/usr/bin/env python3
"""
calibrate_hand_eye.py — SO-ARM101 + RealSense D405 ChArUco Hand-Eye Calibration
& Live Arm Position / Pose Inspector.

Cross-version compatible with OpenCV 4.5 through 4.10+.
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import time
import yaml
import os
import math
from datetime import datetime

PORT = "/dev/arm_controller"
ARM_ID = "jetson_arm"

try:
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
except ImportError:
    print("⚠️  LeRobot SOFollower not found. Will run camera-only / mock mode.")
    SOFollower = None

def get_pos(robot) -> dict:
    """Read joint positions in degrees from SO-ARM101."""
    if robot is None:
        return {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": -57.6,
            "elbow_flex.pos": -3.3,
            "wrist_flex.pos": 86.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 60.0
        }
    obs = robot.get_observation()
    joints = {"shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
              "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"}
    return {k: float(v) for k, v in obs.items() if k in joints}

def get_fk_transform(joints: dict) -> np.ndarray:
    """
    Computes Forward Kinematics to get T_gripper_base (4x4 matrix, coordinates in mm).
    """
    shoulder_pan = joints.get("shoulder_pan.pos", 0.0)
    shoulder_lift = joints.get("shoulder_lift.pos", 0.0)
    elbow_flex = joints.get("elbow_flex.pos", 0.0)
    wrist_flex = joints.get("wrist_flex.pos", 0.0)
    wrist_roll = joints.get("wrist_roll.pos", 0.0)

    L1 = 115.0 # mm (shoulder -> elbow)
    L2 = 137.5 # mm (elbow -> wrist)
    L3 = 90.0  # mm (wrist -> gripper tip)
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

    # Base frame coordinates (origin = shoulder pivot)
    x = gx * math.cos(pan_rad)
    y = gx * math.sin(pan_rad)
    z = gz # relative to shoulder

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

    roll_rad = math.radians(wrist_roll)
    R_roll = np.array([
        [1, 0, 0],
        [0, math.cos(roll_rad), -math.sin(roll_rad)],
        [0, math.sin(roll_rad),  math.cos(roll_rad)]
    ])
    
    R = R_pan @ R_pitch @ R_roll

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T

def estimate_pose_charuco(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs):
    """Robust Charuco board pose estimation working on OpenCV 4.5 through 4.10+."""
    if charuco_ids is None or len(charuco_ids) < 4:
        return False, None, None

    # Method 1: OpenCV 4.7+ / 4.8+ / 4.9+ modern API
    if hasattr(board, "matchImagePoints"):
        try:
            obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
            if obj_points is not None and len(obj_points) >= 4:
                success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
                return success, rvec, tvec
        except Exception:
            pass

    # Method 2: OpenCV <=4.6 legacy API
    if hasattr(cv2.aruco, "estimatePoseCharucoBoard"):
        try:
            return cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None
            )
        except Exception:
            pass

    # Method 3: Direct chessboard corner mapping fallback
    try:
        corners = board.getChessboardCorners() if hasattr(board, "getChessboardCorners") else getattr(board, "chessboardCorners", None)
        if corners is not None:
            obj_pts = np.array([corners[i[0]] for i in charuco_ids], dtype=np.float32)
            img_pts = np.array(charuco_corners, dtype=np.float32)
            if len(obj_pts) >= 4:
                success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)
                return success, rvec, tvec
    except Exception:
        pass

    return False, None, None

def draw_frame_axes_compat(image, camera_matrix, dist_coeffs, rvec, tvec, length=0.05):
    """Draw 3D coordinate axes across OpenCV versions."""
    if hasattr(cv2, "drawFrameAxes"):
        cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, length)
    elif hasattr(cv2.aruco, "drawAxis"):
        cv2.aruco.drawAxis(image, camera_matrix, dist_coeffs, rvec, tvec, length)

class D405Camera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 848, 480, rs.format.yuyv, 15)
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 15)
        
        try:
            self.profile = self.pipeline.start(self.config)
        except RuntimeError:
            print("Trying 640x480 YUYV...")
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

    def get_frames(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                return None, None
                
            raw = np.asanyarray(color_frame.get_data())
            h = color_frame.get_height()
            w = color_frame.get_width()
            
            try:
                yuyv = raw.view(np.uint8).reshape(h, w, 2)
                color_image = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
            except Exception:
                color_image = cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUYV)
                
            depth_image = np.asanyarray(depth_frame.get_data())
            return color_image, depth_image
        except Exception as e:
            return None, None
        
    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

def main():
    params_path = os.path.join(os.path.dirname(__file__), "charuco_board_params.yaml")
    if not os.path.exists(params_path):
        board_params = {"columns": 5, "rows": 7, "square_size_mm": 30.0, "marker_size_mm": 22.0}
        with open(params_path, "w") as f:
            yaml.dump(board_params, f)
    else:
        with open(params_path, "r") as f:
            board_params = yaml.safe_load(f)
        
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
    
    print("🔌 Connecting to SO-ARM101 on", PORT, "...")
    robot = None
    torque_enabled = True
    if SOFollower is not None:
        try:
            config = SOFollowerRobotConfig(port=PORT, id=ARM_ID, use_degrees=True)
            robot = SOFollower(config)
            robot.connect()
            print("   ✅ Robot arm connected successfully!")
        except Exception as e:
            print(f"   ⚠️  Could not connect to arm: {e}")
            print("   Running in camera-only / manual position logging mode.")
    
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []
    
    recorded_poses = {}
    pose_count = 0

    print("\n" + "═"*65)
    print(" SO-ARM101 CALIBRATION & POSITION INSPECTOR")
    print("═"*65)
    print(" Controls:")
    print("   [Enter]  Capture pose for ChArUco hand-eye calibration")
    print("   [s]      Save current position as named reference pose")
    print("   [t]      Toggle arm motor TORQUE ON / OFF (for free hand moving)")
    print("   [c]      Compute and save ChArUco Hand-Eye Calibration")
    print("   [q]      Quit")
    print("═"*65 + "\n")
    
    try:
        while True:
            color_img, depth_img = cam.get_frames()
            if color_img is None:
                time.sleep(0.01)
                continue
                
            joints = get_pos(robot)
            T_base_gripper = get_fk_transform(joints)
            
            x_mm = T_base_gripper[0, 3]
            y_mm = T_base_gripper[1, 3]
            z_mm = T_base_gripper[2, 3]
            rho_mm = math.sqrt(x_mm**2 + y_mm**2)

            display_img = color_img.copy()
            h, w = display_img.shape[:2]

            # Detect ChArUco
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(color_img)
            
            can_capture = False
            rvec_cam = None
            tvec_cam = None

            if charuco_ids is not None and len(charuco_ids) >= 6:
                cv2.aruco.drawDetectedCornersCharuco(display_img, charuco_corners, charuco_ids, (0, 255, 0))
                
                success, rvec_cam, tvec_cam = estimate_pose_charuco(
                    charuco_corners, charuco_ids, board, cam.camera_matrix, cam.dist_coeffs
                )
                if success:
                    draw_frame_axes_compat(display_img, cam.camera_matrix, cam.dist_coeffs, rvec_cam, tvec_cam, 0.05)
                    can_capture = True

            # HUD Overlay
            hud_bg = display_img[:140, :].copy()
            cv2.rectangle(display_img, (0, 0), (w, 135), (20, 20, 20), -1)
            cv2.addWeighted(hud_bg, 0.3, display_img[:140, :], 0.7, 0, display_img[:140, :])

            # Line 1: Live Coordinates
            coord_str = f"FK Pos: X={x_mm:6.1f}mm  Y={y_mm:6.1f}mm  Z={z_mm:6.1f}mm | Reach: {rho_mm:5.1f}mm"
            cv2.putText(display_img, coord_str, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

            # Line 2: Joint Angles
            joint_str = (f"Pan:{joints.get('shoulder_pan.pos',0):5.1f}°  "
                         f"Lift:{joints.get('shoulder_lift.pos',0):5.1f}°  "
                         f"Elbow:{joints.get('elbow_flex.pos',0):5.1f}°  "
                         f"Wrist:{joints.get('wrist_flex.pos',0):5.1f}°  "
                         f"Roll:{joints.get('wrist_roll.pos',0):5.1f}°")
            cv2.putText(display_img, joint_str, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 255, 200), 1)

            # Line 3: Calibration status & prompt
            status_color = (0, 255, 0) if can_capture else (0, 0, 255)
            status_str = f"Board: {'DETECTED (Ready to capture)' if can_capture else 'NOT DETECTED (Need >=6 corners)'} | Poses: {pose_count}/15"
            cv2.putText(display_img, status_str, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 1)

            torque_str = f"Torque: {'ON' if torque_enabled else 'OFF (Free-hand)'} [t] | Save Pose [s] | Calibrate [c] | Quit [q]"
            cv2.putText(display_img, torque_str, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 100), 1)

            cv2.imshow("SO-ARM101 ChArUco Calibration & Inspector", display_img)
            key = cv2.waitKey(20) & 0xFF
            
            if key == ord('q'):
                break

            elif key == ord('t') and robot:
                torque_enabled = not torque_enabled
                try:
                    val = 1 if torque_enabled else 0
                    robot.bus.write("Torque_Enable", [val]*6)
                    print(f"🔧 Motor Torque {'ENABLED' if torque_enabled else 'DISABLED (Free-move mode)'}")
                except Exception as e:
                    print(f"⚠️  Torque toggle error: {e}")

            elif key == ord('s'):
                pose_name = input("\nEnter name for this pose (e.g. scan_base, stow_base, floor_grab): ").strip()
                if not pose_name:
                    pose_name = f"pose_{len(recorded_poses)+1}"
                recorded_poses[pose_name] = {
                    "joints": joints,
                    "fk_xyz_mm": [float(x_mm), float(y_mm), float(z_mm)],
                    "reach_rho_mm": float(rho_mm),
                    "recorded_at": datetime.now().isoformat()
                }
                poses_file = os.path.join(os.path.dirname(__file__), "arm_reference_poses.yaml")
                with open(poses_file, "w") as f:
                    yaml.dump(recorded_poses, f, sort_keys=False)
                print(f"✅ Saved pose '{pose_name}' to {poses_file}")
                print(f"   Joints: {joints}")
                print(f"   FK: X={x_mm:.1f}mm, Y={y_mm:.1f}mm, Z={z_mm:.1f}mm, Reach={rho_mm:.1f}mm")

            elif key == 13 and can_capture:  # Enter key
                T_base_gripper_m = T_base_gripper.copy()
                T_base_gripper_m[:3, 3] /= 1000.0 # convert mm to meters for calibration math
                
                R_gb = T_base_gripper_m[:3, :3]
                t_gb = T_base_gripper_m[:3, 3]
                
                R_gripper2base.append(R_gb)
                t_gripper2base.append(t_gb)
                
                R_tc, _ = cv2.Rodrigues(rvec_cam)
                t_tc = tvec_cam.flatten()
                
                R_target2cam.append(R_tc)
                t_target2cam.append(t_tc)
                
                pose_count += 1
                print(f"📸 Captured calibration pose #{pose_count} (FK: X={x_mm:.1f}mm, Y={y_mm:.1f}mm, Z={z_mm:.1f}mm)")

            elif key == ord('c'):
                if pose_count < 5:
                    print(f"⚠️  Need at least 5 poses to compute calibration (currently have {pose_count}).")
                    continue
                print(f"\n⚙️  Computing Hand-Eye Calibration from {pose_count} poses using Tsai method...")
                
                R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                    R_gripper2base, t_gripper2base,
                    R_target2cam, t_target2cam,
                    method=cv2.CALIB_HAND_EYE_TSAI
                )
                
                out_path = os.path.join(os.path.dirname(__file__), "hand_eye_calibration.yaml")
                with open(out_path, 'w') as f:
                    yaml.dump({
                        "rotation_matrix": R_cam2gripper.tolist(),
                        "translation_mm": (t_cam2gripper.flatten() * 1000.0).tolist(),
                        "num_poses_used": pose_count,
                        "timestamp": datetime.now().isoformat(),
                        "method": "CALIB_HAND_EYE_TSAI"
                    }, f)
                    
                print(f"🎉 Hand-Eye Calibration SUCCESSFUL and saved to:\n   {out_path}")
                print(f"   Translation (mm): X={t_cam2gripper[0,0]*1000:.2f}, Y={t_cam2gripper[1,0]*1000:.2f}, Z={t_cam2gripper[2,0]*1000:.2f}")
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        if robot:
            robot.disconnect()

if __name__ == "__main__":
    main()
