#!/usr/bin/env python3
"""
calibrate_hand_eye.py — High-Precision ChArUco Hand-Eye Calibration for SO-ARM101 + D405.
Uses the exact geometric Forward Kinematics from arm_picker.py with IPPE planar PnP
and multi-algorithm optimization.
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

IK_L1 = 115.0 # mm (shoulder -> elbow)
IK_L2 = 137.5 # mm (elbow -> wrist)
IK_L3 = 90.0  # mm (wrist -> gripper tip)
SHOULDER_HEIGHT = 170.0
PAN_ZERO_OFFSET_DEG = -4.6

try:
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
except ImportError:
    SOFollower = None

def get_pos(robot) -> dict:
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
        robot.bus.write("Torque_Enable", val, motor_names)
        return True
    except Exception:
        pass
    try:
        robot.bus.write("Torque_Enable", [val] * len(motor_names), motor_names)
        return True
    except Exception:
        pass
    try:
        robot.bus.write("Torque_Enable", motor_names, val)
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
    pan  = math.radians(-q.get("shoulder_pan.pos", 0.0) - PAN_ZERO_OFFSET_DEG)
    lift = q.get("shoulder_lift.pos", 0.0)
    elb  = q.get("elbow_flex.pos",    0.0)
    wst  = q.get("wrist_flex.pos",    0.0)
    roll = math.radians(-q.get("wrist_roll.pos",   0.0))

    t1 = math.radians(90.0 - lift)
    t2 = t1 - math.radians(elb + 81.0)
    t3 = t2 - math.radians(wst + 5.0)

    rho_w = IK_L1 * math.cos(t1) + IK_L2 * math.cos(t2)
    z_w   = IK_L1 * math.sin(t1) + IK_L2 * math.sin(t2)

    wx = rho_w * math.cos(pan)
    wy = rho_w * math.sin(pan)
    wz = z_w

    # Approach direction (Wrist X)
    ax = math.cos(t3) * math.cos(pan)
    ay = math.cos(t3) * math.sin(pan)
    az = math.sin(t3)

    # Perpendicular direction (Wrist Z)
    zx = -math.sin(pan)
    zy =  math.cos(pan)
    zz = 0.0

    # Wrist Y = Z cross X
    yx = zy * az - zz * ay
    yy = zz * ax - zx * az
    yz = zx * ay - zy * ax
    
    R_base = np.array([
        [ax, yx, zx],
        [ay, yy, zy],
        [az, yz, zz]
    ])
    
    # Apply wrist roll around Wrist X axis
    R_roll = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(roll), -math.sin(roll)],
        [0.0, math.sin(roll),  math.cos(roll)]
    ])
    
    R_final = R_base @ R_roll

    T = np.array([
        [R_final[0,0], R_final[0,1], R_final[0,2], wx],
        [R_final[1,0], R_final[1,1], R_final[1,2], wy],
        [R_final[2,0], R_final[2,1], R_final[2,2], wz],
        [0., 0., 0., 1.],
    ])
    return T

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

def draw_frame_axes_compat(image, camera_matrix, dist_coeffs, rvec, tvec, length=0.05):
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
        except Exception:
            return None, None
        
    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

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
    params_path = os.path.join(os.path.dirname(__file__), "charuco_board_params.yaml")
    if not os.path.exists(params_path):
        board_params = {"columns": 5, "rows": 7, "square_size_mm": 35.0, "marker_size_mm": 25.0}
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
            print(f"   ⚠️ Could not connect to arm: {e}")
    
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []
    
    recorded_poses = {}
    pose_count = 0

    print("\n" + "═"*65)
    print(" SO-ARM101 HIGH-PRECISION CALIBRATION")
    print("═"*65)
    print(f" Board specs: {board_params['columns']}x{board_params['rows']} grid | Square: {board_params['square_size_mm']}mm | Marker: {board_params['marker_size_mm']}mm")
    print(" Controls:")
    print("   [Enter]  Capture pose for calibration (when green axes visible)")
    print("   [s]      Save named reference pose")
    print("   [t]      Toggle arm TORQUE ON / OFF")
    print("   [c]      Compute Multi-Algorithm Calibration")
    print("   [q]      Quit")
    print("═"*65 + "\n")
    
    try:
        while True:
            color_img, depth_img = cam.get_frames()
            if color_img is None:
                time.sleep(0.01)
                continue
                
            joints = get_pos(robot)
            T_base_wrist = forward_kinematics(joints)
            
            x_mm = T_base_wrist[0, 3]
            y_mm = T_base_wrist[1, 3]
            z_mm = T_base_wrist[2, 3]
            rho_mm = math.sqrt(x_mm**2 + y_mm**2)

            display_img = color_img.copy()
            h, w = display_img.shape[:2]

            # Detect ChArUco
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(color_img)
            
            can_capture = False
            rvec_cam, tvec_cam = None, None

            if charuco_ids is not None and len(charuco_ids) >= 6:
                cv2.aruco.drawDetectedCornersCharuco(display_img, charuco_corners, charuco_ids, (0, 255, 0))
                
                success, rvec_cam, tvec_cam = estimate_pose_charuco(
                    charuco_corners, charuco_ids, board, cam.camera_matrix, cam.dist_coeffs
                )
                if success:
                    draw_frame_axes_compat(display_img, cam.camera_matrix, cam.dist_coeffs, rvec_cam, tvec_cam, 0.05)
                    can_capture = True

            # HUD Overlay
            cv2.rectangle(display_img, (0, 0), (w, 135), (20, 20, 20), -1)

            # Line 1: Live Coordinates
            coord_str = f"Wrist FK: X={x_mm:5.1f}mm  Y={y_mm:5.1f}mm  Z={z_mm:5.1f}mm | Reach: {rho_mm:5.1f}mm"
            cv2.putText(display_img, coord_str, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

            # Line 2: Joint Angles
            joint_str = (f"Pan:{joints.get('shoulder_pan.pos',0):4.1f}° "
                         f"Lift:{joints.get('shoulder_lift.pos',0):4.1f}° "
                         f"Elbow:{joints.get('elbow_flex.pos',0):4.1f}° "
                         f"Wrist:{joints.get('wrist_flex.pos',0):4.1f}°")
            cv2.putText(display_img, joint_str, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 255, 200), 1)

            # Line 3: Calibration status
            status_color = (0, 255, 0) if can_capture else (0, 0, 255)
            status_str = f"Board: {'DETECTED (Ready)' if can_capture else 'NOT DETECTED'} | Poses: {pose_count}/15"
            cv2.putText(display_img, status_str, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 1)

            torque_str = f"Torque: {'ON' if torque_enabled else 'OFF (Free-move)'} [t] | Capture [Enter] | Calibrate [c] | Quit [q]"
            cv2.putText(display_img, torque_str, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 100), 1)

            cv2.imshow("SO-ARM101 ChArUco Calibration", display_img)
            key = cv2.waitKey(20) & 0xFF
            
            if key == ord('q'):
                break

            elif key == ord('t') and robot:
                new_state = not torque_enabled
                if set_torque(robot, new_state):
                    torque_enabled = new_state
                    print(f"🔧 Motor Torque {'ENABLED' if torque_enabled else 'DISABLED (Free-move mode)'}")

            elif key == ord('s'):
                pose_name = input("\nEnter name for this pose (e.g. scan_base, stow_base): ").strip()
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

            elif key == 13 and can_capture:  # Enter key
                T_base_wrist_m = T_base_wrist.copy()
                T_base_wrist_m[:3, 3] /= 1000.0 # to meters
                
                R_gb = T_base_wrist_m[:3, :3]
                t_gb = T_base_wrist_m[:3, 3]
                
                R_gripper2base.append(R_gb)
                t_gripper2base.append(t_gb)
                
                R_tc, _ = cv2.Rodrigues(rvec_cam)
                t_tc = tvec_cam.flatten()
                
                R_target2cam.append(R_tc)
                t_target2cam.append(t_tc)
                
                pose_count += 1
                cam_dist_mm = np.linalg.norm(t_tc) * 1000.0
                print(f"📸 Captured pose #{pose_count:2d} | Wrist FK: X={x_mm:5.1f} Y={y_mm:5.1f} Z={z_mm:5.1f} | Cam Dist: {cam_dist_mm:4.0f}mm")

                # Save raw dataset to avoid recapturing if math changes later
                dataset_file = os.path.join(os.path.dirname(__file__), "calibration_dataset.json")
                try:
                    import json
                    if not os.path.exists(dataset_file):
                        with open(dataset_file, "w") as jf: json.dump([], jf)
                    with open(dataset_file, "r") as jf: ds = json.load(jf)
                    ds.append({
                        "joints": joints,
                        "rvec_cam": rvec_cam.flatten().tolist(),
                        "tvec_cam": tvec_cam.flatten().tolist()
                    })
                    with open(dataset_file, "w") as jf: json.dump(ds, jf, indent=2)
                except Exception as e:
                    pass

            elif key == ord('c'):
                if pose_count < 5:
                    print(f"⚠️ Need at least 5 poses to compute calibration (currently have {pose_count}).")
                    continue
                
                print(f"\n⚙️  Solving Multi-Algorithm Hand-Eye Optimization from {pose_count} poses...")
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
                        mean_err, per_pose_err = evaluate_calibration_error(
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

                print("\n" + "═"*60)
                print(f"🏆 BEST METHOD: {best_method} with Mean Error = {best_error:.2f} mm")
                print("═"*60)

                out_path = os.path.join(os.path.dirname(__file__), "hand_eye_calibration.yaml")
                with open(out_path, 'w') as f:
                    yaml.dump({
                        "rotation_matrix": best_R.tolist(),
                        "translation_mm": (best_t.flatten() * 1000.0).tolist(),
                        "num_poses_used": pose_count,
                        "mean_error_mm": float(best_error),
                        "timestamp": datetime.now().isoformat(),
                        "method": best_method
                    }, f)
                    
                print(f"✅ Calibration saved to: {out_path}")
                print(f"   Translation (mm): X={best_t[0,0]*1000:.2f}, Y={best_t[1,0]*1000:.2f}, Z={best_t[2,0]*1000:.2f}")
                print("\nNow run 'python3 validate_calibration.py' to verify live tracking!")
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        if robot:
            set_torque(robot, True)
            robot.disconnect()

if __name__ == "__main__":
    main()
