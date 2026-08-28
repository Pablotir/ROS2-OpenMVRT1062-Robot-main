#!/usr/bin/env python3
"""Shared utilities for the arm_manipulation package."""

import numpy as np
import cv2
import math
from typing import Optional, Tuple, Dict, List
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time

# Joint name mapping between ROS and LeRobot
ROS_JOINT_NAMES = [
    'shoulder_pan',
    'shoulder_lift', 
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper'
]

LEROBOT_JOINT_KEYS = [
    'shoulder_pan.pos',
    'shoulder_lift.pos',
    'elbow_flex.pos', 
    'wrist_flex.pos',
    'wrist_roll.pos',
    'gripper.pos'
]

def ros_to_lerobot_joints(joint_state: JointState) -> dict:
    """Convert ROS JointState message to LeRobot joint dict."""
    result = {}
    for i, name in enumerate(joint_state.name):
        if name in ROS_JOINT_NAMES:
            idx = ROS_JOINT_NAMES.index(name)
            result[LEROBOT_JOINT_KEYS[idx]] = joint_state.position[idx]
    return result

def lerobot_to_ros_joints(lerobot_dict: dict) -> JointState:
    """Convert LeRobot joint dict to ROS JointState message."""
    msg = JointState()
    for lk, rn in zip(LEROBOT_JOINT_KEYS, ROS_JOINT_NAMES):
        if lk in lerobot_dict:
            msg.name.append(rn)
            msg.position.append(float(lerobot_dict[lk]))
    return msg

def degrees_to_radians(deg: float) -> float:
    return math.radians(deg)

def radians_to_degrees(rad: float) -> float:
    return math.degrees(rad)

def rotation_matrix_to_quaternion(R: np.ndarray) -> tuple:
    """Convert a 3x3 rotation matrix to quaternion (x, y, z, w).
    Uses Shepperd's method for numerical stability."""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return (x, y, z, w)

def yuyv_to_bgr(raw_data: bytes, height: int, width: int) -> np.ndarray:
    """Convert YUYV raw bytes to BGR image. Handles D405 Jetson format."""
    raw = np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width, 2)
    return cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUYV)

def compute_mask_centroid(mask_polygon: np.ndarray, img_shape: tuple) -> Optional[tuple]:
    """Compute centroid of a segmentation mask polygon."""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [mask_polygon.astype(np.int32)], 255)
    M = cv2.moments(mask)
    if M['m00'] > 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)
    return None

def euler_from_quaternion(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    """Convert a quaternion into euler angles (roll, pitch, yaw)."""
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z

def create_transform_stamped(
        translation: Tuple[float, float, float],
        rotation_q: Tuple[float, float, float, float],
        frame_id: str,
        child_frame_id: str,
        stamp: Time) -> TransformStamped:
    """Helper to create a ROS 2 TransformStamped message."""
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = frame_id
    t.child_frame_id = child_frame_id
    t.transform.translation.x = translation[0]
    t.transform.translation.y = translation[1]
    t.transform.translation.z = translation[2]
    t.transform.rotation.x = rotation_q[0]
    t.transform.rotation.y = rotation_q[1]
    t.transform.rotation.z = rotation_q[2]
    t.transform.rotation.w = rotation_q[3]
    return t
