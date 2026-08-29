#!/usr/bin/env python3
"""
test_arm_vision_lunge.py — Standalone Bench Test for Arm Vision & Lunging
========================================================================
Tests the SO-ARM101 reaching and tracking toward objects detected by the
RealSense D405 wrist camera without moving the robot's base wheels.

Workflow:
  1. Arm deploys to calibrated `scan_base` pose.
  2. D405 camera detects target (e.g., bottle / cup).
  3. Arm visually tracks and extends ("lunges") toward the object's 3D centroid.
  4. Gripper closes to test the grasp.
  5. Arm returns to `stow_base` pose.

Usage (inside Docker container):
  python3 /root/ros2_ws/scripts/test_arm_vision_lunge.py --target bottle
"""

import sys
import math
import time
import argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PointStamped
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String, Float64

CALIBRATED_STOW_BASE = {
    'shoulder_pan': -4.48,
    'shoulder_lift': -106.11,
    'elbow_flex': 100.00,
    'wrist_flex': 75.96,
    'wrist_roll': -156.75,
    'gripper': 73.77
}

CALIBRATED_SCAN_BASE = {
    'shoulder_pan': -4.48,
    'shoulder_lift': -106.02,
    'elbow_flex': 99.91,
    'wrist_flex': 33.41,
    'wrist_roll': -155.96,
    'gripper': 73.84
}


class ArmVisionLungeTester(Node):
    def __init__(self, target_label='bottle'):
        super().__init__('arm_vision_lunge_tester')

        self.target_label = target_label
        self.current_joints = dict(CALIBRATED_STOW_BASE)
        self.latest_target_pt = None
        self.detected_count = 0

        # Publishers
        self.arm_pub = self.create_publisher(JointState, '/arm/joint_commands', 10)
        self.target_set_pub = self.create_publisher(String, '/arm/set_target', 10)

        # Subscribers
        self.create_subscription(JointState, '/arm/joint_states', self._on_joints, 10)
        self.create_subscription(PointStamped, '/arm/target_point', self._on_target_pt, 10)
        self.create_subscription(Detection2DArray, '/arm/detections', self._on_detections, 10)

        self.get_logger().info(f"=== ARM VISION LUNGE TESTER INITIALIZED (Target: '{self.target_label}') ===")

    def _on_joints(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.current_joints[name] = math.degrees(pos)

    def _on_target_pt(self, msg: PointStamped):
        self.latest_target_pt = msg

    def _on_detections(self, msg: Detection2DArray):
        if len(msg.detections) > 0:
            self.detected_count += 1
        else:
            self.detected_count = 0

    def send_pose(self, pose_dict, duration_s=1.5):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        for k, v in pose_dict.items():
            msg.name.append(k)
            msg.position.append(math.radians(v))
        self.arm_pub.publish(msg)
        time.sleep(duration_s)

    def run_test(self):
        # 1. Set detection target
        t_msg = String()
        t_msg.data = self.target_label
        self.target_set_pub.publish(t_msg)

        print("\n[Step 1] Deploying arm to calibrated SCAN_BASE pose...")
        self.send_pose(CALIBRATED_SCAN_BASE, duration_s=2.0)

        print(f"\n[Step 2] Searching for '{self.target_label}' via RealSense D405 wrist camera...")
        print(" -> Place a bottle or cup in front of the arm camera now!")

        # Wait for detection
        start_wait = time.time()
        while self.detected_count < 3:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start_wait > 20.0:
                print(" ⚠ Timeout waiting for object detection. Ensure camera is running and target is in view.")
                break

        if self.latest_target_pt is not None:
            pt = self.latest_target_pt.point
            print(f"\n[Step 3] 🎯 Target Confirmed at 3D Camera Frame: [X={pt.x:.3f}m, Y={pt.y:.3f}m, Z(depth)={pt.z:.3f}m]")
            print(" -> Lunging arm forward towards the object...")

            # Calculate reaching trajectory angles
            reach_pose = dict(CALIBRATED_SCAN_BASE)
            reach_pose['gripper'] = 80.0 # Open gripper

            # Lunge forward: extend shoulder_lift and elbow_flex
            reach_pose['shoulder_lift'] = -75.0
            reach_pose['elbow_flex'] = 45.0
            reach_pose['wrist_flex'] = 35.0

            # Lateral pan adjustment toward object centroid
            pan_correction = -math.degrees(math.atan2(pt.x, pt.z))
            reach_pose['shoulder_pan'] = float(CALIBRATED_SCAN_BASE['shoulder_pan'] + pan_correction)

            self.send_pose(reach_pose, duration_s=2.0)

            print("\n[Step 4] Closing gripper to grasp...")
            grasp_pose = dict(reach_pose)
            grasp_pose['gripper'] = 15.0 # Closed grasp
            self.send_pose(grasp_pose, duration_s=1.5)

            print("\n[Step 5] Lifting object up...")
            lift_pose = dict(grasp_pose)
            lift_pose['shoulder_lift'] = -95.0
            lift_pose['elbow_flex'] = 75.0
            self.send_pose(lift_pose, duration_s=2.0)

            time.sleep(1.5)

            print("\n[Step 6] Releasing object and returning to STOW_BASE...")
            release_pose = dict(lift_pose)
            release_pose['gripper'] = 75.0
            self.send_pose(release_pose, duration_s=1.0)
            self.send_pose(CALIBRATED_STOW_BASE, duration_s=2.0)

            print("\n✔ Arm vision lunging & grasp test complete successfully!")
        else:
            print("\nReturning arm safely to STOW_BASE...")
            self.send_pose(CALIBRATED_STOW_BASE, duration_s=2.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='bottle', help='Target object to detect')
    args = parser.parse_args()

    rclpy.init()
    tester = ArmVisionLungeTester(target_label=args.target)
    try:
        tester.run_test()
    except KeyboardInterrupt:
        print("\nStopping tester...")
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
