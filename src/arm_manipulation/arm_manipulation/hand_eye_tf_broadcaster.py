#!/usr/bin/env python3
import os
import yaml
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from scipy.spatial.transform import Rotation

class HandEyeTfBroadcaster(Node):
    def __init__(self):
        super().__init__('hand_eye_tf_broadcaster')

        # Parameters
        self.declare_parameter('calibration_file', '/root/ros2_ws/calibration/hand_eye_calibration.yaml')
        self.declare_parameter('parent_frame', 'wrist_flex_link')
        self.declare_parameter('child_frame', 'camera_link')

        self.calib_file = self.get_parameter('calibration_file').value
        self.parent_frame = self.get_parameter('parent_frame').value
        self.child_frame = self.get_parameter('child_frame').value

        self.tf_broadcaster = StaticTransformBroadcaster(self)

        self.publish_static_transform()

    def publish_static_transform(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame

        loaded_successfully = False

        if os.path.exists(self.calib_file):
            try:
                with open(self.calib_file, 'r') as f:
                    calib_data = yaml.safe_load(f)

                if 'rotation_matrix' in calib_data and 'translation_mm' in calib_data:
                    rot_matrix = calib_data['rotation_matrix']
                    trans_mm = calib_data['translation_mm'][0] if isinstance(calib_data['translation_mm'][0], list) else calib_data['translation_mm']

                    # Convert mm to meters
                    t.transform.translation.x = float(trans_mm[0]) / 1000.0
                    t.transform.translation.y = float(trans_mm[1]) / 1000.0
                    t.transform.translation.z = float(trans_mm[2]) / 1000.0

                    # Convert rotation matrix to quaternion
                    r = Rotation.from_matrix(rot_matrix)
                    quat = r.as_quat() # [x, y, z, w]

                    t.transform.rotation.x = quat[0]
                    t.transform.rotation.y = quat[1]
                    t.transform.rotation.z = quat[2]
                    t.transform.rotation.w = quat[3]

                    self.get_logger().info(f"Loaded calibration from {self.calib_file}")
                    loaded_successfully = True
                else:
                    self.get_logger().error(f"Missing rotation_matrix or translation_mm in {self.calib_file}")
            except Exception as e:
                self.get_logger().error(f"Failed to parse {self.calib_file}: {e}")
        else:
            self.get_logger().warning(f"Calibration file not found: {self.calib_file}")

        if not loaded_successfully:
            self.get_logger().warning("Publishing default transform based on ruler measurements.")
            # Default: translation [0.0, 0.05, 0.0] meters
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.05
            t.transform.translation.z = 0.0

            # Default: rotation 45 deg pitch down around Y axis
            r = Rotation.from_euler('y', -45, degrees=True)
            quat = r.as_quat()

            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)
        self.get_logger().info(f"Published static transform: {self.parent_frame} -> {self.child_frame}")

def main(args=None):
    rclpy.init(args=args)
    node = HandEyeTfBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
