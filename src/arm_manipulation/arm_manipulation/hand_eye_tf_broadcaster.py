#!/usr/bin/env python3
import os
import yaml
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
def rot_matrix_to_quat(R):
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
    m00, m01, m02 = float(R[0][0]), float(R[0][1]), float(R[0][2])
    m10, m11, m12 = float(R[1][0]), float(R[1][1]), float(R[1][2])
    m20, m21, m22 = float(R[2][0]), float(R[2][1]), float(R[2][2])
    tr = m00 + m11 + m22
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    return [qx, qy, qz, qw]

class HandEyeTfBroadcaster(Node):
    def __init__(self):
        super().__init__('hand_eye_tf_broadcaster')

        # Parameters
        self.declare_parameter('calibration_file', '/root/ros2_ws/calibration/hand_eye_calibration.yaml')
        self.declare_parameter('parent_frame', 'wrist_flex_link')
        self.declare_parameter('child_frame', 'd405_color_optical_frame')

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
                    raw_trans = calib_data['translation_mm']
                    
                    # Flatten any nested column vectors e.g. [[102], [75], [27]]
                    flat_trans = []
                    for item in raw_trans:
                        if isinstance(item, (list, tuple)):
                            flat_trans.extend(item)
                        else:
                            flat_trans.append(item)

                    # Convert mm to meters
                    t.transform.translation.x = float(flat_trans[0]) / 1000.0
                    t.transform.translation.y = float(flat_trans[1]) / 1000.0
                    t.transform.translation.z = float(flat_trans[2]) / 1000.0

                    # Convert rotation matrix to quaternion
                    quat = rot_matrix_to_quat(rot_matrix)

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
            self.get_logger().warning("Publishing default transform based on calibrated measurements.")
            t.transform.translation.x = 0.10284
            t.transform.translation.y = 0.07573
            t.transform.translation.z = 0.02792
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

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
