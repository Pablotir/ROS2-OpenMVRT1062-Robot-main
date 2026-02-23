#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import socket
import cv2
import numpy as np
import json

class FusedSLAMReceiver(Node):
    def __init__(self):
        super().__init__('slam_receiver')
        
        self.declare_parameter('udp_port', 5006)
        self.udp_port = self.get_parameter('udp_port').value
        
        self.image_pub = self.create_publisher(Image, '/slam/image', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/slam/pose', 10)
        self.distance_pub = self.create_publisher(Float32, '/slam/distance', 10)
        
        self.encoder_sub = self.create_subscription(
            Point, '/robot/encoders', self.on_encoders, 10
        )
        
        self.bridge = CvBridge()
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', self.udp_port))
        self.sock.settimeout(0.5)
        
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        
        self.last_left_enc = None
        self.last_right_enc = None
        
        self.ticks_per_meter = 1440 / (0.1016 * 3.14159)
        self.wheel_base = 0.3
        
        self.current_distance = 0.0
        self.frame_count = 0
        
        self.get_logger().info(f'Fused SLAM receiver on port {self.udp_port}')
        
        self.create_timer(0.02, self.poll_socket)
    
    def on_imu(self, msg: Imu):
        gz = msg.angular_velocity.z
        dt = 0.05
        self.robot_theta += gz * dt
    
    def on_encoders(self, msg: Point):
        left_enc = msg.x
        right_enc = msg.y
        
        if self.last_left_enc is not None:
            delta_left = (left_enc - self.last_left_enc) / self.ticks_per_meter
            delta_right = (right_enc - self.last_right_enc) / self.ticks_per_meter
            
            delta_distance = (delta_left + delta_right) / 2.0
            delta_theta = (delta_right - delta_left) / self.wheel_base
            
            self.robot_theta += delta_theta
            self.robot_x += delta_distance * np.cos(self.robot_theta)
            self.robot_y += delta_distance * np.sin(self.robot_theta)
        
        self.last_left_enc = left_enc
        self.last_right_enc = right_enc
    
    def poll_socket(self):
        try:
            data, addr = self.sock.recvfrom(65536)
        except socket.timeout:
            return
        
        if not data:
            return
        
        try:
            header_end = data.find(b'\n')
            if header_end == -1:
                return
            
            header_str = data[:header_end].decode('utf-8', errors='ignore')
            slam_data = json.loads(header_str)
            jpeg_data = data[header_end + 1:]
            
            distance = slam_data.get('distance', 0.0)
            self.current_distance = distance
            
            dist_msg = Float32()
            dist_msg.data = distance
            self.distance_pub.publish(dist_msg)
            
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'map'
            pose_msg.pose.position.x = self.robot_x
            pose_msg.pose.position.y = self.robot_y
            pose_msg.pose.position.z = 0.0
            
            qz = np.sin(self.robot_theta / 2)
            qw = np.cos(self.robot_theta / 2)
            pose_msg.pose.orientation.z = qz
            pose_msg.pose.orientation.w = qw
            
            self.pose_pub.publish(pose_msg)
            
            if len(jpeg_data) > 0:
                img_array = np.frombuffer(jpeg_data, dtype=np.uint8)
                cv_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if cv_img is not None:
                    ros_img = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
                    ros_img.header.stamp = self.get_clock().now().to_msg()
                    ros_img.header.frame_id = 'camera_slam'
                    self.image_pub.publish(ros_img)
            
            self.frame_count += 1
            
            if self.frame_count % 100 == 0:
                self.get_logger().info(
                    f'SLAM: {self.frame_count} frames | '
                    f'Pose=({self.robot_x:.2f}, {self.robot_y:.2f}, {np.degrees(self.robot_theta):.1f}°) | '
                    f'Distance={distance:.2f}m'
                )
        
        except Exception as e:
            self.get_logger().warning(f'Process error: {e}')
    
    def destroy_node(self):
        self.sock.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = FusedSLAMReceiver()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
