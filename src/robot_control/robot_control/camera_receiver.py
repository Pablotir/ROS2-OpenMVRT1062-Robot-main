#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import socket
import cv2
import numpy as np

class CameraReceiver(Node):
    def __init__(self):
        super().__init__('camera_receiver')
        
        self.declare_parameter('udp_port', 5005)
        self.udp_port = self.get_parameter('udp_port').value
        
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        
        self.bridge = CvBridge()
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', self.udp_port))
        self.sock.settimeout(0.5)
        
        self.get_logger().info(f'Camera receiver on port {self.udp_port}')
        
        self.frame_count = 0
        self.create_timer(0.05, self.poll_socket)
    
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
            
            header = data[:header_end].decode('utf-8', errors='ignore')
            
            if header.startswith('AI') or header.startswith('FRAME'):
                parts = header.split()
                if len(parts) >= 3:
                    frame_id = int(parts[1])
                    jpeg_size = int(parts[2])
                    
                    jpeg_data = data[header_end + 1:]
                    
                    if len(jpeg_data) >= jpeg_size:
                        img_array = np.frombuffer(jpeg_data[:jpeg_size], dtype=np.uint8)
                        cv_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        
                        if cv_img is not None:
                            ros_img = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
                            ros_img.header.stamp = self.get_clock().now().to_msg()
                            ros_img.header.frame_id = 'camera'
                            
                            self.image_pub.publish(ros_img)
                            self.frame_count += 1
                            
                            if self.frame_count % 5 == 0:
                                self.get_logger().info(f'AI frame {frame_id} ({jpeg_size} bytes)')
        
        except Exception as e:
            self.get_logger().warning(f'Failed to process: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = CameraReceiver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
