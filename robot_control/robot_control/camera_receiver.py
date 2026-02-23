#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point
import socket

class CameraReceiver(Node):
    def __init__(self):
        super().__init__('camera_receiver')
        
        self.declare_parameter('udp_port', 5005)
        self.udp_port = self.get_parameter('udp_port').value
        
        self.tag_pub = self.create_publisher(String, '/camera/tag_raw', 10)
        self.distance_pub = self.create_publisher(Point, '/camera/tag_distance', 10)
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', self.udp_port))
        self.sock.settimeout(0.5)
        
        self.get_logger().info(f'Camera receiver listening on UDP port {self.udp_port}')
        
        self.create_timer(0.05, self.poll_socket)
    
    def poll_socket(self):
        try:
            data, addr = self.sock.recvfrom(65536)
        except socket.timeout:
            return
        
        if not data:
            return
        
        try:
            text = data.decode('utf-8', errors='ignore').strip()
            
            if text.startswith('TAG'):
                msg = String()
                msg.data = text
                self.tag_pub.publish(msg)
                
                parts = text.split()
                if len(parts) >= 6:
                    tag_id = int(parts[1])
                    cx = float(parts[2])
                    cy = float(parts[3])
                    rotation = float(parts[4])
                    distance = float(parts[5])
                    x_offset = float(parts[6])
                    
                    dist_msg = Point()
                    dist_msg.x = distance
                    dist_msg.y = x_offset
                    dist_msg.z = float(tag_id)
                    self.distance_pub.publish(dist_msg)
                    
                    self.get_logger().info(
                        f'Tag {tag_id}: dist={distance:.2f}m, offset={x_offset:.1f}px, rot={rotation:.2f}'
                    )
        except Exception as e:
            self.get_logger().warning(f'Failed to process packet: {e}')

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
