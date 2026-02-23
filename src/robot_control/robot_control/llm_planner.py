#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import time

class LLMSimulator(Node):
    def __init__(self):
        super().__init__('llm_simulator')
        
        self.pub = self.create_publisher(String, '/llm/motor_plan', 10)
        self.tag_sub = self.create_subscription(String, '/camera/tag_raw', self.on_tag, 10)
        self.slam_sub = self.create_subscription(PoseStamped, '/slam/pose', self.on_slam, 10)
        
        self.last_tag = None
        self.last_pose = None
        
        self.create_timer(2.0, self.auto_publish)
    
    def on_tag(self, msg: String):
        self.last_tag = msg.data
        self.get_logger().info(f"LLM saw tag: {msg.data}")
    
    def on_slam(self, msg: PoseStamped):
        self.last_pose = msg
    
    def auto_publish(self):
        if self.last_tag:
            plan = "1,100,2,100,3,100,4,100"
        else:
            plan = "1,-100,2,-100,3,-100,4,-100"
        
        m = String()
        m.data = plan
        self.pub.publish(m)
        self.get_logger().info(f"Published plan: {plan}")

def main(args=None):
    rclpy.init(args=args)
    node = LLMSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
