#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point
import time

class LLMPlanner(Node):
    def __init__(self):
        super().__init__('llm_planner')
        
        self.pub = self.create_publisher(String, '/llm/motor_plan', 10)
        self.tag_sub = self.create_subscription(String, '/camera/tag_raw', self.on_tag, 10)
        self.distance_sub = self.create_subscription(Point, '/camera/tag_distance', self.on_distance, 10)
        self.slam_sub = self.create_subscription(PoseStamped, '/slam/pose', self.on_slam, 10)
        
        self.last_tag_time = time.time()
        self.last_distance = None
        self.last_x_offset = None
        self.last_tag_id = None
        self.last_pose = None
        
        self.target_distance = 0.5
        self.distance_tolerance = 0.1
        self.center_threshold = 20
        
        self.create_timer(0.5, self.plan_movement)
        
        self.get_logger().info("Smart LLM Planner started - using camera feedback")
    
    def on_tag(self, msg: String):
        self.last_tag_time = time.time()
        parts = msg.data.split()
        if len(parts) >= 2:
            self.last_tag_id = int(parts[1])
    
    def on_distance(self, msg: Point):
        self.last_distance = msg.x
        self.last_x_offset = msg.y
        self.last_tag_id = int(msg.z)
    
    def on_slam(self, msg: PoseStamped):
        self.last_pose = msg
    
    def plan_movement(self):
        time_since_tag = time.time() - self.last_tag_time
        
        if time_since_tag > 2.0:
            plan = "1,50,2,-50,3,50,4,-50"
            self.get_logger().info("No tag detected - searching (rotating)")
        
        elif self.last_distance is not None and self.last_x_offset is not None:
            distance_error = self.last_distance - self.target_distance
            
            motor_values = [0, 0, 0, 0]
            
            if abs(self.last_x_offset) > self.center_threshold:
                rotation_power = int(self.last_x_offset * 0.8)
                rotation_power = max(-100, min(100, rotation_power))
                
                motor_values[0] = rotation_power
                motor_values[1] = rotation_power
                motor_values[2] = rotation_power
                motor_values[3] = rotation_power
                
                self.get_logger().info(f"Centering tag (offset={self.last_x_offset:.1f}px)")
            
            elif abs(distance_error) > self.distance_tolerance:
                forward_power = int(distance_error * 200)
                forward_power = max(-100, min(100, forward_power))
                
                motor_values[0] = forward_power
                motor_values[1] = forward_power
                motor_values[2] = forward_power
                motor_values[3] = forward_power
                
                self.get_logger().info(f"Adjusting distance (current={self.last_distance:.2f}m, target={self.target_distance}m)")
            
            else:
                motor_values = [0, 0, 0, 0]
                self.get_logger().info(f"Tag centered and at target distance - holding position")
            
            plan = "{},{},{},{},{},{},{},{}".format(
                1, motor_values[0],
                2, motor_values[1],
                3, motor_values[2],
                4, motor_values[3]
            )
        
        else:
            plan = "1,0,2,0,3,0,4,0"
            self.get_logger().info("Waiting for camera data...")
        
        m = String()
        m.data = plan
        self.pub.publish(m)

def main(args=None):
    rclpy.init(args=args)
    node = LLMPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
