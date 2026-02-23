#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
import requests
import json
import base64
import time
import threading

class AINavigator(Node):
    def __init__(self):
        super().__init__('ai_navigator')
        
        self.declare_parameter('ollama_host', 'http://localhost:8080')
        self.declare_parameter('model', 'gemma3:4b')
        self.declare_parameter('goal', 'find the kitchen to get a water bottle from the fridge')
        self.declare_parameter('stabilization_delay', 1.0)
        
        self.ollama_host = self.get_parameter('ollama_host').value
        self.model = self.get_parameter('model').value
        self.goal = self.get_parameter('goal').value
        self.stabilization_delay = self.get_parameter('stabilization_delay').value
        
        self.bridge = CvBridge()
        
        self.image_sub = self.create_subscription(
            Image,
            '/camera/usb_raw',
            self.on_image,
            10
        )
        
        self.frame_request_pub = self.create_publisher(Bool, '/camera/request_frame', 10)
        
        self.decision_pub = self.create_publisher(String, '/ai/decision', 10)
        self.direction_pub = self.create_publisher(String, '/ai/direction', 10)
        self.label_pub = self.create_publisher(String, '/ai/semantic_label', 10)
        
        self.movement_complete_sub = self.create_subscription(
            Bool,
            '/robot/movement_complete',
            self.on_movement_complete,
            10
        )
        
        self.last_frame = None
        self.waiting_for_frame = False
        self.processing = False
        self.movement_in_progress = False
        
        self.get_logger().info(f"AI Navigator started")
        self.get_logger().info(f"Goal: {self.goal}")
        self.get_logger().info(f"Stabilization delay: {self.stabilization_delay}s")
        
        self.request_first_frame_timer = self.create_timer(2.0, self.request_first_frame)
    
    def request_first_frame(self):
        self.request_first_frame_timer.cancel()
        self.get_logger().info("Requesting initial frame...")
        self.request_frame()
    
    def on_movement_complete(self, msg: Bool):
        if msg.data and not self.processing:
            self.get_logger().info(f"Movement complete, waiting {self.stabilization_delay}s for stabilization...")
            self.movement_in_progress = False
            
            threading.Timer(self.stabilization_delay, self.request_frame_after_stabilization).start()
    
    def request_frame_after_stabilization(self):
        self.get_logger().info("Camera stabilized, requesting new frame...")
        self.request_frame()
    
    def request_frame(self):
        if self.processing:
            self.get_logger().warn("Still processing previous frame, skipping...")
            return
        
        self.waiting_for_frame = True
        req = Bool()
        req.data = True
        self.frame_request_pub.publish(req)
    
    def on_image(self, msg: Image):
        if not self.waiting_for_frame:
            return
        
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.last_frame = cv_img
            self.waiting_for_frame = False
            self.get_logger().info("Frame received, analyzing...")
            self.analyze_frame()
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
    
    def analyze_frame(self):
        if self.last_frame is None:
            self.get_logger().warn("No frame to analyze")
            return
        
        self.processing = True
        self.movement_in_progress = True
        
        try:
            combined_analysis = self.query_vision_model(
                self.last_frame,
                "What room is this? Can robot move forward? Answer format: [room type], [yes/no + reason]"
            )
            
            self.get_logger().info(f"AI says: {combined_analysis}")
            
            direction = self.parse_direction(combined_analysis)
            scene = self.extract_scene(combined_analysis)
            
            label_msg = String()
            label_msg.data = scene
            self.label_pub.publish(label_msg)
            
            dir_msg = String()
            dir_msg.data = direction
            self.direction_pub.publish(dir_msg)
            
            decision_msg = String()
            decision_msg.data = json.dumps({
                'scene': scene,
                'direction': direction,
                'reasoning': combined_analysis
            })
            self.decision_pub.publish(decision_msg)
            
        except Exception as e:
            self.get_logger().error(f"Analysis failed: {e}")
        finally:
            self.processing = False
    
    def query_vision_model(self, frame, prompt):
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=30.0
            )
            
            inference_time = time.time() - start_time
            self.get_logger().info(f"AI inference took {inference_time:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'unclear')
            else:
                self.get_logger().error(f"API error: {response.status_code}")
                return "unclear"
        
        except Exception as e:
            self.get_logger().error(f"Vision model query failed: {e}")
            return "unclear"
    
    def parse_direction(self, response):
        response_lower = response.lower()
        
        if 'yes' in response_lower or 'safe' in response_lower or 'clear' in response_lower:
            return 'forward'
        elif 'no' in response_lower or 'blocked' in response_lower or 'wall' in response_lower:
            return 'left'
        elif 'right' in response_lower:
            return 'right'
        elif 'left' in response_lower:
            return 'left'
        else:
            return 'forward'
    
    def extract_scene(self, response):
        response_lower = response.lower()
        
        if 'kitchen' in response_lower:
            return 'kitchen'
        elif 'bedroom' in response_lower:
            return 'bedroom'
        elif 'living' in response_lower or 'room' in response_lower:
            return 'living room'
        elif 'hallway' in response_lower or 'corridor' in response_lower:
            return 'hallway'
        elif 'bathroom' in response_lower:
            return 'bathroom'
        elif 'storage' in response_lower:
            return 'storage'
        else:
            return 'unknown room'

def main(args=None):
    rclpy.init(args=args)
    node = AINavigator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
