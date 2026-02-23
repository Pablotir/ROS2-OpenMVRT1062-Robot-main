#!/usr/bin/env python3
"""
ai_navigator.py  — scene labeller (NOT a movement controller)
=============================================================
Triggered by exploration_controller every N steps via /robot/movement_complete.
Grabs a camera frame, asks Ollama to label what it sees, and publishes the
label to /ai/semantic_label for later map annotation.

Movement decisions are made by exploration_controller using the ultrasonic
sensor — this node never publishes to /cmd_vel or /ai/direction.

Subscribes
----------
/robot/movement_complete  std_msgs/Bool   trigger from exploration_controller
/camera/usb_raw           sensor_msgs/Image   on-demand frame

Publishes
----------
/camera/request_frame     std_msgs/Bool   requests a frame from usb_camera_node
/ai/semantic_label        std_msgs/String  e.g. "bedroom - near desk"
/ai/decision              std_msgs/String  full JSON for logging
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import cv2
import requests
import json
import base64
import time
import threading


class AINavigator(Node):
    def __init__(self):
        super().__init__('ai_navigator')

        self.declare_parameter('ollama_host',  'http://localhost:11434')
        self.declare_parameter('model',         'gemma3:4b')
        self.declare_parameter('infer_width',   160)
        self.declare_parameter('infer_height',  120)
        self.declare_parameter('jpeg_quality',  65)

        self.ollama_host   = self.get_parameter('ollama_host').value
        self.model         = self.get_parameter('model').value
        self.infer_width   = self.get_parameter('infer_width').value
        self.infer_height  = self.get_parameter('infer_height').value
        self.jpeg_quality  = self.get_parameter('jpeg_quality').value

        self.bridge = CvBridge()
        self._waiting_for_frame = False
        self._processing        = False
        self._last_label        = 'unknown'

        # Subscribers
        self.create_subscription(Bool,  '/robot/movement_complete', self._on_trigger, 10)
        self.create_subscription(Image, '/camera/usb_raw',          self._on_image,   10)

        # Publishers
        self._req_pub   = self.create_publisher(Bool,   '/camera/request_frame', 10)
        self._label_pub = self.create_publisher(String, '/ai/semantic_label',    10)
        self._dec_pub   = self.create_publisher(String, '/ai/decision',          10)

        self.get_logger().info(f'AI scene labeller ready | model={self.model} | '
                               f'inference size={self.infer_width}x{self.infer_height}')

    # ── Trigger ────────────────────────────────────────────────────────────────
    def _on_trigger(self, msg: Bool):
        if not msg.data or self._processing:
            return
        self.get_logger().info('Triggered — requesting camera frame for scene label...')
        self._waiting_for_frame = True
        req = Bool()
        req.data = True
        self._req_pub.publish(req)

    # ── Frame received ─────────────────────────────────────────────────────────
    def _on_image(self, msg: Image):
        if not self._waiting_for_frame or self._processing:
            return
        self._waiting_for_frame = False
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
            return
        # Run inference in background so ROS callbacks stay responsive
        threading.Thread(target=self._label_scene, args=(frame,), daemon=True).start()

    # ── Inference ──────────────────────────────────────────────────────────────
    def _label_scene(self, frame):
        self._processing = True
        try:
            # Resize + compress
            small = cv2.resize(frame, (self.infer_width, self.infer_height),
                               interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode('.jpg', small,
                                  [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            img_b64 = base64.b64encode(buf).decode('utf-8')

            prompt = (
                "You are labelling locations in a bedroom for a robot map. "
                "In 5 words or fewer, describe what area of the bedroom this is "
                "(e.g. 'near desk', 'corner by wardrobe', 'open floor', 'doorway'). "
                "Reply with ONLY the label, nothing else."
            )

            t0 = time.time()
            resp = requests.post(
                f"{self.ollama_host}/api/generate",
                json={"model": self.model, "prompt": prompt,
                      "images": [img_b64], "stream": False},
                timeout=20.0
            )
            elapsed = time.time() - t0

            if resp.status_code == 200:
                label = resp.json().get('response', '').strip().strip('"').lower()
                if not label:
                    label = 'bedroom area'
            else:
                label = 'unknown'
                self.get_logger().warn(f'Ollama returned {resp.status_code}')

            self.get_logger().info(f'Scene label: "{label}" ({elapsed:.1f}s)')
            self._last_label = label

            lmsg = String()
            lmsg.data = label
            self._label_pub.publish(lmsg)

            dmsg = String()
            dmsg.data = json.dumps({'label': label, 'inference_s': round(elapsed, 2)})
            self._dec_pub.publish(dmsg)

        except Exception as e:
            self.get_logger().error(f'Scene labelling failed: {e}')
        finally:
            self._processing = False


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

    def __init__(self):
        super().__init__('ai_navigator')
        
        self.declare_parameter('ollama_host', 'http://localhost:11434')
        self.declare_parameter('model', 'gemma3:4b')
        self.declare_parameter('goal', 'explore the bedroom and build a complete map')
        self.declare_parameter('stabilization_delay', 1.0)
        self.declare_parameter('infer_width', 160)   # resize before sending to Ollama
        self.declare_parameter('infer_height', 120)  # smaller = faster inference
        
        self.ollama_host = self.get_parameter('ollama_host').value
        self.model = self.get_parameter('model').value
        self.goal = self.get_parameter('goal').value
        self.stabilization_delay = self.get_parameter('stabilization_delay').value
        self.infer_width = self.get_parameter('infer_width').value
        self.infer_height = self.get_parameter('infer_height').value
        
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
                "You are a navigation assistant for a robot exploring a bedroom. "
                "Look at this image and answer in exactly this format: "
                "DIRECTION: [forward|left|right|stop] REASON: [one short sentence]. "
                "Choose forward if the path ahead is clear (>1 meter). "
                "Choose left or right if blocked by a wall, furniture or obstacle. "
                "Choose stop only if something dangerous is directly in front."
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
            # Resize to small resolution before encoding — huge speedup on Ollama
            small = cv2.resize(frame, (self.infer_width, self.infer_height),
                               interpolation=cv2.INTER_AREA)
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
            _, buffer = cv2.imencode('.jpg', small, encode_params)
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

        # Look for explicit DIRECTION: tag first
        if 'direction:' in response_lower:
            after = response_lower.split('direction:')[-1].strip()
            if after.startswith('forward'):  return 'forward'
            if after.startswith('left'):     return 'left'
            if after.startswith('right'):    return 'right'
            if after.startswith('stop'):     return 'stop'

        # Fallback keyword scan
        if any(w in response_lower for w in ['blocked', 'wall', 'obstacle', 'furniture', 'cannot', 'no path']):
            return 'left'
        if 'right' in response_lower:
            return 'right'
        if 'left' in response_lower:
            return 'left'
        if 'stop' in response_lower:
            return 'stop'
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
