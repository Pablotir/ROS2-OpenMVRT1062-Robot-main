#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point, Vector3
from sensor_msgs.msg import Imu
import serial
import time
import re

def parse_plan(text):
    text = text.strip().lower()
    pairs = []
    explicit = re.findall(r'(?:motor|m)?\s*(\d+)[^\d\-+\.]*([+-]?\d+(?:\.\d+)?)', text)
    if explicit:
        for m, val in explicit:
            try:
                pairs.append((int(m), float(val)))
            except:
                continue
        return pairs
    nums = re.findall(r'[+-]?\d+(?:\.\d+)?', text)
    if len(nums) % 2 == 0 and len(nums) > 0:
        nums = [float(x) for x in nums]
        for i in range(0, len(nums), 2):
            pairs.append((int(nums[i]), nums[i+1]))
    return pairs

class RobotControlNode(Node):
    def __init__(self):
        super().__init__('robot_control_node')
        
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('batch_mode', True)
        self.declare_parameter('speed', 0.5)
        
        self.serial_port = self.get_parameter('serial_port').value
        self.baud = self.get_parameter('baud').value
        self.batch_mode = self.get_parameter('batch_mode').value
        self.default_speed = float(self.get_parameter('speed').value)
        
        self.get_logger().info(f"Starting robot_control_node (serial={self.serial_port})")
        
        try:
            self.ser = serial.Serial(self.serial_port, int(self.baud), timeout=1)
            time.sleep(0.1)
            self.get_logger().info(f"Serial opened: {self.serial_port}")
        except Exception as e:
            self.get_logger().error(f"Failed to open serial {self.serial_port}: {e}")
            self.ser = None
        
        # Subscribe to motor commands
        self.sub = self.create_subscription(String, '/llm/motor_plan', self.on_plan, 10)
        
        # Publishers for encoder and IMU data
        self.encoder_pub = self.create_publisher(Point, '/robot/encoders', 10)
        self.imu_pub = self.create_publisher(Imu, '/robot/imu', 10)
        
        # Timer to read serial
        self.create_timer(0.05, self.read_serial)
        
        self.serial_buffer = ""
    
    def read_serial(self):
        """Read encoder and IMU data from Arduino"""
        if not self.ser or not self.ser.is_open:
            return
        
        try:
            while self.ser.in_waiting > 0:
                char = self.ser.read(1).decode('utf-8', errors='ignore')
                
                if char == '\n':
                    line = self.serial_buffer.strip()
                    self.serial_buffer = ""
                    
                    # Parse DATA line: "DATA enc1 enc2 enc3 enc4 ax ay az gx gy gz"
                    if line.startswith('DATA'):
                        parts = line.split()
                        if len(parts) >= 5:
                            try:
                                # Encoders (average left/right)
                                left_avg = (float(parts[1]) + float(parts[3])) / 2.0
                                right_avg = (float(parts[2]) + float(parts[4])) / 2.0
                                
                                enc_msg = Point()
                                enc_msg.x = left_avg
                                enc_msg.y = right_avg
                                enc_msg.z = 0.0
                                self.encoder_pub.publish(enc_msg)                             
                            except:
                                pass
                else:
                    self.serial_buffer += char
                    
        except Exception as e:
            self.get_logger().error(f"Serial read error: {e}")
    
    def on_plan(self, msg: String):
        text = msg.data.strip()
        self.get_logger().info(f"Received plan: {text}")
        
        pairs = parse_plan(text)
        if not pairs:
            self.get_logger().warning("Could not parse plan")
            return
        
        if self.batch_mode:
            parts = []
            for (m, d) in pairs:
                parts.append(str(int(m)))
                parts.append(str(float(d)))
            line = "B " + " ".join(parts) + "\n"
            sent = self._write_line(line)
            if sent:
                self.get_logger().info(f"Sent batch: {line.strip()}")
        else:
            for (m, d) in pairs:
                line = f"M {int(m)} {float(d):.2f} {float(self.default_speed):.2f}\n"
                sent = self._write_line(line)
                if sent:
                    self.get_logger().info(f"Sent: {line.strip()}")
                time.sleep(0.05)
    
    def _write_line(self, line: str) -> bool:
        if not self.ser:
            return False
        try:
            self.ser.write(line.encode('utf-8'))
            self.ser.flush()
            return True
        except Exception as e:
            self.get_logger().error(f"Serial write failed: {e}")
            return False

def main(args=None):
    rclpy.init(args=args)
    node = RobotControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down - stopping all motors")
        if node.ser:
            node.ser.write(b"STOP\n")
            node.ser.flush()
            time.sleep(0.1)
    finally:
        if node.ser:
            node.ser.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
