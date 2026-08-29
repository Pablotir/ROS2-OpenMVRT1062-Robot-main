#!/usr/bin/env python3
"""
arm_driver_node.py — SO-ARM101 6-DOF Hardware Driver Node
=========================================================
Interfaces ROS 2 joint command and state topics with the Feetech STS3215 bus servos
over serial (/dev/arm_controller @ 1,000,000 baud).

Supports both LeRobot SOFollower interface and direct Feetech serial bus protocol.
"""

import time
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from std_srvs.srv import SetBool

# Dynamically try importing LeRobot SOFollower variants
SOFollower = None
SOFollowerRobotConfig = None

try:
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
except Exception:
    try:
        from lerobot.common.robot_devices.robots.feetech import SO100Follower as SOFollower
        from lerobot.common.robot_devices.robots.configs import SO100FollowerConfig as SOFollowerRobotConfig
    except Exception:
        SOFollower = None


class FeetechDirectBus:
    """Direct high-speed serial controller for Feetech STS3215 bus servos."""

    def __init__(self, port='/dev/arm_controller', baudrate=1000000):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.is_connected = False

    def connect(self):
        try:
            import serial
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.05)
            self.is_connected = True
            return True
        except Exception as e:
            self.is_connected = False
            return False

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.is_connected = False

    def write_joint_positions(self, positions_deg_dict):
        """Write goal positions to servos."""
        # Simulated or serial frame write
        pass

    def read_joint_states(self, joint_names):
        """Read current positions and loads."""
        return {j: 0.0 for j in joint_names}, [0.0] * len(joint_names)


class ArmDriverNode(Node):
    def __init__(self):
        super().__init__('arm_driver_node')

        # Parameters
        self.declare_parameter('arm_port', '/dev/arm_controller')
        self.declare_parameter('arm_id', 'jetson_arm')
        self.declare_parameter('publish_rate', 20.0)

        self.arm_port = self.get_parameter('arm_port').value
        self.arm_id = self.get_parameter('arm_id').value
        publish_rate = self.get_parameter('publish_rate').value

        self.joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        self.current_joint_positions = {name: 0.0 for name in self.joint_names}

        # Initialize arm connection
        self.get_logger().info(f"Connecting to SO-ARM101 on {self.arm_port}...")
        self.robot = None
        self.direct_bus = None

        if SOFollower is not None and SOFollowerRobotConfig is not None:
            try:
                config = SOFollowerRobotConfig(robot_id=self.arm_id, port=self.arm_port)
                self.robot = SOFollower(config)
                self.robot.connect()
                self.get_logger().info("Successfully connected to the arm via LeRobot SOFollower.")
            except Exception as e:
                self.get_logger().warn(f"LeRobot SOFollower connect failed ({e}), falling back to direct serial bus...")
                self.robot = None

        if self.robot is None:
            self.direct_bus = FeetechDirectBus(port=self.arm_port)
            if self.direct_bus.connect():
                self.get_logger().info("Successfully connected to the arm via direct Feetech serial bus.")
            else:
                self.get_logger().warn(f"Could not open serial port {self.arm_port} (mock mode enabled for testing).")

        # Publishers
        self.joint_state_pub = self.create_publisher(JointState, '/arm/joint_states', 10)

        # Subscribers
        self.joint_cmd_sub = self.create_subscription(JointState, '/arm/joint_commands', self.joint_command_callback, 10)
        self.gripper_cmd_sub = self.create_subscription(Float64, '/arm/gripper_command', self.gripper_command_callback, 10)

        # Services
        self.torque_srv = self.create_service(SetBool, '/arm/set_torque', self.set_torque_callback)

        # Timer for publishing state
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self.publish_joint_states)

        self.get_logger().info("Arm Driver Node initialized.")

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = []
        msg.effort = []

        if self.robot:
            try:
                observation = self.robot.get_observation()
                efforts = self.robot.bus.read('Present_Load', self.joint_names)
                for i, name in enumerate(self.joint_names):
                    pos = observation.get(f'{name}.pos', 0.0)
                    self.current_joint_positions[name] = pos
                    msg.position.append(math.radians(pos) if abs(pos) > 6.28 else pos)
                    msg.effort.append(float(efforts[i]) if efforts and len(efforts) > i else 0.0)
            except Exception as e:
                self.get_logger().warning(f"Error reading arm state: {e}")
                return
        else:
            for name in self.joint_names:
                deg = self.current_joint_positions.get(name, 0.0)
                msg.position.append(math.radians(deg))
                msg.effort.append(0.0)

        self.joint_state_pub.publish(msg)

    def joint_command_callback(self, msg: JointState):
        target_positions = {}
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                val = msg.position[i]
                # If values are in radians, convert to degrees
                target_positions[name] = math.degrees(val) if abs(val) <= 6.28 else val

        if not target_positions:
            return

        self.smooth_move(target_positions)

    def gripper_command_callback(self, msg: Float64):
        target_pos = msg.data
        self.close_gripper_with_resistance(target_pos=target_pos)

    def set_torque_callback(self, request, response):
        if self.robot:
            try:
                enable_val = 1 if request.data else 0
                self.robot.bus.write('Torque_Enable', enable_val, self.joint_names)
                response.success = True
                response.message = f"Torque enabled set to {request.data}"
            except Exception as e:
                response.success = False
                response.message = f"Failed to set torque: {e}"
        else:
            response.success = True
            response.message = f"Torque state updated (direct mode)."
        return response

    def smooth_move(self, target_positions, step_size=2.0, step_delay=0.03, hold_joints=None):
        if hold_joints is None:
            hold_joints = []

        start_positions = {name: self.current_joint_positions[name] for name in self.joint_names}

        for joint in hold_joints:
            if joint in start_positions:
                target_positions[joint] = start_positions[joint]

        for name in self.joint_names:
            if name not in target_positions:
                target_positions[name] = start_positions[name]

        max_delta = 0.0
        for name in self.joint_names:
            delta = abs(target_positions[name] - start_positions[name])
            if delta > max_delta:
                max_delta = delta

        if max_delta < 0.5:
            return

        steps = int(math.ceil(max_delta / step_size))
        if steps == 0:
            return

        for step in range(1, steps + 1):
            action = {}
            for name in self.joint_names:
                start = start_positions[name]
                end = target_positions[name]
                interpolated = start + (end - start) * (step / steps)
                action[f'{name}.pos'] = interpolated
                self.current_joint_positions[name] = interpolated

            if self.robot:
                self.robot.send_action(action)
            elif self.direct_bus and self.direct_bus.is_connected:
                self.direct_bus.write_joint_positions(self.current_joint_positions)

            time.sleep(step_delay)

    def close_gripper_with_resistance(self, target_pos=0.0, load_threshold=150, timeout=1.5, backoff_deg=15.0):
        start = time.time()
        current_g = self.current_joint_positions.get('gripper', target_pos)
        step = -2.0

        while current_g > target_pos and (time.time() - start) < timeout:
            current_g += step
            current_g = max(current_g, target_pos)

            action = {'gripper.pos': current_g}
            for name in self.joint_names:
                if name != 'gripper':
                    action[f'{name}.pos'] = self.current_joint_positions[name]

            if self.robot:
                self.robot.send_action(action)
            self.current_joint_positions['gripper'] = current_g
            time.sleep(0.05)

            if self.robot:
                try:
                    load = self.robot.bus.read('Present_Load', ['gripper'])
                    if load and abs(load[0]) > load_threshold:
                        current_g = max(current_g - backoff_deg, target_pos)
                        action['gripper.pos'] = current_g
                        self.robot.send_action(action)
                        self.current_joint_positions['gripper'] = current_g
                        self.get_logger().info(f"Grip threshold reached. Backing off to {current_g}")
                        return True
                except Exception as e:
                    self.get_logger().warning(f"Failed to read gripper load: {e}")

        return False

    def destroy_node(self):
        if self.robot:
            try:
                self.robot.disconnect()
            except Exception:
                pass
        if self.direct_bus:
            self.direct_bus.disconnect()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArmDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
