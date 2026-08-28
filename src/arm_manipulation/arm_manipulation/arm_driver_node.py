#!/usr/bin/env python3
import time
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from std_srvs.srv import SetBool

# LeRobot imports
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

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
        self.joint_keys = [f'{name}.pos' for name in self.joint_names]

        # Initialize arm connection
        self.get_logger().info(f"Connecting to SO-ARM101 on {self.arm_port}...")
        self.robot = None
        try:
            config = SOFollowerRobotConfig(
                robot_id=self.arm_id,
                port=self.arm_port
            )
            self.robot = SOFollower(config)
            self.robot.connect()
            self.get_logger().info("Successfully connected to the arm.")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to arm on {self.arm_port}: {e}")
            raise

        # Current state
        self.current_joint_positions = {name: 0.0 for name in self.joint_names}

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
        if not self.robot:
            return

        try:
            observation = self.robot.get_observation()
            
            # Read effort (Present_Load)
            # Depending on LeRobot version, bus.read might take a list of joint names.
            efforts = self.robot.bus.read('Present_Load', self.joint_names)

            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joint_names
            msg.position = []
            msg.effort = []

            for i, name in enumerate(self.joint_names):
                pos = observation.get(f'{name}.pos', 0.0)
                self.current_joint_positions[name] = pos
                msg.position.append(pos)
                
                # Assuming efforts list aligns with joint_names
                if efforts and len(efforts) > i:
                    msg.effort.append(float(efforts[i]))
                else:
                    msg.effort.append(0.0)

            self.joint_state_pub.publish(msg)

        except Exception as e:
            self.get_logger().warning(f"Error reading arm state: {e}")

    def joint_command_callback(self, msg: JointState):
        if not self.robot:
            return
            
        target_positions = {}
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                target_positions[name] = msg.position[i]
                
        if not target_positions:
            return

        self.smooth_move(target_positions)

    def gripper_command_callback(self, msg: Float64):
        # 0 = closed, 60 = open (based on prompt)
        target_pos = msg.data
        self.close_gripper_with_resistance(target_pos=target_pos)

    def set_torque_callback(self, request, response):
        if not self.robot:
            response.success = False
            response.message = "Robot not connected."
            return response

        try:
            enable_val = 1 if request.data else 0
            # Assuming write method accepts list of joint names
            self.robot.bus.write('Torque_Enable', enable_val, self.joint_names)
            response.success = True
            response.message = f"Torque enabled set to {request.data}"
        except Exception as e:
            response.success = False
            response.message = f"Failed to set torque: {e}"
        return response

    def smooth_move(self, target_positions, step_size=2.0, step_delay=0.03, hold_joints=None):
        if not self.robot:
            return

        if hold_joints is None:
            hold_joints = []

        start_positions = {name: self.current_joint_positions[name] for name in self.joint_names}
        
        # Override start positions with target positions for hold_joints
        for joint in hold_joints:
            if joint in start_positions:
                target_positions[joint] = start_positions[joint]

        # Fill missing targets with current positions
        for name in self.joint_names:
            if name not in target_positions:
                target_positions[name] = start_positions[name]

        # Compute max delta
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

            self.robot.send_action(action)
            time.sleep(step_delay)

    def close_gripper_with_resistance(self, target_pos=0.0, load_threshold=150, timeout=1.5, backoff_deg=15.0):
        """Close gripper until load resistance > threshold, then back off."""
        if not self.robot:
            return False

        start = time.time()
        current_g = self.current_joint_positions.get('gripper', target_pos)
        step = -2.0  # degrees per step

        while current_g > target_pos and (time.time() - start) < timeout:
            current_g += step
            current_g = max(current_g, target_pos)
            
            action = {'gripper.pos': current_g}
            # Add other joints to maintain position if required by send_action
            for name in self.joint_names:
                if name != 'gripper':
                    action[f'{name}.pos'] = self.current_joint_positions[name]
                    
            self.robot.send_action(action)
            self.current_joint_positions['gripper'] = current_g
            time.sleep(0.05)
            
            try:
                load = self.robot.bus.read('Present_Load', ['gripper'])
                if load and abs(load[0]) > load_threshold:
                    current_g = max(current_g - backoff_deg, target_pos)
                    # Back off action
                    action['gripper.pos'] = current_g
                    self.robot.send_action(action)
                    self.current_joint_positions['gripper'] = current_g
                    self.get_logger().info(f"Grip threshold reached. Backing off to {current_g}")
                    return True  # Got grip
            except Exception as e:
                self.get_logger().warning(f"Failed to read gripper load: {e}")

        return False  # Timeout or reached target without hitting load threshold

    def destroy_node(self):
        if self.robot:
            self.get_logger().info("Shutting down. Moving to stow pose...")
            stow_target = {
                'shoulder_pan': self.current_joint_positions.get('shoulder_pan', 0.0), # keep current pan
                'shoulder_lift': -104.5,
                'elbow_flex': 96.5,
                'wrist_flex': 0.0,
                'wrist_roll': self.current_joint_positions.get('wrist_roll', 0.0),
                'gripper': self.current_joint_positions.get('gripper', 60.0) # open gripper
            }
            try:
                self.smooth_move(stow_target, step_size=2.0, step_delay=0.03)
                time.sleep(0.5)
                self.robot.disconnect()
            except Exception as e:
                self.get_logger().error(f"Error during shutdown sequence: {e}")
                
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
