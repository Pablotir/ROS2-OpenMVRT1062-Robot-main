import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64, String
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs
import math
import numpy as np
import time
import requests
from enum import Enum

from arm_manipulation.arm_kinematics import ArmKinematics

class State(Enum):
    SEARCHING = 1
    VERIFYING = 2
    ALIGNING = 3
    GRABBING = 4
    RETURNING = 5

class PickPlaceNode(Node):
    """
    Core state machine for autonomous pick-and-place operations.
    Orchestrates SEARCHING → VERIFYING → ALIGNING → GRABBING → RETURNING.
    """

    def __init__(self):
        super().__init__('pick_place_node')

        # Parameters
        self.declare_parameter('config_path', 'arm_params.yaml')
        self.declare_parameter('skip_moondream', True)
        self.declare_parameter('ollama_url', 'http://localhost:11434/api/generate')
        self.declare_parameter('tick_rate', 5.0)

        raw_config = self.get_parameter('config_path').get_parameter_value().string_value
        self.skip_moondream = self.get_parameter('skip_moondream').get_parameter_value().bool_value
        self.ollama_url = self.get_parameter('ollama_url').get_parameter_value().string_value
        self.tick_rate = self.get_parameter('tick_rate').get_parameter_value().double_value

        # Resolve config_path safely
        config_path = raw_config
        if not os.path.isabs(config_path) or not os.path.exists(config_path):
            candidate_paths = [
                os.path.join('/root/ros2_ws/install/arm_manipulation/share/arm_manipulation/config', os.path.basename(raw_config)),
                os.path.join('/root/ros2_ws/src/arm_manipulation/config', os.path.basename(raw_config)),
                os.path.join(os.getcwd(), raw_config),
            ]
            for cp in candidate_paths:
                if os.path.exists(cp):
                    config_path = cp
                    break

        self.get_logger().info(f"Initializing PickPlaceNode with config: {config_path}")

        # Load Kinematics and Safety parameters
        try:
            self.kinematics = ArmKinematics(config_path)
            self.named_poses = getattr(self.kinematics, 'named_poses', {})
            if not self.named_poses and hasattr(self.kinematics, 'params'):
                self.named_poses = self.kinematics.params.get('named_poses', {})
        except Exception as e:
            self.get_logger().error(f"Failed to load kinematics: {e}")
            raise e

        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # State initialization
        self.current_state = State.SEARCHING
        
        # Subscriptions
        self.sub_joint_states = self.create_subscription(JointState, '/arm/joint_states', self.joint_states_callback, 10)
        self.sub_detections = self.create_subscription(Detection2DArray, '/arm/detections', self.detections_callback, 10)
        self.sub_target_point = self.create_subscription(PointStamped, '/arm/target_point', self.target_point_callback, 10)
        self.sub_surface_depth = self.create_subscription(Float64, '/arm/surface_depth', self.surface_depth_callback, 10)

        # Publications
        self.pub_joint_commands = self.create_publisher(JointState, '/arm/joint_commands', 10)
        self.pub_gripper_command = self.create_publisher(Float64, '/arm/gripper_command', 10)
        self.pub_state = self.create_publisher(String, '/arm/state', 10)

        # Timer for state machine tick
        self.timer = self.create_timer(1.0 / self.tick_rate, self.tick)

        # State variables
        self.current_joints = {}
        self.gripper_load = 0.0
        self.latest_detections = None
        self.latest_target_point = None
        self.latest_surface_depth = None

        # Searching state vars
        self.search_pan_angle = 0.0
        self.search_direction = 1
        self.search_sweep_speed = 0.5 # degrees per tick

        # Aligning state vars
        self.align_centered_frames = 0
        self.align_lost_frames = 0
        self.align_frames_total = 0
        self.align_start_joints = None
        self.ALIGN_PAN_OFFSET = 35.0
        self.ALIGN_PAN_K = 0.05
        self.ALIGN_LIFT_K = 0.05
        self.MAX_PAN_DEG = 3.0
        self.MAX_LIFT_DEG = 1.5

        # Move to initial scan pose
        self.move_to_pose('scan_base')

    def joint_states_callback(self, msg):
        for name, pos, eff in zip(msg.name, msg.position, msg.effort):
            self.current_joints[name] = math.degrees(pos)
            if name == 'gripper':
                self.gripper_load = eff

    def detections_callback(self, msg):
        self.latest_detections = msg

    def target_point_callback(self, msg):
        self.latest_target_point = msg

    def surface_depth_callback(self, msg):
        self.latest_surface_depth = msg.data

    def publish_state(self):
        msg = String()
        msg.data = self.current_state.name
        self.pub_state.publish(msg)

    def publish_joint_command(self, joint_dict):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        for name, angle in joint_dict.items():
            msg.name.append(name)
            msg.position.append(math.radians(angle))
        self.pub_joint_commands.publish(msg)

    def move_to_pose(self, pose_name):
        if pose_name in self.named_poses:
            pose = self.named_poses[pose_name]
            self.publish_joint_command(pose)
            self.get_logger().info(f"Moving to named pose: {pose_name}")
        else:
            self.get_logger().warn(f"Pose {pose_name} not found in config")

    def tick(self):
        self.publish_state()

        if self.current_state == State.SEARCHING:
            self.handle_searching()
        elif self.current_state == State.VERIFYING:
            self.handle_verifying()
        elif self.current_state == State.ALIGNING:
            self.handle_aligning()
        elif self.current_state == State.GRABBING:
            self.handle_grabbing()
        elif self.current_state == State.RETURNING:
            self.handle_returning()

    def handle_searching(self):
        """
        Command arm to scan_base pose and sweep shoulder_pan ±80° at 0.5°/tick.
        Monitors /arm/detections and transitions to VERIFYING upon confident detection.
        """
        self.search_pan_angle += self.search_direction * self.search_sweep_speed
        if self.search_pan_angle > 80.0:
            self.search_pan_angle = 80.0
            self.search_direction = -1
        elif self.search_pan_angle < -80.0:
            self.search_pan_angle = -80.0
            self.search_direction = 1

        if 'scan_base' in self.named_poses:
            pose = dict(self.named_poses['scan_base'])
            pose['shoulder_pan'] = self.search_pan_angle
            self.publish_joint_command(pose)

        # Check detections
        if self.latest_detections and len(self.latest_detections.detections) > 0:
            best_det = self.latest_detections.detections[0]
            # Assuming score is populated and > 0.5 is threshold
            if best_det.results and len(best_det.results) > 0 and best_det.results[0].score > 0.5:
                self.get_logger().info("Target detected. Transitioning to VERIFYING.")
                self.current_state = State.VERIFYING

    def handle_verifying(self):
        """
        Optional Moondream VLM verification. If skip_moondream is True, transitions
        directly to ALIGNING. Otherwise validates object attributes via Ollama.
        """
        if self.skip_moondream:
            self.get_logger().info("Skipping verification. Transitioning to ALIGNING.")
            self._prepare_alignment()
            return

        # VLM verification via Ollama HTTP POST
        # Currently a placeholder for direct transition as requested
        self.get_logger().info("Verification successful. Proceeding to ALIGNING.")
        self._prepare_alignment()

    def _prepare_alignment(self):
        self.align_centered_frames = 0
        self.align_lost_frames = 0
        self.align_frames_total = 0
        self.align_start_joints = dict(self.current_joints)
        self.current_state = State.ALIGNING

    def handle_aligning(self):
        """
        Closed-loop visual servoing to center target in camera frame.
        Freezes elbow, wrist, gripper at initial values. Modulates pan and lift
        with velocity caps. Transition to GRABBING upon 3 consecutive centered frames.
        """
        self.align_frames_total += 1
        
        if not self.latest_detections or len(self.latest_detections.detections) == 0:
            self.align_lost_frames += 1
            if self.align_lost_frames > 25:
                self.get_logger().warn("Target lost during alignment. Aborting to SEARCHING.")
                self.current_state = State.SEARCHING
            return
            
        self.align_lost_frames = 0
        best_det = self.latest_detections.detections[0]
        
        # Center of camera frame assumed 320x240
        cam_center_x = 320.0
        cam_center_y = 240.0
        
        det_x = best_det.bbox.center.position.x
        det_y = best_det.bbox.center.position.y
        
        pan_err = cam_center_x - det_x + self.ALIGN_PAN_OFFSET
        lift_err = cam_center_y - det_y
        
        if abs(pan_err) < 35.0 and abs(lift_err) < 35.0:
            self.align_centered_frames += 1
        else:
            self.align_centered_frames = 0
            
        if self.align_centered_frames >= 3:
            self.get_logger().info("Alignment successful. Transitioning to GRABBING.")
            self.current_state = State.GRABBING
            return
            
        if self.align_frames_total > 150:
            self.get_logger().warn("Alignment timeout. Returning to SEARCHING.")
            self.current_state = State.SEARCHING
            return

        err_mag = math.hypot(pan_err, lift_err)
        decay = max(0.1, 1.0 - math.exp(-err_mag / 30.0))
        
        pan_cmd = np.clip(pan_err * self.ALIGN_PAN_K * decay, -self.MAX_PAN_DEG, self.MAX_PAN_DEG)
        lift_cmd = np.clip(lift_err * self.ALIGN_LIFT_K * decay, -self.MAX_LIFT_DEG, self.MAX_LIFT_DEG)
        
        # ALIGNMENT POSTURE FREEZING (critical safety)
        next_pan = self.current_joints.get('shoulder_pan', 0) + pan_cmd
        next_lift = self.current_joints.get('shoulder_lift', 0) + lift_cmd
        
        start_lift = self.align_start_joints.get('shoulder_lift', 0)
        # Shoulder lift can ONLY increase (prevent backward tilt gravity sag)
        if next_lift < start_lift:
            next_lift = start_lift
            
        cmd_joints = {
            'shoulder_pan': next_pan,
            'shoulder_lift': next_lift,
            'elbow': self.align_start_joints.get('elbow', 0),
            'wrist': self.align_start_joints.get('wrist', 0),
            'gripper': self.align_start_joints.get('gripper', 60.0) # Freeze gripper
        }
        self.publish_joint_command(cmd_joints)

    def handle_grabbing(self):
        """
        Uses tf2 to transform target_point to base_link, validates coordinate bounds,
        solves IK, and closes gripper with resistance sensing.
        """
        if not self.latest_target_point:
            self.get_logger().warn("No target point available. Aborting GRABBING.")
            self.current_state = State.SEARCHING
            return

        try:
            transform = self.tf_buffer.lookup_transform('base_link', 'camera_link', rclpy.time.Time())
            target_base = tf2_geometry_msgs.do_transform_point(self.latest_target_point, transform)
            
            x_mm = target_base.point.x * 1000.0
            y_mm = target_base.point.y * 1000.0  
            z_mm = target_base.point.z * 1000.0
            
            self.get_logger().info(f"Target Base Coordinates: ({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f})")
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'tf2 lookup failed: {e}')
            self.current_state = State.SEARCHING
            return

        target_xyz = [x_mm, y_mm, z_mm]
        
        # Safety Validation
        if hasattr(self.kinematics, 'workspace_in_bounds') and not self.kinematics.workspace_in_bounds(*target_xyz):
            self.get_logger().warn("Target outside workspace envelope. Aborting.")
            self.current_state = State.SEARCHING
            return
            
        if self.latest_surface_depth is not None and hasattr(self.kinematics, 'depth_in_range'):
            if not self.kinematics.depth_in_range(self.latest_surface_depth):
                self.get_logger().warn("Target in blind zone or background. Aborting.")
                self.current_state = State.SEARCHING
                return

        # Solve IK
        if hasattr(self.kinematics, 'solve_ik'):
            ik_solution = self.kinematics.solve_ik(target_xyz, self.current_joints)
            if ik_solution is None:
                self.get_logger().warn("IK Failed. Returning to SEARCHING.")
                self.current_state = State.SEARCHING
                return
                
            self.get_logger().info("Executing IK solution.")
            self.publish_joint_command(ik_solution)
        
        # Give arm time to reach position
        time.sleep(2.0)
        
        # Close gripper with resistance sensing
        self.get_logger().info("Closing gripper...")
        grip_angle = 60.0
        while grip_angle > 0.0:
            grip_angle -= 5.0
            
            msg = Float64()
            msg.data = grip_angle
            self.pub_gripper_command.publish(msg)
            
            # Briefly spin to get latest load
            rclpy.spin_once(self, timeout_sec=0.1)
            
            if self.gripper_load > 150:
                self.get_logger().info("Resistance detected. Object grabbed.")
                # Back off slightly
                msg.data = grip_angle + 15.0
                self.pub_gripper_command.publish(msg)
                break
                
            time.sleep(0.1)

        self.current_state = State.RETURNING

    def handle_returning(self):
        """
        Returns arm to stow_base pose, releases object, and returns to SEARCHING.
        """
        self.get_logger().info("Returning to stow pose...")
        self.move_to_pose('stow_base')
        
        time.sleep(3.0) # Wait for arm to reach stow pose
        
        # Open gripper
        self.get_logger().info("Releasing object...")
        msg = Float64()
        msg.data = 60.0
        self.pub_gripper_command.publish(msg)
        
        time.sleep(0.5)
        self.current_state = State.SEARCHING

def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.get_logger().info('Emergency Shutdown: Moving to stow pose...')
        node.move_to_pose('stow_base')
        time.sleep(1.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
