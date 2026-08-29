#!/usr/bin/env python3
"""
ovmm_supervisor_node.py — 5-Layer Autonomous OVMM Execution Pipeline
====================================================================
Implements the full Open-Vocabulary Mobile Manipulation supervisor:

  Layer 1: Scene Graphing & Room Centroids (via SceneGraphDB)
  Layer 2: Task Routing & Opportunistic Visual Preemption (Nav2 + D405 YOLO)
  Layer 3: Kinematic Docking & Visual Servoing (~0.4m Standoff)
  Layer 4: Modular Skill Execution (PyTorch / LeRobot Policy Library)
  Layer 5: Gripper Verification & Low-Profile Stow Return Home

Prompt Example:
  "Grab the blue bottle in the kitchen and throw it in the bin"
"""

import math
import time
from enum import Enum
from typing import Optional, Dict, Any, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist, PointStamped, PoseStamped
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import String, Float64, Bool
from vision_msgs.msg import Detection2DArray
from std_srvs.srv import Trigger
from nav2_msgs.action import NavigateToPose

from arm_manipulation.skill_registry import SkillExecutor, JOINT_NAMES
from .scene_graph_db import SceneGraphDB
from .map_manager import get_maps_base_dir


class OVMMState(Enum):
    IDLE = 0
    TASK_ROUTING = 1
    OPPORTUNISTIC_PREEMPTION = 2
    KINEMATIC_DOCKING = 3
    POLICY_EXECUTION = 4
    VERIFY_AND_STOW = 5
    RETURNING_HOME = 6
    TASK_COMPLETE = 7
    ERROR_RECOVERY = 8


# Calibrated reference poses (from calibration/arm_reference_poses.yaml)
CALIBRATED_STOW_BASE = {
    'shoulder_pan': -4.48,
    'shoulder_lift': -106.11,
    'elbow_flex': 100.00,
    'wrist_flex': 75.96,
    'wrist_roll': -156.75,
    'gripper': 73.77
}

CALIBRATED_SCAN_BASE = {
    'shoulder_pan': -4.48,
    'shoulder_lift': -106.02,
    'elbow_flex': 99.91,
    'wrist_flex': 33.41,
    'wrist_roll': -155.96,
    'gripper': 73.84
}


class OVMMSupervisorNode(Node):
    """Main state machine orchestrating 5-layer mobile manipulation."""

    def __init__(self):
        super().__init__('ovmm_supervisor_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('maps_dir', '/root/maps')
        self.declare_parameter('preemption_confidence_thresh', 0.75)
        self.declare_parameter('preemption_consecutive_frames', 3)
        self.declare_parameter('docking_standoff_m', 0.40)
        self.declare_parameter('control_rate_hz', 20.0)

        self._maps_dir = get_maps_base_dir(self.get_parameter('maps_dir').value)
        self._conf_thresh = float(self.get_parameter('preemption_confidence_thresh').value)
        self._consec_frames_req = int(self.get_parameter('preemption_consecutive_frames').value)
        self._standoff_m = float(self.get_parameter('docking_standoff_m').value)
        self._control_hz = float(self.get_parameter('control_rate_hz').value)

        # Database & Skill Executor
        self._db = SceneGraphDB(f"{self._maps_dir}/scene_graph.db")
        self._skill_executor = SkillExecutor()

        # State tracking
        self._state = OVMMState.IDLE
        self._state_start_t = time.monotonic()

        # Active Task Metadata
        self._target_object: str = 'bottle'
        self._target_room: str = 'kitchen'
        self._action_verb: str = 'grab'
        self._destination_room: Optional[str] = None

        # Visual Preemption Filter
        self._consecutive_detections: int = 0
        self._latest_target_point: Optional[PointStamped] = None
        self._latest_depth_standoff: float = 1.0

        # Arm & Base Feedback
        self._current_joints: Dict[str, float] = dict(CALIBRATED_STOW_BASE)
        self._gripper_load: float = 0.0
        self._latest_wrist_img = None

        # ── Action Clients ────────────────────────────────────────────────────
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._active_nav_goal = None

        # ── Publishers ────────────────────────────────────────────────────────
        self._cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._arm_cmd_pub = self.create_publisher(JointState, '/arm/joint_commands', 10)
        self._gripper_cmd_pub = self.create_publisher(Float64, '/arm/gripper_command', 10)
        self._status_pub = self.create_publisher(String, '/ovmm/status', 10)
        self._target_class_pub = self.create_publisher(String, '/arm/set_target', 10)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(String, '/ovmm/task_prompt', self._on_task_prompt, 10)
        self.create_subscription(JointState, '/arm/joint_states', self._on_joint_states, 10)
        self.create_subscription(Detection2DArray, '/arm/detections', self._on_detections, 10)
        self.create_subscription(PointStamped, '/arm/target_point', self._on_target_point, 10)

        # ── Service Clients ───────────────────────────────────────────────────
        self._arm_mask_disable_cli = self.create_client(Trigger, '/arm_mask/disable')
        self._arm_mask_enable_cli = self.create_client(Trigger, '/arm_mask/enable')

        # ── Services ──────────────────────────────────────────────────────────
        self.create_service(Trigger, '/ovmm/cancel_task', self._srv_cancel_task)

        # Main Control Loop Timer
        self.create_timer(1.0 / self._control_hz, self._control_loop)

        self.get_logger().info(
            f"OVMM Supervisor ready | Control rate: {self._control_hz} Hz | Standby for natural language task prompt.")

    # ── Task Prompt Parser ──────────────────────────────────────────────────

    def _on_task_prompt(self, msg: String):
        """Parse natural language task string into action verb, target object, and target room."""
        prompt = msg.data.lower().strip()
        self.get_logger().info(f"Received Natural Language Task Prompt: '{prompt}'")

        # Parse verbs
        verb = 'grab'
        for v in ['throw', 'flip', 'place', 'pick', 'grab']:
            if v in prompt:
                verb = v
                break

        # Parse target room from scene graph DB taxonomy
        room = 'kitchen'
        for r in ['kitchen', 'living_room', 'living room', 'bedroom', 'bathroom', 'office', 'hallway']:
            if r in prompt:
                room = r.replace(' ', '_')
                break

        # Parse target object keywords
        obj = 'bottle'
        for o in ['bottle', 'cup', 'can', 'box', 'bowl', 'apple', 'banana', 'remote']:
            if o in prompt:
                obj = o
                break

        self.start_task(action_verb=verb, target_object=obj, target_room=room)

    def start_task(self, action_verb: str, target_object: str, target_room: str):
        """Initiate autonomous OVMM pipeline for target task."""
        self._action_verb = action_verb
        self._target_object = target_object
        self._target_room = target_room
        self._consecutive_detections = 0

        # Configure detection node target
        t_msg = String()
        t_msg.data = target_object
        self._target_class_pub.publish(t_msg)

        # Load manipulation skill
        self._skill_executor.load_skill(action_verb)

        # Transition to Layer 2: Task Routing
        self._set_state(OVMMState.TASK_ROUTING)

    # ── Main State Machine Control Loop ─────────────────────────────────────

    def _control_loop(self):
        # Publish status string
        status_msg = String()
        status_msg.data = f"STATE: {self._state.name} | TASK: {self._action_verb} '{self._target_object}' in {self._target_room}"
        self._status_pub.publish(status_msg)

        if self._state == OVMMState.IDLE:
            return

        elif self._state == OVMMState.TASK_ROUTING:
            # 1. Deploy arm to scan_base pose for forward D405 scouting
            self._send_arm_pose(CALIBRATED_SCAN_BASE)

            # 2. Dispatch Nav2 goal to target room centroid
            region = self._db.find_region_by_label(self._target_room)
            if region is not None:
                cx, cy = region['centroid_x'], region['centroid_y']
                self._dispatch_nav2_goal(cx, cy)
                self._set_state(OVMMState.OPPORTUNISTIC_PREEMPTION)
            else:
                self.get_logger().warn(f"Room '{self._target_room}' not found in Scene Graph! Defaulting to search.")
                self._set_state(OVMMState.OPPORTUNISTIC_PREEMPTION)

        elif self._state == OVMMState.OPPORTUNISTIC_PREEMPTION:
            # Check if target object was detected with high confidence for N consecutive frames
            if self._consecutive_detections >= self._consec_frames_req:
                self.get_logger().info(
                    f"🎯 PREEMPTION TRIGGERED! Target '{self._target_object}' spotted en route! Cancelling Nav2 goal.")
                self._cancel_active_nav_goal()
                self._set_state(OVMMState.KINEMATIC_DOCKING)

        elif self._state == OVMMState.KINEMATIC_DOCKING:
            # Visual servoing micro-alignment to standoff distance (~0.4m)
            is_docked = self._execute_visual_servoing_docking()
            if is_docked:
                self._stop_base()
                self.get_logger().info(
                    f"Docking complete at {self._latest_depth_standoff:.2f}m standoff. Beginning manipulation.")
                # Disable LiDAR arm masking for grab
                self._call_trigger_service(self._arm_mask_disable_cli)
                self._set_state(OVMMState.POLICY_EXECUTION)

        elif self._state == OVMMState.POLICY_EXECUTION:
            # Execute skill policy at 30 Hz
            target_3d = (
                self._latest_target_point.point.x,
                self._latest_target_point.point.y,
                self._latest_target_point.point.z
            ) if self._latest_target_point else None

            action_joints, is_done = self._skill_executor.step(
                wrist_rgb=self._latest_wrist_img,
                current_joint_angles=self._current_joints,
                target_3d_rel=target_3d
            )
            self._send_arm_pose(action_joints)

            if is_done:
                self.get_logger().info("Skill policy execution trajectory complete. Verifying grasp.")
                self._set_state(OVMMState.VERIFY_AND_STOW)

        elif self._state == OVMMState.VERIFY_AND_STOW:
            # Layer 5: Verify grasp from gripper feedback
            gripper_pos = self._current_joints.get('gripper', 0.0)
            if gripper_pos < 5.0:
                self.get_logger().warn("Grasp verification failed: Gripper fully closed (empty grasp).")
            else:
                self.get_logger().info(f"Grasp verified! Gripper holding object at angle {gripper_pos:.1f}°.")

            # Fold arm into low-profile stow pose
            self._send_arm_pose(CALIBRATED_STOW_BASE)

            # Re-enable LiDAR arm masking
            self._call_trigger_service(self._arm_mask_enable_cli)

            # Dispatch Nav2 Return Home (0,0,0)
            self.get_logger().info("Returning robot to Home base origin (0, 0, 0)...")
            self._dispatch_nav2_goal(0.0, 0.0)
            self._set_state(OVMMState.RETURNING_HOME)

        elif self._state == OVMMState.RETURNING_HOME:
            # Navigation back to home in progress
            pass

        elif self._state == OVMMState.TASK_COMPLETE:
            self._stop_base()

    # ── Visual Servoing Kinematic Docking ───────────────────────────────────

    def _execute_visual_servoing_docking(self) -> bool:
        """Visual servoing centering target in D405 optical frame using mecanum /cmd_vel."""
        if self._latest_target_point is None:
            # If target temporarily lost, gentle rotation search
            twist = Twist()
            twist.angular.z = 0.15
            self._cmd_vel_pub.publish(twist)
            return False

        tx = self._latest_target_point.point.x  # lateral error in optical frame (m)
        ty = self._latest_target_point.point.y  # vertical error (m)
        tz = self._latest_target_point.point.z  # forward depth standoff (m)

        self._latest_depth_standoff = tz

        # Proportional controller gains
        K_LATERAL = 0.60
        K_FORWARD = 0.50
        K_YAW = 0.80

        depth_error = tz - self._standoff_m

        twist = Twist()

        # Forward velocity (approach until standoff ~0.4m)
        if abs(depth_error) > 0.04:
            twist.linear.x = float(np.clip(K_FORWARD * depth_error, -0.10, 0.15))

        # Lateral strafe velocity (mecanum omnidirectional centering)
        if abs(tx) > 0.03:
            twist.linear.y = float(np.clip(-K_LATERAL * tx, -0.12, 0.12))

        # Yaw alignment correction
        if abs(tx) > 0.05:
            twist.angular.z = float(np.clip(-K_YAW * tx, -0.25, 0.25))

        self._cmd_vel_pub.publish(twist)

        # Check alignment convergence (within 4cm of standoff and 3cm of center)
        if abs(depth_error) <= 0.05 and abs(tx) <= 0.04:
            return True

        return False

    # ── Feedback Callbacks ──────────────────────────────────────────────────

    def _on_joint_states(self, msg: JointState):
        for name, pos, eff in zip(msg.name, msg.position, msg.effort):
            self._current_joints[name] = math.degrees(pos)
            if name == 'gripper':
                self._gripper_load = eff

    def _on_detections(self, msg: Detection2DArray):
        found = False
        for det in msg.detections:
            for hyp in det.results:
                score = hyp.hypothesis.score if hasattr(hyp, 'hypothesis') else hyp.score
                if score >= self._conf_thresh:
                    found = True
                    break

        if found:
            self._consecutive_detections += 1
        else:
            self._consecutive_detections = 0

    def _on_target_point(self, msg: PointStamped):
        self._latest_target_point = msg

    # ── Helpers & Action / Service Dispatch ─────────────────────────────────

    def _send_arm_pose(self, joint_dict: Dict[str, float]):
        """Publish joint command to Feetech STS3215 servo driver."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        for name, deg in joint_dict.items():
            msg.name.append(name)
            msg.position.append(math.radians(deg))
        self._arm_cmd_pub.publish(msg)

    def _stop_base(self):
        self._cmd_vel_pub.publish(Twist())

    def _set_state(self, new_state: OVMMState):
        self.get_logger().info(f"OVMM Pipeline Transition: {self._state.name} -> {new_state.name}")
        self._state = new_state
        self._state_start_t = time.monotonic()

    def _dispatch_nav2_goal(self, x: float, y: float):
        """Send NavigateToPose action goal to Nav2."""
        if not self._nav_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().warn("Nav2 action server not available.")
            return

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation.w = 1.0

        future = self._nav_client.send_goal_async(goal)
        future.add_done_callback(self._on_nav_goal_response)

    def _on_nav_goal_response(self, future):
        goal_handle = future.result()
        if goal_handle.accepted:
            self._active_nav_goal = goal_handle
            self.get_logger().info("Nav2 goal accepted.")

    def _cancel_active_nav_goal(self):
        if self._active_nav_goal is not None:
            self._active_nav_goal.cancel_goal_async()
            self._active_nav_goal = None
            self.get_logger().info("Nav2 goal cancelled successfully.")

    def _call_trigger_service(self, client):
        if client.wait_for_service(timeout_sec=1.0):
            req = Trigger.Request()
            client.call_async(req)

    def _srv_cancel_task(self, request, response):
        self._cancel_active_nav_goal()
        self._stop_base()
        self._send_arm_pose(CALIBRATED_STOW_BASE)
        self._set_state(OVMMState.IDLE)
        response.success = True
        response.message = "OVMM task cancelled."
        return response


def main(args=None):
    rclpy.init(args=args)
    node = OVMMSupervisorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_base()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
