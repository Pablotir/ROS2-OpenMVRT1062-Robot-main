#!/usr/bin/env python3
"""
skill_registry.py — Modular OVMM Skill Library & Policy Runner
==============================================================
Implements Layer 4 of the Modular OVMM framework:
  Maps natural language action verbs ('grab', 'throw', 'flip', 'place')
  to lightweight LeRobot PyTorch policy checkpoints (ACT or Diffusion Policy).
  
Bypasses classical inverse kinematics (IK) entirely during manipulation,
streaming joint actions directly to the Feetech STS3215 bus servos at 30-50 Hz.

Trained offline from WH148 leader-arm teleoperation demonstrations.
"""

import os
import math
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple, List


# Default skill definitions & checkpoint mappings
SKILL_REGISTRY: Dict[str, Dict[str, Any]] = {
    'grab': {
        'description': 'Zero-shot tabletop / floor grasping policy',
        'checkpoint_path': '/root/ros2_ws/models/skills/grab_act.pt',
        'policy_type': 'ACT',
        'standoff_distance_m': 0.40,
        'gripper_action': 'close',
    },
    'pick': {
        'description': 'Alias for grab policy',
        'checkpoint_path': '/root/ros2_ws/models/skills/grab_act.pt',
        'policy_type': 'ACT',
        'standoff_distance_m': 0.40,
        'gripper_action': 'close',
    },
    'throw': {
        'description': 'Dynamic toss into waste bin / container',
        'checkpoint_path': '/root/ros2_ws/models/skills/throw_act.pt',
        'policy_type': 'ACT',
        'standoff_distance_m': 0.50,
        'gripper_action': 'open_dynamic',
    },
    'flip': {
        'description': 'Flip target object orientation',
        'checkpoint_path': '/root/ros2_ws/models/skills/flip_act.pt',
        'policy_type': 'ACT',
        'standoff_distance_m': 0.35,
        'gripper_action': 'sequence',
    },
    'place': {
        'description': 'Gentle surface placement policy',
        'checkpoint_path': '/root/ros2_ws/models/skills/place_act.pt',
        'policy_type': 'ACT',
        'standoff_distance_m': 0.40,
        'gripper_action': 'open',
    },
}

JOINT_NAMES = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper'
]


class SkillExecutor:
    """Manages loading and executing PyTorch manipulation policies."""

    def __init__(self, models_dir: str = '/root/ros2_ws/models/skills'):
        self.models_dir = models_dir
        self._active_policy = None
        self._active_verb: Optional[str] = None
        self._policy_loaded = False
        self._step_counter = 0

    def get_available_skills(self) -> List[str]:
        """Return list of supported action verbs."""
        return list(SKILL_REGISTRY.keys())

    def get_skill_info(self, verb: str) -> Optional[Dict[str, Any]]:
        """Fetch specification metadata for a given action verb."""
        norm_verb = verb.lower().strip()
        return SKILL_REGISTRY.get(norm_verb)

    def load_skill(self, verb: str) -> bool:
        """
        Load policy checkpoint for the requested action verb.
        Falls back to scripted compliant trajectory if PyTorch checkpoint
        is not yet present on disk.
        """
        norm_verb = verb.lower().strip()
        if norm_verb not in SKILL_REGISTRY:
            return False

        spec = SKILL_REGISTRY[norm_verb]
        ckpt_path = spec['checkpoint_path']
        self._active_verb = norm_verb
        self._step_counter = 0

        if os.path.exists(ckpt_path):
            try:
                import torch
                self._active_policy = torch.jit.load(ckpt_path) if ckpt_path.endswith('.ts') else torch.load(ckpt_path)
                self._policy_loaded = True
                return True
            except Exception:
                self._policy_loaded = False
                return True
        else:
            # Fallback compliant scripted trajectory mode enabled
            self._policy_loaded = False
            return True

    def step(self,
             wrist_rgb: np.ndarray,
             current_joint_angles: Dict[str, float],
             target_3d_rel: Optional[Tuple[float, float, float]] = None) -> Tuple[Dict[str, float], bool]:
        """
        Execute one policy inference step (30-50 Hz).
        
        Args:
          wrist_rgb: BGR image crop from RealSense D405 (H, W, 3)
          current_joint_angles: Current Feetech joint states in degrees
          target_3d_rel: Relative 3D target coordinates [x, y, z] in camera frame
          
        Returns:
          action_joints: Dict of joint commands in degrees
          is_complete: True when the action trajectory has finished
        """
        self._step_counter += 1

        # 1. PyTorch Policy Inference Mode (if trained checkpoint loaded)
        if self._policy_loaded and self._active_policy is not None:
            try:
                import torch
                # Preprocess image crop & joint states into observation tensor
                img_tensor = torch.from_numpy(wrist_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                state_vec = torch.tensor([
                    math.radians(current_joint_angles.get(j, 0.0)) for j in JOINT_NAMES
                ]).float().unsqueeze(0)

                with torch.no_grad():
                    action_tensor = self._active_policy(img_tensor, state_vec)
                    action_rad = action_tensor.squeeze(0).cpu().numpy()

                action_deg = {
                    j: math.degrees(action_rad[i]) for i, j in enumerate(JOINT_NAMES)
                }
                is_done = self._step_counter > 150 # standard policy chunk length
                return action_deg, is_done
            except Exception:
                pass

        # 2. Compliant Scripted Grasp / Manipulation Trajectory Fallback
        # Enables end-to-end execution testing before offline ACT checkpoint training
        return self._generate_scripted_step(current_joint_angles, target_3d_rel)

    def _generate_scripted_step(self,
                                current_joints: Dict[str, float],
                                target_3d: Optional[Tuple[float, float, float]]) -> Tuple[Dict[str, float], bool]:
        """Compliant multi-phase trajectory generator for the SO-ARM101."""
        step = self._step_counter
        action = dict(current_joints)

        # Baseline Reach & Grasp sequence across 120 control steps (at 30 Hz = 4.0s)
        if self._active_verb in ('grab', 'pick'):
            if step < 30:
                # Phase 1: Open gripper and lower shoulder to approach
                action['gripper'] = 80.0
                action['shoulder_lift'] = -80.0
                action['elbow_flex'] = 40.0
                action['wrist_flex'] = 45.0
                return action, False
            elif step < 70:
                # Phase 2: Extend toward target depth
                action['gripper'] = 80.0
                action['shoulder_lift'] = -60.0
                action['elbow_flex'] = 20.0
                action['wrist_flex'] = 30.0
                return action, False
            elif step < 95:
                # Phase 3: Close gripper to grasp
                action['gripper'] = 15.0 # Closed grasping position
                return action, False
            elif step < 120:
                # Phase 4: Lift object upward
                action['gripper'] = 15.0
                action['shoulder_lift'] = -85.0
                action['elbow_flex'] = 60.0
                action['wrist_flex'] = 45.0
                return action, False
            else:
                return action, True

        elif self._active_verb == 'throw':
            if step < 30:
                # Wind up
                action['shoulder_lift'] = -100.0
                action['elbow_flex'] = 80.0
                return action, False
            elif step < 45:
                # Dynamic forward flick & release
                action['shoulder_lift'] = -40.0
                action['elbow_flex'] = 10.0
                action['gripper'] = 85.0 # Open
                return action, False
            elif step < 75:
                # Return to neutral
                action['shoulder_lift'] = -90.0
                action['elbow_flex'] = 70.0
                return action, False
            else:
                return action, True

        elif self._active_verb == 'place':
            if step < 40:
                # Lower to surface
                action['shoulder_lift'] = -55.0
                action['elbow_flex'] = 15.0
                action['wrist_flex'] = 20.0
                return action, False
            elif step < 65:
                # Open gripper
                action['gripper'] = 75.0
                return action, False
            elif step < 90:
                # Retract arm
                action['shoulder_lift'] = -95.0
                action['elbow_flex'] = 80.0
                return action, False
            else:
                return action, True

        return action, True
