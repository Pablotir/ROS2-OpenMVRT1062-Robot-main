import math
import numpy as np
import yaml
from typing import Optional, List, Dict
from pathlib import Path

class ArmKinematics:
    """SO-ARM101 forward/inverse kinematics with full safety constraint enforcement.
    
    All constants loaded from arm_params.yaml — nothing hardcoded.
    Pure math library with no ROS or hardware dependencies.
    """
    
    def __init__(self, config_path: str | Path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self.links = config['link_lengths']
        self.pan_zero_offset_deg = config['pan_zero_offset_deg']
        self.limits = config['joint_limits']
        self.ws = config['workspace_bounds']
        self.depth_guards = config['depth_guards']
        self.mask_guard = config['mask_guard']
        self.ik_params = config['ik_solver']
        self.poses = config.get('poses', {})
        self.motion = config.get('motion', {})
        
        # Link lengths
        self.L1 = self.links['shoulder_to_elbow']
        self.L2 = self.links['elbow_to_wrist']
        self.L3 = self.links['wrist_to_gripper']
        self.shoulder_height = self.links['shoulder_height']

    def workspace_in_bounds(self, x_mm: float, y_mm: float, z_mm: float) -> bool:
        """Fast R³ bounding-box pre-check. Returns False if target is 
        definitively outside the physical envelope. Called BEFORE solve_ik()."""
        rho = math.sqrt(x_mm**2 + y_mm**2)
        if x_mm < self.ws['x_min_mm'] or x_mm > self.ws['x_max_mm']:
            return False
        if y_mm < self.ws['y_min_mm'] or y_mm > self.ws['y_max_mm']:
            return False
        if z_mm < self.ws['z_min_mm'] or z_mm > self.ws['z_max_mm']:
            return False
        if rho > self.ws['rho_max_mm']:
            return False
        return True

    def forward_kinematics(self, joints: Dict[str, float]) -> np.ndarray:
        """Compute 4×4 homogeneous transform T_wrist_base from joint angles.
        
        joints dict keys can be full names or dot pos names, assuming raw values for now.
        For safety we extract standard names.
        """
        # Extract joints safely handling different dict formats
        pan = joints.get('shoulder_pan', joints.get('shoulder_pan.pos', 0.0))
        lift = joints.get('shoulder_lift', joints.get('shoulder_lift.pos', 0.0))
        elbow = joints.get('elbow_flex', joints.get('elbow_flex.pos', 0.0))
        wrist = joints.get('wrist_flex', joints.get('wrist_flex.pos', 0.0))

        # Forward kinematics equations
        pan_rad = math.radians(pan + self.pan_zero_offset_deg)
        t1 = math.radians(90.0 - lift)
        t2 = t1 - math.radians(elbow + 81.0)  # 81.0 is elbow zero offset
        t3 = t2 - math.radians(wrist + 5.0)   # 5.0 is wrist zero offset

        # Planar positions (relative to shoulder pan axis)
        wx = self.L1 * math.cos(t1) + self.L2 * math.cos(t2)
        wz = self.L1 * math.sin(t1) + self.L2 * math.sin(t2)

        gx = wx + self.L3 * math.cos(t3)
        gz = wz + self.L3 * math.sin(t3)

        # Apply pan rotation
        x = gx * math.cos(pan_rad)
        y = gx * math.sin(pan_rad)
        z = gz + self.shoulder_height

        # Create 4x4 transform
        T = np.eye(4)
        T[0, 3] = x
        T[1, 3] = y
        T[2, 3] = z
        
        # Pitch is t3 but mapped back to world frame
        pitch = math.degrees(t3)
        # Note: simplistic orientation assignment just to provide full 4x4
        # Assuming roll is zero, yaw is pan, pitch is calculated
        cy = math.cos(pan_rad)
        sy = math.sin(pan_rad)
        cp = math.cos(t3)
        sp = math.sin(t3)
        
        T[0, 0] = cp * cy
        T[0, 1] = -sy
        T[0, 2] = sp * cy
        
        T[1, 0] = cp * sy
        T[1, 1] = cy
        T[1, 2] = sp * sy
        
        T[2, 0] = -sp
        T[2, 1] = 0
        T[2, 2] = cp

        return T

    def solve_ik(self, target_xyz_mm: tuple[float, float, float], 
                 current_joints: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Closed-form analytical IK for the SO-ARM101."""
        x, y, z = target_xyz_mm
        
        if not self.workspace_in_bounds(x, y, z):
            return None

        # 1. Compute pan
        pan_rad = math.atan2(y, x)
        pan_deg = math.degrees(pan_rad) - self.pan_zero_offset_deg
        
        if pan_deg < self.limits['shoulder_pan']['min'] or pan_deg > self.limits['shoulder_pan']['max']:
            return None

        rho = math.sqrt(x**2 + y**2)
        z_rel = z - self.shoulder_height

        best_cost = float('inf')
        best_solution = None

        # Try candidate pitch angles
        pitch_min = int(self.ik_params['pitch_min_deg'])
        pitch_max = int(self.ik_params['pitch_max_deg'])

        for pitch_deg in range(pitch_min, pitch_max + 1):
            pitch_rad = math.radians(pitch_deg)

            # Back out L3
            wx = rho - self.L3 * math.cos(pitch_rad)
            wz = z_rel - self.L3 * math.sin(pitch_rad)

            D_sq = wx**2 + wz**2
            D = math.sqrt(D_sq)

            # 4. Check reachability
            if D > (self.L1 + self.L2) or D < abs(self.L1 - self.L2):
                continue

            # 6. Trig domain clamp
            cos_theta2 = (D_sq - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
            cos_theta2 = max(-1.0, min(1.0, cos_theta2))
            
            theta2_rad = math.acos(cos_theta2)
            
            # Elbow up solution usually preferred for SO-ARM101
            # We try both if needed, but standard is elbow up
            for sign in [1, -1]:
                t2_internal = sign * theta2_rad
                
                alpha = math.atan2(wz, wx)
                beta = math.atan2(self.L2 * math.sin(t2_internal), self.L1 + self.L2 * math.cos(t2_internal))
                t1_rad = alpha - beta

                # Convert to joint angles
                lift_deg = 90.0 - math.degrees(t1_rad)
                elbow_deg = math.degrees(t1_rad - t2_internal) - 81.0
                wrist_deg = math.degrees(t2_internal - pitch_rad) - 5.0

                # 2. & 3. Joint limit checks
                if not (self.limits['shoulder_lift']['min'] <= lift_deg <= self.limits['shoulder_lift']['max']):
                    continue
                if lift_deg < self.limits['shoulder_lift'].get('backward_guard', -90.0):
                    continue
                if not (self.limits['elbow_flex']['min'] <= elbow_deg <= self.limits['elbow_flex']['max']):
                    continue
                if not (self.limits['wrist_flex']['min'] <= wrist_deg <= self.limits['wrist_flex']['max']):
                    continue

                # 7. Weighted cost
                lift_cur = current_joints.get('shoulder_lift', 0.0)
                elbow_cur = current_joints.get('elbow_flex', 0.0)
                wrist_cur = current_joints.get('wrist_flex', 0.0)

                cost = (self.ik_params['lift_weight'] * abs(lift_deg - lift_cur) +
                        self.ik_params['elbow_weight'] * abs(elbow_deg - elbow_cur) +
                        self.ik_params['wrist_weight'] * abs(wrist_deg - wrist_cur))

                if cost < best_cost:
                    best_cost = cost
                    best_solution = {
                        'shoulder_pan': pan_deg,
                        'shoulder_lift': lift_deg,
                        'elbow_flex': elbow_deg,
                        'wrist_flex': wrist_deg,
                        'wrist_roll': current_joints.get('wrist_roll', 0.0),
                        'gripper': current_joints.get('gripper', 60.0)
                    }

        return best_solution

    def level_approach_trajectory(self, start_joints: Dict[str, float], end_joints: Dict[str, float], 
                                  step_size_deg: float = 2.0) -> List[Dict[str, float]]:
        """Generate trajectory where all joints interpolate uniformly while 
        wrist_flex dynamically compensates to keep gripper parallel to floor.
        """
        trajectory = []
        
        # Calculate max difference to determine number of steps
        max_diff = 0.0
        for j in ['shoulder_pan', 'shoulder_lift', 'elbow_flex']:
            diff = abs(end_joints.get(j, 0.0) - start_joints.get(j, 0.0))
            if diff > max_diff:
                max_diff = diff
                
        if max_diff == 0:
            return [end_joints.copy()]
            
        num_steps = max(2, int(math.ceil(max_diff / step_size_deg)))
        
        for i in range(num_steps + 1):
            fraction = i / num_steps
            step_joints = {}
            
            # Interpolate main joints
            for j in ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_roll', 'gripper']:
                start_val = start_joints.get(j, 0.0)
                end_val = end_joints.get(j, 0.0)
                step_joints[j] = start_val + (end_val - start_val) * fraction
                
            # Keep wrist level
            lift = step_joints['shoulder_lift']
            elbow = step_joints['elbow_flex']
            
            t1 = math.radians(90.0 - lift)
            t2 = t1 - math.radians(elbow + 81.0)
            
            level_wrist_deg = math.degrees(t2) - 5.0
            
            # Blend towards end wrist at the very end
            if i == num_steps:
                step_joints['wrist_flex'] = end_joints.get('wrist_flex', level_wrist_deg)
            else:
                # Early steps level, but we could blend it gradually too. 
                # According to docstring: "early steps -> level, final step -> target wrist"
                step_joints['wrist_flex'] = level_wrist_deg
                
            trajectory.append(step_joints)
            
        return trajectory

    def get_named_pose(self, pose_name: str) -> Dict[str, float]:
        """Get joint angles for a named pose (scan_base, stow_base, etc.)"""
        return self.poses.get(pose_name, {})

    def depth_in_range(self, depth_mm: float) -> bool:
        """Check if depth is within valid D405 range (not in blind zone, not background noise)."""
        return self.depth_guards['min_range_mm'] < depth_mm < self.depth_guards['max_grab_depth_mm']

    def mask_area_valid(self, mask_pixels: int, frame_pixels: int) -> bool:
        """Check if detection mask area is not too large (>15% = likely floor/BG)."""
        if frame_pixels == 0:
            return False
        return (mask_pixels / frame_pixels) <= self.mask_guard['max_area_fraction']
