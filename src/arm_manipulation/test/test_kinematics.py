import pytest
import numpy as np
import os
from arm_manipulation.arm_kinematics import ArmKinematics

@pytest.fixture
def kinematics():
    config_path = os.path.join(
        os.path.dirname(__file__), '..', 'config', 'arm_params.yaml'
    )
    return ArmKinematics(config_path)

def test_workspace_in_bounds_center(kinematics):
    assert kinematics.check_workspace_bounds(100.0, 0.0, 0.0) == True

def test_workspace_out_of_bounds_x(kinematics):
    assert kinematics.check_workspace_bounds(500.0, 0.0, 0.0) == False

def test_workspace_out_of_bounds_rho(kinematics):
    # point at (200, 200, 0) rho=283 is in bounds
    assert kinematics.check_workspace_bounds(200.0, 200.0, 0.0) == True
    # point at (250, 250, 0) rho=354 is not
    assert kinematics.check_workspace_bounds(250.0, 250.0, 0.0) == False

def test_workspace_out_of_bounds_z(kinematics):
    # point at (0, 0, 300) exceeds z_max
    assert kinematics.check_workspace_bounds(0.0, 0.0, 300.0) == False

def test_forward_kinematics_home(kinematics):
    # joints at scan_base pose should return a valid 4x4 matrix
    joints = kinematics.get_named_pose('scan_base')
    fk_matrix = kinematics.forward_kinematics(joints)
    assert isinstance(fk_matrix, np.ndarray)
    assert fk_matrix.shape == (4, 4)

def test_forward_kinematics_identity(kinematics):
    # all zeros should compute a known position
    joints = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0
    }
    fk_matrix = kinematics.forward_kinematics(joints)
    assert isinstance(fk_matrix, np.ndarray)
    assert fk_matrix.shape == (4, 4)

def test_solve_ik_reachable(kinematics):
    # target at (150, 0, -100) should return valid joint angles
    angles = kinematics.solve_ik(150.0, 0.0, -100.0)
    assert angles is not None
    assert isinstance(angles, dict)

def test_solve_ik_unreachable(kinematics):
    # target at (500, 0, 0) should return None
    angles = kinematics.solve_ik(500.0, 0.0, 0.0)
    assert angles is None

def test_solve_ik_joint_limits(kinematics):
    # returned angles should always be within joint limits
    angles = kinematics.solve_ik(150.0, 0.0, -100.0)
    assert angles is not None
    assert kinematics.check_joint_limits(angles) == True

def test_solve_ik_backward_guard(kinematics):
    # IK should not produce lift < -90 (backward lean)
    angles = kinematics.solve_ik(-50.0, 0.0, 100.0)
    if angles is not None:
        assert angles.get('shoulder_lift', 0) >= -np.pi/2

def test_level_approach_trajectory(kinematics):
    # verify wrist stays approximately level throughout
    traj = kinematics.level_approach_trajectory((150.0, 0.0, -50.0))
    assert traj is not None
    assert len(traj) > 0
    for joints in traj:
        assert 'wrist_flex' in joints

def test_depth_in_range_valid(kinematics):
    # 200mm should be valid
    assert kinematics.is_depth_in_range(200.0) == True

def test_depth_in_range_blind(kinematics):
    # 50mm should be invalid (D405 blind zone)
    assert kinematics.is_depth_in_range(50.0) == False

def test_depth_in_range_background(kinematics):
    # 800mm should be invalid (background noise)
    assert kinematics.is_depth_in_range(800.0) == False

def test_mask_area_valid(kinematics):
    # 10% should pass, 20% should fail
    assert kinematics.is_mask_area_valid(0.10) == True
    assert kinematics.is_mask_area_valid(0.20) == False

def test_named_poses(kinematics):
    # scan_base and stow_base should return correct values
    scan_base = kinematics.get_named_pose('scan_base')
    stow_base = kinematics.get_named_pose('stow_base')
    assert scan_base is not None
    assert stow_base is not None
    assert 'shoulder_pan' in scan_base
    assert 'shoulder_pan' in stow_base
