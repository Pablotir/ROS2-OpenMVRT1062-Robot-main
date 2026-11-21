import sys, os
import math
import numpy as np
from soarm101_anygrasp_pipeline import ThreeStageAnyGraspPipeline

def make_object_cloud(label: str, x: float, y: float, z: float,
                      roll_deg: float, pitch_deg: float, yaw_deg: float,
                      num_pts: int = 600) -> np.ndarray:
    if label == "bottle":
        theta = np.random.uniform(0, 2 * np.pi, num_pts)
        z_loc = np.random.uniform(0, 170, num_pts)
        r = np.where(z_loc < 130, 28.0, np.where(z_loc < 155, 14.0 + 14.0 * (155.0 - z_loc) / 25.0, 14.0))
        pts_local = np.column_stack([r * np.cos(theta), r * np.sin(theta), z_loc])
    elif label == "phone":
        px = np.random.uniform(-72.5, 72.5, num_pts)
        py = np.random.uniform(-36, 36, num_pts)
        pz = np.random.uniform(0, 9, num_pts)
        pts_local = np.column_stack([px, py, pz])
    else:
        theta = np.random.uniform(0, 2 * np.pi, num_pts)
        z_loc = np.random.uniform(0, 115, num_pts)
        r = np.full(num_pts, 26.0)
        pts_local = np.column_stack([r * np.cos(theta), r * np.sin(theta), z_loc])

    rx = math.radians(roll_deg)
    ry = math.radians(pitch_deg)
    rz = math.radians(yaw_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    R = Rx @ Ry @ Rz

    pts_rot = (R @ pts_local.T).T
    pts_rot[:, 0] += x
    pts_rot[:, 1] += y
    pts_rot[:, 2] += z

    z_min = np.min(pts_rot[:, 2])
    if z_min < -80.0:
        pts_rot[:, 2] += (-80.0 - z_min)
    return pts_rot

pipeline = ThreeStageAnyGraspPipeline()
scan_q = {
    "shoulder_pan.pos":  -11.60,
    "shoulder_lift.pos": -104.92,
    "elbow_flex.pos":     96.09,
    "wrist_flex.pos":     32.53,
    "wrist_roll.pos":     29.27,
    "gripper.pos":        72.6
}

for sc, label, roll_deg, pitch_deg, yaw_deg in [
    ("Flat Phone (0 deg)", "phone", 0.0, 0.0, 0.0),
    ("Rotated Phone (45 deg yaw)", "phone", 0.0, 0.0, 45.0),
    ("Tilted Phone (45 deg tilt)", "phone", 0.0, 45.0, 0.0),
    ("Standing Bottle", "bottle", 0.0, 0.0, 0.0)
]:
    print(f"\n==========================================")
    print(f"Testing Contact-GraspNet Pipeline on '{sc}'")
    print(f"==========================================")
    pts_world = make_object_cloud(label, 280.0, 0.0, -80.0, roll_deg, pitch_deg, yaw_deg)
    
    T_grasps, scores, widths = pipeline.stage2_engine.synthesize_6dof_grasps(
        P_obj_base=pts_world,
        P_bg_base=pts_world,
        target_label=label
    )
    best_idx, solver_result = pipeline.stage3_solver.solve_pipeline(
        T_grasps=T_grasps,
        scores=scores,
        widths=widths,
        P_obj_base=pts_world,
        Z_table=-80.0,
        q_current=scan_q,
        target_label=label
    )
    print(f"Valid candidates: {solver_result['valid_candidate_count']} / {solver_result['total_candidate_count']}")
    print(f"Composite score: {solver_result['composite_score']:.2f}")
    print(f"Target joints: {solver_result['target_joints']}")
    print(f"Pitch deg: {solver_result.get('pitch_deg', 0.0):.1f}°")
    print(f"Delta phi deg: {solver_result.get('delta_phi_deg', 0.0):.1f}°")
