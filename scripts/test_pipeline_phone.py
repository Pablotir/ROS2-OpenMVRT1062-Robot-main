import numpy as np
import simulate_arm_gui as sag

pipeline = sag.get_pipeline()
scan_q = dict(sag.DEFAULT_SCAN_Q)

for sc, label, roll_deg, pitch_deg, yaw_deg in [
    ("Flat Phone (0 deg)", "phone", 0.0, 0.0, 0.0),
    ("Rotated Phone (45 deg yaw)", "phone", 0.0, 0.0, 45.0),
    ("Tilted Phone (45 deg tilt)", "phone", 0.0, 45.0, 0.0),
    ("Standing Bottle", "bottle", 0.0, 0.0, 0.0)
]:
    print(f"\n==========================================")
    print(f"Testing AnyGrasp Pipeline on '{sc}'")
    print(f"==========================================")
    pts_world = sag.make_object_cloud(label, 280.0, 0.0, -80.0, roll_deg, pitch_deg, yaw_deg)
    
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
    print(f"Target width: {solver_result.get('target_width')}")
    print(f"Pitch deg: {solver_result.get('pitch_deg')}")
    print(f"Delta phi deg: {solver_result.get('delta_phi_deg')}")
