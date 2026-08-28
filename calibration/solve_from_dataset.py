import json
import math
import numpy as np
import cv2
import yaml

dataset_file = "calibration_dataset.json"
out_path = "hand_eye_calibration.yaml"

with open(dataset_file, 'r') as f:
    ds = json.load(f)

R_gb_list = []
t_gb_list = []
R_tc_list = []
t_tc_list = []

PAN_ZERO_OFFSET_DEG = -4.6
IK_L1 = 115.0
IK_L2 = 137.5

for d in ds:
    j = d['joints']
    # OPTIMIZED POLARITIES
    pan  = math.radians(-j.get("shoulder_pan.pos", 0) - PAN_ZERO_OFFSET_DEG)
    lift = j.get("shoulder_lift.pos", 0)
    elb  = j.get("elbow_flex.pos", 0)
    wst  = j.get("wrist_flex.pos", 0)
    roll = math.radians(-j.get("wrist_roll.pos", 0))

    t1 = math.radians(90.0 - lift)
    t2 = t1 - math.radians(elb + 81.0)
    t3 = t2 - math.radians(wst + 5.0)

    rho_w = IK_L1 * math.cos(t1) + IK_L2 * math.cos(t2)
    z_w   = IK_L1 * math.sin(t1) + IK_L2 * math.sin(t2)

    wx = rho_w * math.cos(pan)
    wy = rho_w * math.sin(pan)
    wz = z_w

    ax = math.cos(t3) * math.cos(pan)
    ay = math.cos(t3) * math.sin(pan)
    az = math.sin(t3)

    zx = -math.sin(pan)
    zy =  math.cos(pan)
    zz = 0.0

    yx = zy * az - zz * ay
    yy = zz * ax - zx * az
    yz = zx * ay - zy * ax

    R_base = np.array([
        [ax, yx, zx],
        [ay, yy, zy],
        [az, yz, zz]
    ])

    R_roll = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(roll), -math.sin(roll)],
        [0.0, math.sin(roll),  math.cos(roll)]
    ])
    
    R_final = R_base @ R_roll
    
    R_gb_list.append(R_final)
    t_gb_list.append(np.array([wx/1000.0, wy/1000.0, wz/1000.0]))
    
    rvec = np.array(d['rvec_cam'], dtype=np.float32)
    tvec = np.array(d['tvec_cam'], dtype=np.float32)
    R_tc, _ = cv2.Rodrigues(rvec)
    
    R_tc_list.append(R_tc)
    t_tc_list.append(tvec)

R_cg, t_cg = cv2.calibrateHandEye(R_gb_list, t_gb_list, R_tc_list, t_tc_list, method=cv2.CALIB_HAND_EYE_DANIILIDIS)

# calculate error
errs = []
for i in range(len(ds)):
    T_bg = np.eye(4); T_bg[:3,:3]=R_gb_list[i]; T_bg[:3,3]=t_gb_list[i].flatten()
    T_tc = np.eye(4); T_tc[:3,:3]=R_tc_list[i]; T_tc[:3,3]=t_tc_list[i].flatten()
    T_cg_mat = np.eye(4); T_cg_mat[:3,:3]=R_cg; T_cg_mat[:3,3]=t_cg.flatten()
    T_target_base = T_bg @ T_cg_mat @ T_tc
    errs.append(T_target_base[:3, 3])

errs = np.array(errs)
mean_err = np.mean(np.linalg.norm(errs - np.mean(errs, axis=0), axis=1)) * 1000.0

print(f"? Hand-Eye Calibration solved from dataset!")
print(f"Mean 3D Error: {mean_err:.2f} mm")
print(f"Translation: X={t_cg[0][0]*1000:.2f}, Y={t_cg[1][0]*1000:.2f}, Z={t_cg[2][0]*1000:.2f} mm")

data = {
    'rotation_matrix': R_cg.tolist(),
    'translation_mm': (t_cg * 1000.0).tolist(),
    'reprojection_error': float(mean_err),
    'num_poses_used': len(ds),
    'method': 'CALIB_HAND_EYE_DANIILIDIS'
}
with open(out_path, 'w') as f:
    yaml.dump(data, f, sort_keys=False)
print(f"Saved to {out_path}")
