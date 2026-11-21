from cgn_pytorch.pointnet2 import (
    FPModule,
    PointNet,
    SAModule,
    GlobalSAModule,
    MLP,
    safe_fps,
    safe_radius,
    safe_knn_interpolate
)

# For backward compatibility if any module imports fps, radius, etc. from models
fps = safe_fps
radius = safe_radius
knn_interpolate = safe_knn_interpolate
