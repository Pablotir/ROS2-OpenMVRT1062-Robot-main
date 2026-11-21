#!/usr/bin/env bash
# =============================================================================
# deploy_jetson_contact_graspnet.sh
# Zero-Shot 6-DoF Contact-GraspNet Deployment & Verification for NVIDIA Jetson Orin
# Robot: 5-DoF SO-ARM101 (SO-100) + Feetech STS3215 Servos + Intel RealSense D405
# =============================================================================

set -e

# Prevent PyTorch/OpenMP glibc heap corruption on Jetson Orin (aarch64)
export MALLOC_TRIM_THRESHOLD_=0
export MALLOC_MMAP_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "=================================================================="
echo "🚀 Setting up Contact-GraspNet for SO-ARM101 on NVIDIA Jetson Orin"
echo "=================================================================="

# 1. Verify Python & PyTorch CUDA acceleration
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️ WARNING: CUDA not detected. Contact-GraspNet will run on CPU.')
"

# 2. Check and install torch-geometric & trimesh dependencies if missing
python3 -c "import torch_geometric" 2>/dev/null || {
    echo "📦 Installing torch-geometric..."
    pip3 install torch-geometric
}

python3 -c "import trimesh" 2>/dev/null || {
    echo "📦 Installing trimesh..."
    pip3 install trimesh
}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 3. Verify Contact-GraspNet Model Initialization and Benchmark
echo "🔍 Running Contact-GraspNet 5-DoF Pipeline Benchmark..."
if [ -f "test_pipeline_phone.py" ]; then
    python3 test_pipeline_phone.py
elif [ -f "scripts/test_pipeline_phone.py" ]; then
    python3 scripts/test_pipeline_phone.py
fi

echo "=================================================================="
echo "✅ Deployment verification complete!"
echo "To run live object picking on the physical robot:"
echo "   python3 scripts/arm_picker.py  (if in ~/ros2_ws)"
echo "   OR"
echo "   python3 arm_picker.py          (if in ~/ros2_ws/scripts)"
echo "=================================================================="
