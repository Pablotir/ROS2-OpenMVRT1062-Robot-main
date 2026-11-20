#!/usr/bin/env bash
# =============================================================================
# deploy_jetson_contact_graspnet.sh
# Zero-Shot 6-DoF Contact-GraspNet Deployment & Verification for NVIDIA Jetson Orin
# Robot: 5-DoF SO-ARM101 (SO-100) + Feetech STS3215 Servos + Intel RealSense D405
# =============================================================================

set -e

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
    print('⚠️ WARNING: CUDA not detected. Contact-GraspNet will run on CPU (~15s/eval instead of ~150ms).')
"

# 2. Check and install torch-geometric dependencies if missing
python3 -c "import torch_geometric" 2>/dev/null || {
    echo "📦 Installing torch-geometric..."
    pip3 install torch-geometric
}

# 3. Verify directory layout
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

if [ ! -f "checkpoints/current.pth" ]; then
    echo "⚠️ checkpoints/current.pth not found in $SCRIPT_DIR/checkpoints/!"
    if [ -f "cgn_pytorch/checkpoints/current.pth" ]; then
        mkdir -p checkpoints
        cp cgn_pytorch/checkpoints/current.pth checkpoints/
        echo "✅ Copied current.pth from cgn_pytorch/checkpoints/."
    fi
fi

# 4. Verify Contact-GraspNet Model Initialization and Benchmark
echo "🔍 Running Contact-GraspNet 5-DoF Pipeline Benchmark..."
python3 test_pipeline_phone.py

echo "=================================================================="
echo "✅ Deployment verification complete!"
echo "To run live object picking on the physical robot:"
echo "   python3 arm_picker.py"
echo "=================================================================="
