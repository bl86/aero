#!/bin/bash

echo "GPU Test for WSL2"
echo "================="
echo ""

# Load environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

echo "1. Testing nvidia-smi..."
echo "------------------------"
if nvidia-smi; then
    echo "✓ nvidia-smi works!"
else
    echo "✗ nvidia-smi failed!"
    echo ""
    echo "Fix:"
    echo "1. Install NVIDIA drivers on Windows (not WSL)"
    echo "2. In Windows PowerShell: wsl --shutdown"
    echo "3. Reopen WSL terminal"
    exit 1
fi

echo ""
echo "2. Testing CUDA compiler..."
echo "---------------------------"
if nvcc --version 2>/dev/null; then
    echo "✓ CUDA compiler works!"
elif [ -f /usr/local/cuda/bin/nvcc ]; then
    echo "⚠ CUDA compiler exists but not in PATH"
    echo ""
    echo "Fix: Run this command:"
    echo "  source ~/activate_cuda.sh"
    /usr/local/cuda/bin/nvcc --version
else
    echo "✗ CUDA compiler not installed"
    echo ""
    echo "Fix: Run setup_gpu_wsl2.sh again"
    exit 1
fi

echo ""
echo "3. Testing CUDA with Python..."
echo "-------------------------------"

# Check if PyTorch is installed
if python3 -c "import torch" 2>/dev/null; then
    python3 << 'PYEOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("✓ PyTorch + CUDA working!")
else:
    print("⚠ PyTorch installed but CUDA not available")
    print("")
    print("Fix: Reinstall PyTorch with CUDA support:")
    print("  pip3 uninstall torch torchvision torchaudio")
    print("  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
PYEOF
else
    echo "⚠ PyTorch not installed yet (this is OK)"
    echo ""
    echo "PyTorch will be installed when you run install_llm.sh"
fi

echo ""
echo "4. GPU Information..."
echo "---------------------"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader

echo ""
echo "========================================"
echo "GPU Test Complete!"
echo "========================================"
echo ""

# Check if everything passed
if nvidia-smi &>/dev/null && (nvcc --version &>/dev/null || [ -f /usr/local/cuda/bin/nvcc ]); then
    echo "✓ GPU setup is working correctly!"
    echo ""
    echo "You can now run: ./install_llm.sh"
else
    echo "⚠ Some issues found. See messages above."
fi

echo ""
