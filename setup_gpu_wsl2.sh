#!/bin/bash

set -e

echo "GPU Setup for WSL2 with RTX 3090"
echo "=================================="
echo ""

detect_wsl() {
    if grep -qi microsoft /proc/version; then
        echo "✓ WSL2 detected"
        return 0
    else
        echo "✗ This script is for WSL2 only"
        echo "Use setup_gpu.sh for native Linux"
        exit 1
    fi
}

check_windows_driver() {
    echo ""
    echo "IMPORTANT: WSL2 GPU Setup Requirements"
    echo "======================================"
    echo ""
    echo "You MUST install NVIDIA drivers on Windows first:"
    echo ""
    echo "1. Download NVIDIA Driver for Windows:"
    echo "   https://www.nvidia.com/Download/index.aspx"
    echo "   - Product Type: GeForce"
    echo "   - Product Series: GeForce RTX 30 Series"
    echo "   - Product: GeForce RTX 3090"
    echo "   - Operating System: Windows 11 or Windows 10"
    echo ""
    echo "2. Install the driver on Windows (not in WSL)"
    echo ""
    echo "3. Reboot Windows"
    echo ""
    read -p "Have you installed NVIDIA drivers on Windows? (y/n): " has_driver

    if [ "$has_driver" != "y" ]; then
        echo ""
        echo "Please install Windows NVIDIA drivers first, then run this script again."
        exit 1
    fi
}

verify_gpu_access() {
    echo ""
    echo "Verifying GPU access from WSL2..."

    # Check if nvidia-smi works
    if command -v nvidia-smi &> /dev/null; then
        echo "✓ nvidia-smi found, testing GPU access..."
        if nvidia-smi &> /dev/null; then
            echo "✓ GPU is accessible!"
            nvidia-smi
            return 0
        fi
    fi

    echo "✗ GPU not accessible yet. Installing CUDA toolkit for WSL2..."
    return 1
}

install_cuda_wsl2() {
    echo ""
    echo "Installing CUDA Toolkit for WSL2..."
    echo "===================================="

    # Remove old CUDA/NVIDIA packages
    sudo apt-get remove --purge -y 'cuda*' 'nvidia*' 2>/dev/null || true

    # Install dependencies
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        wget \
        curl \
        git \
        vim \
        htop \
        python3-pip \
        python3-dev \
        cmake \
        ninja-build

    # Add NVIDIA package repository
    echo "Adding NVIDIA repository..."

    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update

    # Install CUDA toolkit (WSL2 specific - NO DRIVER)
    echo "Installing CUDA toolkit..."
    sudo apt-get install -y cuda-toolkit-12-4

    # Install cuDNN
    echo "Installing cuDNN..."
    sudo apt-get install -y cudnn9-cuda-12

    # Clean up
    rm cuda-keyring_1.1-1_all.deb
}

setup_environment() {
    echo ""
    echo "Setting up environment variables..."

    ENV_FILE="$HOME/.bashrc"

    # Remove old CUDA paths
    sed -i '/CUDA_HOME/d' $ENV_FILE 2>/dev/null || true
    sed -i '/cuda/d' $ENV_FILE 2>/dev/null || true

    # Add new paths
    cat >> $ENV_FILE << 'EOF'

# CUDA for WSL2
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
EOF

    # Apply immediately
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

    echo "✓ Environment configured"
}

install_monitoring() {
    echo ""
    echo "Installing GPU monitoring tools..."

    pip3 install gpustat pynvml --user
}

verify_installation() {
    echo ""
    echo "========================================"
    echo "Verifying Installation"
    echo "========================================"
    echo ""

    # Source the environment
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

    echo "1. Checking nvidia-smi..."
    if nvidia-smi; then
        echo "✓ nvidia-smi working"
    else
        echo "✗ nvidia-smi failed"
        echo ""
        echo "Troubleshooting:"
        echo "1. Ensure Windows NVIDIA driver is installed"
        echo "2. Restart WSL: wsl --shutdown (in Windows PowerShell)"
        echo "3. Check Windows driver version supports CUDA"
        return 1
    fi

    echo ""
    echo "2. Checking CUDA compiler..."
    if nvcc --version; then
        echo "✓ CUDA compiler working"
    else
        echo "⚠ CUDA compiler not found (may need to restart WSL)"
    fi

    echo ""
    echo "3. Testing GPU with Python..."
    python3 << 'PYEOF'
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠ CUDA not available (install PyTorch with CUDA support)")
except ImportError:
    print("⚠ PyTorch not installed yet (will be installed with LLM system)")
PYEOF
}

create_wsl_config() {
    echo ""
    echo "Creating WSL configuration..."

    # Check if .wslconfig exists in Windows home
    WINDOWS_USER=$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d '\r')
    WSLCONFIG="/mnt/c/Users/$WINDOWS_USER/.wslconfig"

    if [ ! -f "$WSLCONFIG" ]; then
        echo "Creating recommended .wslconfig..."
        cat > "$WSLCONFIG" << 'EOF'
[wsl2]
memory=32GB
processors=8
swap=8GB
localhostForwarding=true

# GPU settings
nestedVirtualization=true
EOF
        echo "✓ Created $WSLCONFIG"
        echo ""
        echo "IMPORTANT: Restart WSL for config to take effect:"
        echo "  In Windows PowerShell: wsl --shutdown"
        echo "  Then reopen WSL"
    else
        echo "✓ .wslconfig already exists"
    fi
}

main() {
    detect_wsl
    check_windows_driver

    if ! verify_gpu_access; then
        install_cuda_wsl2
        setup_environment
    fi

    install_monitoring
    create_wsl_config

    echo ""
    echo "Reloading environment..."
    source ~/.bashrc 2>/dev/null || true

    verify_installation

    echo ""
    echo "========================================"
    echo "WSL2 GPU Setup Complete!"
    echo "========================================"
    echo ""
    echo "Next steps:"
    echo "1. Close and reopen your WSL terminal (or run: source ~/.bashrc)"
    echo "2. Verify GPU: nvidia-smi"
    echo "3. Continue with: ./install_llm.sh"
    echo ""
    echo "WSL2 Notes:"
    echo "- GPU drivers are managed by Windows, not WSL"
    echo "- Update Windows NVIDIA drivers to update GPU support"
    echo "- If GPU stops working, restart WSL: wsl --shutdown"
    echo ""
}

main
