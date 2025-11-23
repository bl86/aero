#!/bin/bash

set -e

GPU_MODEL="RTX_3090"
CUDA_VERSION="12.4"
CUDNN_VERSION="9.0"

echo "GPU Setup for ${GPU_MODEL}"
echo "=============================="

detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VER=$VERSION_ID
    else
        echo "Cannot detect OS"
        exit 1
    fi
    echo "Detected OS: $OS $VER"
}

remove_old_drivers() {
    echo "Removing old NVIDIA drivers..."
    sudo apt-get remove --purge -y '^nvidia-.*' '^libnvidia-.*' '^cuda-.*' 2>/dev/null || true
    sudo apt-get autoremove -y
    sudo apt-get autoclean
}

install_dependencies() {
    echo "Installing dependencies..."
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        dkms \
        linux-headers-$(uname -r) \
        pkg-config \
        libglvnd-dev \
        wget \
        curl \
        git \
        vim \
        htop \
        nvtop \
        python3-pip \
        python3-dev \
        cmake \
        ninja-build
}

install_nvidia_driver() {
    echo "Installing NVIDIA driver..."

    sudo add-apt-repository -y ppa:graphics-drivers/ppa
    sudo apt-get update

    DRIVER_VERSION=$(ubuntu-drivers devices | grep recommended | awk '{print $3}')

    if [ -z "$DRIVER_VERSION" ]; then
        DRIVER_VERSION="nvidia-driver-550"
    fi

    echo "Installing $DRIVER_VERSION"
    sudo apt-get install -y $DRIVER_VERSION
}

install_cuda() {
    echo "Installing CUDA Toolkit ${CUDA_VERSION}..."

    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit-12-4

    rm cuda-keyring_1.1-1_all.deb
}

install_cudnn() {
    echo "Installing cuDNN ${CUDNN_VERSION}..."

    sudo apt-get install -y cudnn9-cuda-12
}

setup_environment() {
    echo "Setting up environment variables..."

    ENV_FILE="$HOME/.bashrc"

    grep -q "CUDA_HOME" $ENV_FILE || cat >> $ENV_FILE << 'EOF'

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
EOF

    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
}

optimize_gpu() {
    echo "Optimizing GPU settings..."

    sudo nvidia-smi -pm 1

    sudo nvidia-smi --auto-boost-default=0

    sudo nvidia-smi -pl 350

    sudo nvidia-smi -ac 9751,1395
}

install_monitoring() {
    echo "Installing GPU monitoring tools..."

    pip3 install gpustat pynvml
}

verify_installation() {
    echo ""
    echo "Verifying installation..."
    echo "=========================="

    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
        echo ""
        nvcc --version
        echo ""
        echo "GPU setup completed successfully!"
    else
        echo "Installation may require a reboot to take effect."
        echo "Please reboot and run: nvidia-smi"
    fi
}

main() {
    detect_os
    remove_old_drivers
    install_dependencies
    install_nvidia_driver
    install_cuda
    install_cudnn
    setup_environment
    install_monitoring

    echo ""
    echo "========================================"
    echo "GPU Setup Complete!"
    echo "========================================"
    echo ""
    echo "IMPORTANT: Please reboot your system now:"
    echo "  sudo reboot"
    echo ""
    echo "After reboot, verify with:"
    echo "  nvidia-smi"
    echo "  nvcc --version"
    echo ""
    echo "Then optimize GPU with:"
    echo "  sudo nvidia-smi -pm 1"
    echo "  sudo nvidia-smi -pl 350"
    echo ""
}

main
