#!/bin/bash

echo "Fixing WSL2 Environment"
echo "======================="
echo ""

# 1. Clean up broken PyTorch installation
echo "1. Removing incorrect PyTorch installation..."
pip3 uninstall -y torch torchvision torchaudio 2>/dev/null || true
sudo pip3 uninstall -y torch torchvision torchaudio 2>/dev/null || true

# 2. Reload environment
echo ""
echo "2. Loading CUDA environment..."
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# 3. Verify CUDA is accessible
echo ""
echo "3. Verifying CUDA..."
if [ -f /usr/local/cuda/bin/nvcc ]; then
    echo "✓ CUDA compiler found at: /usr/local/cuda/bin/nvcc"
    /usr/local/cuda/bin/nvcc --version
else
    echo "✗ CUDA compiler not found"
    echo "Reinstalling CUDA..."

    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit-12-4
    rm cuda-keyring_1.1-1_all.deb
fi

# 4. Test GPU access
echo ""
echo "4. Testing GPU access..."
if nvidia-smi; then
    echo "✓ GPU accessible!"
else
    echo "✗ GPU not accessible"
    echo ""
    echo "Please ensure:"
    echo "1. NVIDIA drivers are installed on Windows"
    echo "2. Run in Windows PowerShell: wsl --shutdown"
    echo "3. Restart this WSL terminal"
    exit 1
fi

# 5. Update bashrc to ensure paths are always loaded
echo ""
echo "5. Updating shell configuration..."

# Remove old CUDA entries
sed -i '/CUDA_HOME/d' ~/.bashrc 2>/dev/null || true
sed -i '/cuda/d' ~/.bashrc 2>/dev/null || true
sed -i '/nvidia/d' ~/.bashrc 2>/dev/null || true

# Add clean CUDA configuration
cat >> ~/.bashrc << 'EOF'

# CUDA Configuration for WSL2
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
EOF

echo "✓ Shell configuration updated"

# 6. Create activation script
echo ""
echo "6. Creating environment activation script..."

cat > ~/activate_cuda.sh << 'EOF'
#!/bin/bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

echo "CUDA environment activated!"
echo "CUDA_HOME: $CUDA_HOME"
echo ""
echo "Verify with: nvcc --version"
EOF

chmod +x ~/activate_cuda.sh

echo "✓ Created ~/activate_cuda.sh"

echo ""
echo "========================================"
echo "Environment Fix Complete!"
echo "========================================"
echo ""
echo "IMPORTANT: Close this terminal and open a new one, then:"
echo ""
echo "1. Activate CUDA environment:"
echo "   source ~/activate_cuda.sh"
echo ""
echo "2. Verify CUDA works:"
echo "   nvcc --version"
echo "   nvidia-smi"
echo ""
echo "3. Continue with LLM installation:"
echo "   cd ~/aero"
echo "   ./install_llm.sh"
echo ""
