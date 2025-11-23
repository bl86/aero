# WSL2 Setup Guide for Local LLM System

You're running on WSL2 (Windows Subsystem for Linux). This requires a different setup process than native Linux.

## Prerequisites

### 1. Windows Requirements

- **Windows 11** or **Windows 10** (Version 21H2 or higher)
- **WSL2** installed and updated
- At least **32GB RAM** (for RTX 3090)
- **100GB+ free disk space**

### 2. Update WSL2

In **Windows PowerShell** (as Administrator):

```powershell
wsl --update
wsl --set-default-version 2
```

## Step-by-Step Setup

### Step 1: Install NVIDIA Driver on Windows (REQUIRED)

**You MUST install NVIDIA drivers on Windows, NOT in WSL2.**

1. **Download Driver:**
   - Go to: https://www.nvidia.com/Download/index.aspx
   - Select:
     - Product Type: **GeForce**
     - Product Series: **GeForce RTX 30 Series**
     - Product: **GeForce RTX 3090**
     - Operating System: **Windows 11** or **Windows 10**
   - Download: **Game Ready Driver** (latest version)

2. **Install the driver on Windows**

3. **Reboot Windows**

4. **Verify in Windows:**
   Open PowerShell and run:
   ```powershell
   nvidia-smi
   ```
   You should see your RTX 3090 listed.

### Step 2: Configure WSL2 Resources

Create or edit `.wslconfig` in your Windows home directory:

**Windows PowerShell:**
```powershell
notepad $env:USERPROFILE\.wslconfig
```

Add this content:
```ini
[wsl2]
memory=32GB
processors=8
swap=8GB
localhostForwarding=true
nestedVirtualization=true
```

Save and restart WSL:
```powershell
wsl --shutdown
```

Then reopen your WSL terminal.

### Step 3: Run WSL2 GPU Setup

In your **WSL2 terminal**:

```bash
cd ~/aero
chmod +x setup_gpu_wsl2.sh
./setup_gpu_wsl2.sh
```

This will:
- Verify Windows NVIDIA driver is accessible
- Install CUDA toolkit for WSL2 (without drivers)
- Install cuDNN
- Configure environment variables
- Test GPU access

### Step 4: Verify GPU Access

```bash
nvidia-smi
```

You should see your RTX 3090 with memory and GPU usage.

### Step 5: Install LLM System

```bash
chmod +x install_llm.sh
./install_llm.sh
```

### Step 6: Download Models

```bash
source ~/llm_system/venv/bin/activate
python3 download_models.py
```

## Quick Start (After Setup)

```bash
source ~/llm_system/venv/bin/activate
python3 -m system.main --mode interactive
```

## WSL2-Specific Considerations

### GPU Access

- **GPU drivers are managed by Windows**, not WSL2
- WSL2 accesses the GPU through the Windows driver
- To update GPU drivers, update on Windows, not in WSL

### Performance

- **Slightly slower** than native Linux (5-10% overhead)
- Still very fast for LLM inference
- Full CUDA support including cuDNN

### Distributed Training

- **Remote RTX 3080 rigs must be native Linux** (not WSL2)
- WSL2 can coordinate training but remote workers need native Linux
- Network access from WSL2 to remote GPUs works fine

### File System

- **Use WSL2 file system** (`~/` or `/home/user/`) for best performance
- Avoid `/mnt/c/` (Windows drives) - they are much slower
- All models and data should be in WSL2 filesystem

### Troubleshooting

#### GPU Not Found

1. **Verify Windows driver:**
   ```powershell
   # In Windows PowerShell
   nvidia-smi
   ```

2. **Restart WSL:**
   ```powershell
   # In Windows PowerShell
   wsl --shutdown
   ```
   Then reopen WSL terminal

3. **Check driver version:**
   - Need NVIDIA driver **470.76 or newer** for WSL2 GPU support
   - Update Windows NVIDIA driver if needed

#### CUDA Not Working

1. **Reload environment:**
   ```bash
   source ~/.bashrc
   ```

2. **Check CUDA path:**
   ```bash
   echo $CUDA_HOME
   echo $LD_LIBRARY_PATH
   ```

3. **Reinstall CUDA toolkit:**
   ```bash
   ./setup_gpu_wsl2.sh
   ```

#### Out of Memory

1. **Increase WSL2 memory in `.wslconfig`:**
   ```ini
   [wsl2]
   memory=48GB
   ```

2. **Restart WSL:**
   ```powershell
   wsl --shutdown
   ```

#### Slow Performance

- **Move data to WSL2 filesystem** (not `/mnt/c/`)
- **Disable Windows Defender** for WSL2 directories
- **Close other Windows applications** using GPU

### Windows Defender Exclusion (Recommended)

To improve performance, exclude WSL2 from Windows Defender:

**Windows PowerShell (as Administrator):**
```powershell
Add-MpPreference -ExclusionPath "$env:USERPROFILE\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc\LocalState\ext4.vhdx"
```

## WSL2 vs Native Linux

| Feature | WSL2 | Native Linux |
|---------|------|--------------|
| GPU Access | ✓ (via Windows) | ✓ (direct) |
| Performance | 90-95% | 100% |
| Setup Complexity | Medium | Low |
| Driver Updates | Windows only | Linux updates |
| File I/O | Slower on /mnt/c | Fast everywhere |
| Networking | Some limitations | Full access |

## Recommended WSL2 Workflow

1. **Development in WSL2** - Use VSCode with WSL extension
2. **Models in WSL2** - Store in `~/llm_system/models/`
3. **Training data in WSL2** - Not on Windows drives
4. **Remote training** - Use native Linux machines for remote GPUs
5. **API server** - Works great from WSL2, accessible from Windows

## Accessing from Windows

### API Server

Start in WSL2:
```bash
python3 -m system.main --mode server --port 8000
```

Access from Windows browser:
```
http://localhost:8000/docs
```

### File Access

Access WSL2 files from Windows Explorer:
```
\\wsl$\Ubuntu\home\user\llm_system
```

## Performance Tips

1. **Use WSL2 filesystem** for all LLM operations
2. **Allocate sufficient memory** in `.wslconfig`
3. **Keep Windows NVIDIA drivers updated**
4. **Close unnecessary Windows apps** when running LLM
5. **Use SSD** for best performance

## Network Configuration

For distributed training, you may need to configure networking:

```bash
# Get WSL2 IP
ip addr show eth0

# Windows firewall may need rules for ports
# Add rules in Windows Firewall for ports: 8000, 6379 (Ray)
```

## Docker in WSL2 (Optional)

If you want to use Docker:

```bash
# Install Docker in WSL2
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Enable GPU support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo service docker restart
```

## System Limits

Edit `/etc/security/limits.conf` in WSL2:

```bash
sudo nano /etc/security/limits.conf
```

Add:
```
* soft nofile 65535
* hard nofile 65535
* soft nproc 65535
* hard nproc 65535
```

## Summary

WSL2 provides excellent GPU support for LLM development:

✓ Full CUDA support
✓ cuDNN support
✓ Nearly native performance
✓ Easy Windows integration
✓ Same codebase as Linux

The main difference is driver management (Windows handles it) and slightly lower file I/O on Windows drives.

---

**Need Help?**

Check logs: `~/llm_system/logs/`

Common issues: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
