# GPU Setup Guide

Comprehensive guide for setting up GPU computing environments for CUDA, HIP, and SYCL development.

## Table of Contents

1. [Overview](#overview)
2. [NVIDIA CUDA Setup](#nvidia-cuda-setup)
3. [AMD ROCm/HIP Setup](#amd-rocmhip-setup)
4. [Intel SYCL/OneAPI Setup](#intel-sycloneapi-setup)
5. [Verification](#verification)
6. [Environment Variables](#environment-variables)
7. [Troubleshooting](#troubleshooting)
8. [Multi-GPU Configurations](#multi-gpu-configurations)

---

## Overview

This repository supports three GPU programming models:

| Platform | API | Vendor | Tested Version |
|----------|-----|--------|----------------|
| **CUDA** | CUDA C++ | NVIDIA | 11.8+ |
| **HIP** | HIP/ROCm | AMD | 6.0.0+ |
| **SYCL** | SYCL/DPC++ | Intel | OneAPI 2024.0+ |

### Which Do You Need?

- **NVIDIA GPU** → Install CUDA
- **AMD GPU** → Install ROCm/HIP
- **Intel GPU/CPU** → Install OneAPI/SYCL
- **No GPU** → Skip to Python-only projects

---

## NVIDIA CUDA Setup

### Prerequisites

- NVIDIA GPU with compute capability 3.5+ (check: https://developer.nvidia.com/cuda-gpus)
- Linux kernel 3.10+ or Windows 10+
- GCC 11+ (Linux) or Visual Studio 2019+ (Windows)

### Step 1: Check GPU

```bash
# Check if NVIDIA GPU is present
lspci | grep -i nvidia

# If drivers installed, check GPU info
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
...
```

### Step 2: Install NVIDIA Drivers

**Ubuntu/Debian:**
```bash
# Automatic driver installation
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall

# Or manual installation
sudo apt install nvidia-driver-525

# Reboot required
sudo reboot
```

**Arch Linux:**
```bash
sudo pacman -S nvidia nvidia-utils
```

**Verify:**
```bash
nvidia-smi
```

### Step 3: Install CUDA Toolkit

**Ubuntu/Debian (Network installer):**
```bash
# Download keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb

# Update and install
sudo apt-get update
sudo apt-get -y install cuda

# OR specific version
sudo apt-get -y install cuda-11-8
```

**Arch Linux:**
```bash
sudo pacman -S cuda
```

**Manual Download:**
- Visit: https://developer.nvidia.com/cuda-downloads
- Select your OS and architecture
- Follow provided instructions

### Step 4: Set Environment Variables

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# CUDA paths
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

# Optional: Set default GPU
export CUDA_VISIBLE_DEVICES=0
```

Apply changes:
```bash
source ~/.bashrc
```

### Step 5: Verify CUDA Installation

```bash
# Check NVCC (CUDA compiler)
nvcc --version

# Should output something like:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2023 NVIDIA Corporation
# Built on Tue_Jun_13_19:16:58_PDT_2023
# Cuda compilation tools, release 11.8, V11.8.89
```

### Step 6: Test CUDA Project

```bash
cd cpp/cuda
make cuda
./cuda/vadd.elf
```

Expected output:
```
CUDA Vector Addition
Array size: 1048576
Success! Result verified.
```

### CUDA Troubleshooting

**Problem: "nvcc: command not found"**
```bash
# CUDA not in PATH
export PATH=/usr/local/cuda/bin:$PATH
```

**Problem: "libcudart.so: cannot open shared object file"**
```bash
# CUDA libraries not in LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Problem: "CUDA error: no kernel image is available"**
- Your GPU architecture may not be supported by compiled code
- Recompile with correct architecture flag: `-arch=sm_XX`
- Check your GPU's compute capability: https://developer.nvidia.com/cuda-gpus

**Problem: "CUDA out of memory"**
```bash
# Check GPU memory usage
nvidia-smi

# Kill GPU processes
sudo fuser -v /dev/nvidia*
```

---

## AMD ROCm/HIP Setup

### Prerequisites

- AMD GPU (check compatibility: https://github.com/RadeonOpenCompute/ROCm#supported-gpus)
- Linux (ROCm officially supports Ubuntu, RHEL, SLES)
- Kernel 4.17+ recommended

### Step 1: Check GPU

```bash
# Check AMD GPU
lspci | grep -i amd | grep -i vga

# Check if ROCm kernel driver is loaded
lsmod | grep amdgpu
```

### Step 2: Install ROCm

**Ubuntu 20.04/22.04:**
```bash
# Add ROCm repository
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/focal/amdgpu-install_6.0.60000-1_all.deb
sudo dpkg -i amdgpu-install_6.0.60000-1_all.deb

# Install ROCm
sudo amdgpu-install -y --usecase=hiplibsdk,rocm

# Add user to video and render groups
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER

# Reboot required
sudo reboot
```

**Arch Linux:**
```bash
# Install from AUR
yay -S rocm-hip-sdk
```

### Step 3: Set Environment Variables

Add to `~/.bashrc`:
```bash
# ROCm paths
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# HIP platform
export HIP_PLATFORM=amd

# Optional: GPU override for specific architectures
# export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

### Step 4: Verify ROCm Installation

```bash
# Check ROCm version
/opt/rocm/bin/rocm-smi --showproductname

# Check HIP compiler
hipcc --version

# Check GPU info
rocm-smi
```

Expected `rocm-smi` output:
```
======================= ROCm System Management Interface =======================
================================= Concise Info =================================
GPU  Temp   AvgPwr  SCLK    MCLK     Fan  Perf  PwrCap  VRAM%  GPU%
0    45.0c  20.0W   800Mhz  1200Mhz  0%   auto  180.0W  5%     0%
================================================================================
```

### Step 5: Test HIP Project

```bash
cd cpp/cuda
make hip
./hip/vadd.elf
```

### ROCm Troubleshooting

**Problem: "HSA Error: Incompatible kernel and runtime"**
```bash
# Set GPU architecture override
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # Adjust for your GPU
```

**Problem: "hipcc: command not found"**
```bash
export PATH=/opt/rocm/bin:$PATH
```

**Problem: "Permission denied on /dev/kfd"**
```bash
# Add user to video/render groups
sudo usermod -a -G video,render $USER
# Logout and login again
```

---

## Intel SYCL/OneAPI Setup

### Prerequisites

- Intel CPU or Intel GPU
- Linux or Windows
- Recent kernel (5.4+) for GPU support

### Step 1: Download Intel OneAPI

Visit: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

**Linux Online Installer:**
```bash
# Download installer
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/163da6e4-56eb-4948-aba3-debcec61c064/l_BaseKit_p_2024.0.1.46_offline.sh

# Run installer
sudo sh ./l_BaseKit_p_2024.0.1.46_offline.sh
```

Follow the installation wizard.

### Step 2: Set Environment Variables

Intel OneAPI requires sourcing environment setup script:

```bash
# Add to ~/.bashrc
source /opt/intel/oneapi/setvars.sh
```

Or manually each session:
```bash
source /opt/intel/oneapi/setvars.sh
```

### Step 3: Verify Installation

```bash
# Check Intel C++ compiler
icpx --version

# Check DPC++ compiler
dpcpp --version

# List available devices
sycl-ls
```

Expected `sycl-ls` output:
```
[opencl:cpu][opencl:0] Intel(R) OpenCL, Intel(R) Core(TM) i7-XXXX CPU @ X.XXGHz OpenCL 3.0 (Build 0)
[opencl:gpu][opencl:1] Intel(R) OpenCL HD Graphics, Intel(R) Iris(R) Xe Graphics OpenCL 3.0 NEO
[level_zero:gpu][level_zero:0] Intel(R) Level-Zero, Intel(R) Iris(R) Xe Graphics 1.3
```

### Step 4: Test SYCL Project

```bash
cd cpp/cuda
make dpc
./dpc/vadd.elf
```

### SYCL Troubleshooting

**Problem: "icpx: command not found"**
```bash
source /opt/intel/oneapi/setvars.sh
```

**Problem: "No SYCL devices found"**
```bash
# Check available devices
sycl-ls

# Install GPU drivers if using Intel GPU
sudo apt install intel-opencl-icd  # Ubuntu/Debian
```

---

## Verification

### Quick Verification Script

Create `verify_gpu.sh`:

```bash
#!/bin/bash

echo "=== GPU Environment Verification ==="
echo

# Check CUDA
echo "--- NVIDIA CUDA ---"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "nvidia-smi failed"
else
    echo "CUDA not found"
fi
echo

# Check ROCm/HIP
echo "--- AMD ROCm/HIP ---"
if command -v hipcc &> /dev/null; then
    hipcc --version | grep "HIP version"
    rocm-smi --showproductname 2>/dev/null || echo "rocm-smi failed"
else
    echo "ROCm not found"
fi
echo

# Check Intel SYCL
echo "--- Intel SYCL ---"
if command -v icpx &> /dev/null; then
    icpx --version | head -1
    sycl-ls | head -3
else
    echo "Intel SYCL not found (run: source /opt/intel/oneapi/setvars.sh)"
fi
echo

echo "=== Verification Complete ==="
```

Run:
```bash
chmod +x verify_gpu.sh
./verify_gpu.sh
```

---

## Environment Variables

### Essential Variables

**CUDA:**
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0  # Which GPU to use (0, 1, 2, etc.)
```

**ROCm/HIP:**
```bash
export ROCM_HOME=/opt/rocm
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH
export HIP_PLATFORM=amd
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # If needed for compatibility
```

**Intel SYCL:**
```bash
source /opt/intel/oneapi/setvars.sh
export SYCL_DEVICE_FILTER=level_zero:gpu  # Or opencl:gpu, opencl:cpu
```

### Complete Environment Template

Create `~/.gpu_env`:
```bash
# GPU Computing Environment

# CUDA (NVIDIA)
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
fi

# ROCm (AMD)
if [ -d "/opt/rocm" ]; then
    export ROCM_HOME=/opt/rocm
    export PATH=$ROCM_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH
    export HIP_PLATFORM=amd
fi

# Intel OneAPI
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
fi

echo "GPU environment loaded"
```

Source in `~/.bashrc`:
```bash
source ~/.gpu_env
```

---

## Multi-GPU Configurations

### Select Specific GPU

**CUDA:**
```bash
# Use GPU 0
export CUDA_VISIBLE_DEVICES=0

# Use GPU 1
export CUDA_VISIBLE_DEVICES=1

# Use GPUs 0 and 2
export CUDA_VISIBLE_DEVICES=0,2

# Hide all GPUs (CPU only)
export CUDA_VISIBLE_DEVICES=""
```

**ROCm:**
```bash
export HIP_VISIBLE_DEVICES=0  # Similar to CUDA
```

### Check Available GPUs

```bash
# CUDA
nvidia-smi --list-gpus

# ROCm
rocm-smi --showid

# Intel
sycl-ls
```

---

## Performance Monitoring

### NVIDIA

```bash
# Real-time monitoring
nvidia-smi -l 1  # Update every 1 second

# Watch specific metrics
watch -n 1 nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
```

### AMD

```bash
# Real-time monitoring
watch -n 1 rocm-smi
```

---

## Common Issues

### Issue: Driver/Runtime Version Mismatch

**CUDA:**
```bash
# Check versions
nvidia-smi  # Shows driver version and CUDA runtime version
nvcc --version  # Shows CUDA compiler version

# These can differ - that's OK!
# Driver version should be >= runtime version
```

### Issue: GPU Not Detected

```bash
# Check PCI devices
lspci | grep -i 'vga\|3d\|nvidia\|amd'

# Check kernel modules
lsmod | grep -E 'nvidia|amdgpu'

# Check dmesg for errors
sudo dmesg | grep -E 'nvidia|amdgpu' | tail -20
```

### Issue: Compilation Errors

**Wrong architecture:**
```bash
# CUDA: Specify architecture for your GPU
nvcc -arch=sm_75 ...  # For Turing
nvcc -arch=sm_80 ...  # For Ampere
nvcc -arch=sm_86 ...  # For RTX 30xx

# Check your GPU's compute capability:
# https://developer.nvidia.com/cuda-gpus
```

---

## Project-Specific Notes

### cpp/cuda/
Supports all three platforms with Makefile targets:
```bash
make cuda    # NVIDIA
make hip     # AMD
make dpc     # Intel
```

### cpp/cuda_sum/
CUDA-only currently:
```bash
make
```

### Python Projects
Use PyTorch's CUDA support:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

## Further Reading

- **CUDA**: https://docs.nvidia.com/cuda/
- **ROCm**: https://rocm.docs.amd.com/
- **SYCL**: https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html
- **PyTorch GPU**: https://pytorch.org/get-started/locally/

---

For help, open an issue on GitHub with your:
- OS and version
- GPU model
- Driver version
- Error messages
- Output of verification script
