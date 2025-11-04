# Getting Started Guide

This guide will help you set up the development environment and get started with the projects in this repository.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Python Setup](#python-setup)
3. [C++ Setup](#c-setup)
4. [GPU Setup](#gpu-setup)
5. [Verifying Installation](#verifying-installation)
6. [Running Examples](#running-examples)
7. [Troubleshooting](#troubleshooting)

## System Requirements

### Operating System
- Linux (tested on Ubuntu 20.04+)
- macOS (limited GPU support)
- Windows (via WSL2 recommended)

### Software Requirements

**Python:**
- Python 3.8 or higher
- pip package manager
- virtualenv (recommended)

**C++:**
- GCC 11+ with C++20 support
- Make build system
- CMake (optional)

**GPU Computing (Optional):**
- CUDA Toolkit 11.8+ (NVIDIA GPUs)
- ROCm 6.0.0+ (AMD GPUs)
- Intel OneAPI 2024.0+ (Intel GPUs/CPUs)

**Storage:**
- At least 5GB free space for dependencies
- Additional space for datasets (varies by project)

## Python Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/bartokon/software.git
cd software
```

### Step 2: Create Virtual Environment

Using venv (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
# OR
venv\Scripts\activate  # On Windows
```

Using conda:
```bash
conda create -n software python=3.10
conda activate software
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### Step 4: Verify Python Installation

```bash
# Check Python version
python --version

# Test imports
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Test package import
python -c "from python.icp_2d import point_to_point_utils_2d; print('Package import successful!')"
```

## C++ Setup

### Step 1: Install Compiler

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install build-essential g++-11 make
```

**macOS:**
```bash
xcode-select --install
# Or install via Homebrew
brew install gcc@11
```

**Arch Linux:**
```bash
sudo pacman -S base-devel gcc
```

### Step 2: Verify C++ Installation

```bash
# Check GCC version (should be 11+)
g++ --version

# Check Make
make --version
```

### Step 3: Build a Test Project

```bash
cd cpp/bubble_pointer_sort
make
./main
```

If this builds and runs successfully, your C++ environment is ready!

## GPU Setup

GPU support is optional but required for CUDA/HIP/SYCL projects.

### NVIDIA CUDA Setup

**1. Check GPU:**
```bash
lspci | grep -i nvidia
nvidia-smi  # Should show your GPU info
```

**2. Install CUDA Toolkit 11.8+:**

Follow instructions at: https://developer.nvidia.com/cuda-downloads

Ubuntu example:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
```

**3. Set Environment Variables:**
```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**4. Verify CUDA:**
```bash
nvcc --version
```

**5. Test CUDA Project:**
```bash
cd cpp/cuda
make cuda
./cuda/vadd.elf
```

### AMD ROCm Setup

**1. Check GPU:**
```bash
lspci | grep -i amd
rocm-smi  # After installation
```

**2. Install ROCm:**

Follow: https://rocmdocs.amd.com/en/latest/

**3. Set Environment Variables:**
```bash
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

**4. Verify ROCm:**
```bash
hipcc --version
```

**5. Test HIP Project:**
```bash
cd cpp/cuda
make hip
./hip/vadd.elf
```

### Intel SYCL Setup

**1. Install Intel OneAPI:**

Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

**2. Source Environment:**
```bash
source /opt/intel/oneapi/setvars.sh
```

**3. Verify:**
```bash
icpx --version
```

**4. Test SYCL Project:**
```bash
cd cpp/cuda
make dpc
./dpc/vadd.elf
```

See [gpu-setup.md](gpu-setup.md) for detailed GPU configuration.

## Verifying Installation

### Complete Verification Script

Create a file `verify_setup.py`:

```python
#!/usr/bin/env python3
import sys

def check_python_version():
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ⚠ Warning: Python 3.8+ recommended")

def check_packages():
    packages = [
        'numpy',
        'torch',
        'matplotlib',
        'scipy'
    ]

    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {pkg} {version}")
        except ImportError:
            print(f"✗ {pkg} NOT FOUND")

def check_project_imports():
    try:
        from python.icp_2d import point_to_point_utils_2d
        print("✓ Project imports working")
    except ImportError as e:
        print(f"✗ Project imports failed: {e}")

def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available (CPU only)")
    except:
        print("⚠ Could not check GPU")

if __name__ == '__main__':
    print("=== Software Repository Setup Verification ===\n")
    check_python_version()
    print()
    check_packages()
    print()
    check_project_imports()
    print()
    check_gpu()
    print("\n=== Verification Complete ===")
```

Run it:
```bash
python verify_setup.py
```

## Running Examples

### Python Examples

**ICP 2D:**
```bash
cd python/icp_2d
python point_to_point_least_squares_2d.py
```

**ICP 3D:**
```bash
cd python/icp_3d
python point_to_point_least_squares_3d.py
```

**PyTorch Model:**
```bash
cd python/pytorch
python main.py --help  # See available options
```

### C++ Examples

**Point Cloud Matching:**
```bash
cd cpp/point_cloud_matching
make
./main.elf
```

**String Class:**
```bash
cd cpp/string
make
./string_test.elf
```

**CUDA Examples:**
```bash
cd cpp/cuda
make cuda
./cuda/vadd.elf

cd ../cuda_sum
make
./cuda/main_sum.elf
```

## Troubleshooting

### Python Issues

**Problem: "No module named 'python'"**
```bash
# Make sure you installed in editable mode
pip install -e .

# Check if __init__.py files exist
ls python/__init__.py
ls python/icp_2d/__init__.py
```

**Problem: "ImportError: cannot import name 'XXX'"**
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

**Problem: PyTorch CUDA not available**
```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with correct CUDA version
# Visit: https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### C++ Issues

**Problem: "g++: command not found"**
```bash
# Install GCC
sudo apt install build-essential  # Ubuntu/Debian
brew install gcc  # macOS
```

**Problem: "error: C++20 features not supported"**
```bash
# Update GCC to version 11+
sudo apt install g++-11
# Update Makefile to use g++-11
```

**Problem: "fatal error: cuda_runtime.h: No such file"**
```bash
# CUDA not installed or not in PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### GPU Issues

**Problem: "CUDA out of memory"**
- Reduce batch size in training scripts
- Close other GPU applications
- Use smaller models for testing

**Problem: "nvidia-smi command not found"**
```bash
# NVIDIA drivers not installed
# Install appropriate drivers for your GPU
sudo ubuntu-drivers autoinstall  # Ubuntu
```

**Problem: "hipcc: command not found"**
```bash
# ROCm not in PATH
export PATH=/opt/rocm/bin:$PATH
```

## Next Steps

Now that your environment is set up:

1. **Explore Projects**: Read [projects-overview.md](projects-overview.md) for project descriptions
2. **Try Examples**: Run the example scripts in each project directory
3. **Read Code**: Each project has a README explaining the implementation
4. **Experiment**: Modify parameters and see what happens!

## Additional Resources

- [Projects Overview](projects-overview.md) - Detailed project descriptions
- [GPU Setup](gpu-setup.md) - In-depth GPU configuration
- [Main README](../README.md) - Repository overview

## Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Read project-specific READMEs
3. Check GPU setup documentation
4. Open an issue on GitHub with:
   - Your OS and version
   - Python version (`python --version`)
   - Error messages
   - Steps to reproduce

Happy coding! 🚀
