#!/bin/bash
# Setup script for the software repository
# This script sets up the Python environment and installs all dependencies

set -e  # Exit on error

echo "============================================"
echo "Software Repository - Environment Setup"
echo "============================================"
echo ""

# Check Python version
echo "[1/6] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

echo "  Found Python $python_version"

# Compare versions
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "  ✓ Python version is sufficient (>= 3.8)"
else
    echo "  ✗ Python 3.8+ is required"
    echo "  Please install Python 3.8 or higher"
    exit 1
fi

echo ""

# Create virtual environment
echo "[2/6] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "  Virtual environment already exists"
    read -p "  Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo "  ✓ Virtual environment recreated"
    else
        echo "  Using existing virtual environment"
    fi
else
    python3 -m venv venv
    echo "  ✓ Virtual environment created"
fi

echo ""

# Activate virtual environment
echo "[3/6] Activating virtual environment..."
source venv/bin/activate
echo "  ✓ Virtual environment activated"

echo ""

# Upgrade pip
echo "[4/6] Upgrading pip..."
pip install --upgrade pip --quiet
pip_version=$(pip --version | awk '{print $2}')
echo "  ✓ pip upgraded to version $pip_version"

echo ""

# Install dependencies
echo "[5/6] Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "  ✓ Dependencies installed"
else
    echo "  ✗ requirements.txt not found"
    exit 1
fi

echo ""

# Install package in editable mode
echo "[6/6] Installing package in editable mode..."
pip install -e .
echo "  ✓ Package installed"

echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "To activate the environment in the future:"
echo "  source venv/bin/activate"
echo ""
echo "To verify installation:"
echo "  python -c 'from python.common import geometry; print(\"✓ Imports working\")'"
echo ""
echo "To run examples:"
echo "  cd python/icp_2d"
echo "  python point_to_point_least_squares_2d.py"
echo ""
echo "Happy coding! 🚀"
