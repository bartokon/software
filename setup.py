"""
Setup script for the software repository.

This allows the repository to be installed as a package:
    pip install -e .

After installation, you can import from any project:
    from python.icp_2d import point_to_point_utils_2d
    from python.common.geometry import rotation_matrix_from_euler
"""

from setuptools import setup, find_packages
import os

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements():
    """Read requirements from requirements.txt"""
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return requirements

setup(
    name="software",
    version="0.1.0",
    author="bartokon",
    description="Research and learning repository for 3D point cloud processing, GPU computing, and deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bartokon/software",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
