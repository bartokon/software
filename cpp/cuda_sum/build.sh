#!/bin/bash

#CUDA COMPILATION
/usr/local/cuda-11.8/bin/nvcc main_vadd.cu --output-directory=cuda -o=main_vadd.elf

#ROCM TRANSLATION
/opt/rocm-6.0.0/bin/hipify-clang main_vadd.cu --o-dir=rocm
/opt/rocm-6.0.0/bin/hipify-clang vadd.cu --o-dir=rocm

#ONEAPI TRANSLATION
/opt/intel/oneapi/2024.0/bin/dpct main_vadd.cu --out-root=oneapi
/opt/intel/oneapi/2024.0/bin/dpct vadd.cu --out-root=oneapi