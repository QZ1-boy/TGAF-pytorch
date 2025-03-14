#!/usr/bin/env bash

# You may need to modify the following paths before compiling.  10.1
CUDA_HOME=/usr/local/cuda-11.3 \
CUDNN_INCLUDE_DIR=/usr/local/cuda-11.3/include \
CUDNN_LIB_DIR=/usr/local/cuda-11.3/lib64 \

python setup.py build_ext --inplace

if [ -d "build" ]; then
    rm -r build
fi
