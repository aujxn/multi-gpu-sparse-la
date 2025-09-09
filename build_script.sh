#!/bin/bash

set -e

echo "Loading modules..."
module load cuda

export NCCL_HOME=$(pwd)/nccl_2.28.3-1+cuda12.9_x86_64

echo "Building..."
mkdir -p build
cmake -S . -B build \
  -DNCCL_ROOT="$NCCL_HOME" \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cd build
cmake --build . --parallel `nproc`
cd ..
echo "Build complete."
