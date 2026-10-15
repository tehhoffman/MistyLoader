#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/bootstrap.py
export CCACHE_DIR="${CCACHE_DIR:-$PWD/build/ccache}"
cmake -S vendor/llama.cpp -B build/mistyloader -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA="${MISTY_CUDA:-ON}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-86}" \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
  -DMISTRAL_MOE_NVTX="${MISTY_NVTX:-OFF}" \
  "$@"
cmake --build build/mistyloader --target llama-server test-mistral-moe-cpu test-backend-ops -j "${BUILD_JOBS:-8}"
