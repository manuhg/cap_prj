#!/bin/bash
set -e

# Paths
SRC_DIR="$(dirname "$0")"
CPP_FILE="$SRC_DIR/npu_test.cpp"
HEADER_FILE="$SRC_DIR/npu_accelerator.h"
LIB_DIR="/Users/manu/proj_tldr/tldr-dekstop/release-products/libs"
LIB_FILE="$LIB_DIR/libnpu-accelerator.a"

# Output
OUT_EXE="$SRC_DIR/npu_test"

# Compile npu_test.cpp and link with the static Swift library and CoreML framework
clang++ \
    -std=c++17 \
    -I"$SRC_DIR" \
    -L"$LIB_DIR" \
    "$CPP_FILE" \
    -lnpu-accelerator \
    -framework CoreML \
    -framework Foundation \
    -o "$OUT_EXE"

echo "Build complete: $OUT_EXE"
echo "------------------------------"
echo "Running $OUT_EXE..."
"$OUT_EXE"
