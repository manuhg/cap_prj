#!/bin/bash

# Script to build the npu-acclerator project and prepare release artifacts.

# Get the directory where this script is located (npu)
NPU_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
RELEASE_PRODUCTS_DIR="${NPU_DIR}/../release-products"

BUILD_SCRIPT="${NPU_DIR}/build_release.sh"

echo "--- Starting Release Packaging Process ---"

# Step 1: Run the build script
echo "Executing build script: ${BUILD_SCRIPT}"
"${BUILD_SCRIPT}"
BUILD_STATUS=$?

if [ ${BUILD_STATUS} -ne 0 ]; then
    echo "Error: Build script failed with status ${BUILD_STATUS}. Aborting packaging."
    exit 1
fi

echo "Build script completed successfully."

# Step 2: Run the artifact preparation script
echo "Executing artifact release preparation: $PWD"
cp npu-acclerator/npu-acclerator/npu_accelerator.h $RELEASE_PRODUCTS_DIR/include
cd $RELEASE_PRODUCTS_DIR && cp -v Release/npu-acclerator libs/libnpu-accelerator.a && cp -rv Release/*.mlmodelc artefacts/.

echo "Artifact preparation script completed successfully."

echo "--- Release Packaging Process Completed Successfully ---"

exit 0
