#!/bin/bash

# Script to build the Release configuration of the npu-acclerator Xcode project

# Define the project directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/npu-acclerator"

echo "Building npu-acclerator project in Release mode..."
echo "Project directory: ${PROJECT_DIR}"

# Navigate to the project directory and run xcodebuild
cd "${PROJECT_DIR}" || exit 1

xcodebuild -scheme npu-acclerator -configuration Release build

BUILD_STATUS=$?

cd - > /dev/null # Return to original directory

if [ ${BUILD_STATUS} -eq 0 ]; then
    echo "Build succeeded."
    exit 0
else
    echo "Build failed with status ${BUILD_STATUS}."
    exit 1
fi
