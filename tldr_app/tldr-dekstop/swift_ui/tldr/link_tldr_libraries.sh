#!/bin/bash

# Set absolute paths
TLDR_CPP_DIR="/Users/manu/dev/UW/cap_prj/tldr_app/tldr-dekstop/tldr_cpp"
SWIFT_UI_DIR="/Users/manu/dev/UW/cap_prj/tldr_app/tldr-dekstop/swift_ui/tldr"

# Build the self-contained static library
echo "Building tldr library with all dependencies..."
cd "$TLDR_CPP_DIR"

# Make build directory if it doesn't exist
mkdir -p build
cd build

# Build the library with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release
make tldr

# Copy the self-contained library to the Swift project
echo "Copying library to Swift project..."
cp "$TLDR_CPP_DIR/build/libtldr.a" "$SWIFT_UI_DIR/TldrAPI/"

# Output success message
echo "Successfully built and copied the self-contained static library"
echo "Library location: $SWIFT_UI_DIR/TldrAPI/libtldr.a"
