#!/bin/bash

set -e  # Exit on error

# Set absolute paths
TLDR_CPP_DIR="/Users/manu/dev/UW/cap_prj/tldr_app/tldr-dekstop/tldr_cpp"
SWIFT_UI_DIR="/Users/manu/dev/UW/cap_prj/tldr_app/tldr-dekstop/swift_ui/tldr"
TLDR_INCLUDE_DIR="$SWIFT_UI_DIR/TldrAPI/include"

# Create directories if they don't exist
mkdir -p "$TLDR_INCLUDE_DIR"

# Build the self-contained static library with all dependencies
echo "Building tldr library with all dependencies..."
cd "$TLDR_CPP_DIR"

# Make build directory if it doesn't exist
mkdir -p build
cd build

# Build the library with CMake - this will create a fully self-contained static library
cmake .. -DCMAKE_BUILD_TYPE=Release
make tldr

# Copy the fully self-contained static library to the Swift project
echo "Copying compiled library to Swift project..."
cp -v "$TLDR_CPP_DIR/build/libtldr.a" "$SWIFT_UI_DIR/TldrAPI/"

# Only copy the main header file needed for the API
echo "Copying main header file..."
cp -v "$TLDR_CPP_DIR/src/lib_tldr/tldr_api.h" "$TLDR_INCLUDE_DIR/"

# Update the module.modulemap file to reflect the actual C++ library
echo "Updating module.modulemap..."
cat > "$SWIFT_UI_DIR/TldrAPI/module.modulemap" << 'EOF'
module TldrAPI {
    header "TldrAPI.hpp"
    link "tldr"
    export *
}
EOF

# Update Dependencies.xcconfig with all necessary dependencies
echo "Updating Dependencies.xcconfig..."
cat > "$SWIFT_UI_DIR/TldrAPI/Dependencies.xcconfig" << 'EOF'
// Link flags for the self-contained static library
OTHER_LDFLAGS = -lstdc++ -framework CoreFoundation -framework Security

// Search paths for the library
LIBRARY_SEARCH_PATHS = $(inherited) $(PROJECT_DIR)/TldrAPI

// Header search paths
HEADER_SEARCH_PATHS = $(inherited) $(PROJECT_DIR)/TldrAPI $(PROJECT_DIR)/TldrAPI/include \
                       $(PROJECT_DIR)/TldrAPI/include/db

// C++ standard library
CLANG_CXX_LIBRARY = libc++
CLANG_CXX_LANGUAGE_STANDARD = c++20
EOF

echo "Successfully built and configured the real C++ library with all dependencies"
echo "Library location: $SWIFT_UI_DIR/TldrAPI/libtldr.a"
echo "Header files location: $TLDR_INCLUDE_DIR"
