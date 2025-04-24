#!/bin/bash

# Set absolute paths
TLDR_CPP_DIR="/Users/manu/dev/UW/cap_prj/tldr_app/tldr-dekstop/tldr_cpp"
SWIFT_UI_DIR="/Users/manu/dev/UW/cap_prj/tldr_app/tldr-dekstop/swift_ui/tldr"
SYSTEM_LIBS_DIR="$SWIFT_UI_DIR/TldrAPI/SystemLibs"

# Create directory for system libraries if it doesn't exist
mkdir -p "$SYSTEM_LIBS_DIR"

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

# Copy system libraries that the C++ code depends on
echo "Copying system dependencies..."

# Find required system libraries
LIBCURL_PATH=$(find /usr/lib -name "libcurl*.dylib" | head -n 1)
LIBSQLITE_PATH=$(find /usr/lib -name "libsqlite3*.dylib" | head -n 1)

# Check for Homebrew libraries
BREW_PREFIX="/opt/homebrew"
if [ -d "$BREW_PREFIX" ]; then
    echo "Found Homebrew at $BREW_PREFIX"
    
    # PostgreSQL and libpqxx libraries
    if [ -d "$BREW_PREFIX/opt/libpqxx" ]; then
        echo "Copying libpqxx and related libraries..."
        cp -v "$BREW_PREFIX/opt/libpqxx/lib/libpqxx.dylib" "$SYSTEM_LIBS_DIR/" 2>/dev/null || echo "Could not copy libpqxx.dylib"
        cp -v "$BREW_PREFIX/opt/libpqxx/lib/libpqxx.a" "$SYSTEM_LIBS_DIR/" 2>/dev/null || echo "Could not copy libpqxx.a"
    fi
    
    if [ -d "$BREW_PREFIX/opt/libpq" ]; then
        cp -v "$BREW_PREFIX/opt/libpq/lib/libpq.dylib" "$SYSTEM_LIBS_DIR/" 2>/dev/null || echo "Could not copy libpq.dylib"
        cp -v "$BREW_PREFIX/opt/libpq/lib/libpq.a" "$SYSTEM_LIBS_DIR/" 2>/dev/null || echo "Could not copy libpq.a"
    fi
    
    # Poppler library
    if [ -d "$BREW_PREFIX/opt/poppler" ]; then
        echo "Copying poppler libraries..."
        cp -v "$BREW_PREFIX/opt/poppler/lib/libpoppler-cpp.dylib" "$SYSTEM_LIBS_DIR/" 2>/dev/null || echo "Could not copy libpoppler-cpp.dylib"
        cp -v "$BREW_PREFIX/opt/poppler/lib/libpoppler-cpp.a" "$SYSTEM_LIBS_DIR/" 2>/dev/null || echo "Could not copy libpoppler-cpp.a"
    fi
fi

# Copy system libraries if found
if [ -n "$LIBCURL_PATH" ]; then
    cp "$LIBCURL_PATH" "$SYSTEM_LIBS_DIR/"
    echo "Copied $LIBCURL_PATH"
fi

if [ -n "$LIBSQLITE_PATH" ]; then
    cp "$LIBSQLITE_PATH" "$SYSTEM_LIBS_DIR/"
    echo "Copied $LIBSQLITE_PATH"
fi

# Output success message
echo "Successfully built and copied the self-contained static library"
echo "Library location: $SWIFT_UI_DIR/TldrAPI/libtldr.a"
echo "System libraries location: $SYSTEM_LIBS_DIR"
