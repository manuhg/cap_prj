#!/bin/bash

# This script adds the necessary frameworks and libraries to link with the C++ code
# Add this to a "Run Script" build phase in your Xcode project

# 1. Create a new file that will be used in the Xcode project as a custom linker flags file
cat > "/Users/manu/dev/UW/cap_prj/tldr_app/tldr-dekstop/swift_ui/tldr/external_libs.txt" << EOF
-lcurl
-lsqlite3
-lpq
-lpqxx
-lpoppler-cpp
EOF

echo "Created external_libs.txt with required libraries"
echo "-----------------------------------------------"
echo "IMPORTANT: Add these steps in Xcode:"
echo "1. Open your Xcode project"
echo "2. Select the 'tldr' target"
echo "3. Go to 'Build Phases'"
echo "4. Click '+' at the top and select 'New Run Script Phase'"
echo "5. Add this script command:"
echo "   cat \"\${PROJECT_DIR}/external_libs.txt\" >> \"\${TARGET_BUILD_DIR}/\${PRODUCT_NAME}.app/Contents/MacOS/lib_flags.txt\""
echo "6. Under 'Build Settings' > 'Other Linker Flags', add:"
echo "   -L/opt/homebrew/lib -lcurl -lsqlite3 -lpq -lpqxx -lpoppler-cpp"
echo "7. Under 'Build Settings' > 'Library Search Paths', add:"
echo "   /opt/homebrew/lib"
echo "   \${PROJECT_DIR}/TldrAPI"
echo "-----------------------------------------------"
