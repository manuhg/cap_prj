# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(tldr_cpp VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(PostgreSQL REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(CURL REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(SQLiteCpp REQUIRED)

# Find additional packages using pkg-config
pkg_check_modules(PQXX REQUIRED IMPORTED_TARGET libpqxx)
pkg_check_modules(POPPLER REQUIRED IMPORTED_TARGET poppler-cpp)

# Add homebrew paths for macOS
if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/poppler")
    list(APPEND CMAKE_LIBRARY_PATH "/opt/homebrew/opt/poppler/lib")
    list(APPEND CMAKE_INCLUDE_PATH "/opt/homebrew/opt/poppler/include")
endif()

# Define source files
set(LIB_SRC_FILES
    src/lib_tldr/tldr_lib.cpp
    src/lib_tldr/db/sqlite_database.cpp
    src/lib_tldr/db/postgres_database.cpp
)

# Create static library
add_library(tldr_cpp_lib STATIC ${LIB_SRC_FILES})

# Set include directories for the library
target_include_directories(tldr_cpp_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${PostgreSQL_INCLUDE_DIRS}
    ${PQXX_INCLUDE_DIRS}
    ${SQLite3_INCLUDE_DIRS}
    ${POPPLER_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIRS}
)

# Link libraries for the library
target_link_libraries(tldr_cpp_lib PUBLIC
    ${PostgreSQL_LIBRARIES}
    PkgConfig::PQXX
    SQLite::SQLite3
    SQLiteCpp
    PkgConfig::POPPLER
    CURL::libcurl
    nlohmann_json::nlohmann_json
)

# Create executable
add_executable(tldr_cpp src/main.cpp)

# Link the library to the executable
target_link_libraries(tldr_cpp PRIVATE tldr_cpp_lib)

# Install rules
install(TARGETS tldr_cpp_lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(TARGETS tldr_cpp
    RUNTIME DESTINATION bin
)

# Add compile definitions
target_compile_definitions(tldr_cpp_lib PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
)

# Add compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tldr_cpp_lib PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()