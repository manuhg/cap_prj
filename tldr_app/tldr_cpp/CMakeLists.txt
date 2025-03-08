# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(tldr_cpp)

set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/c++)
include_directories(${SOURCE_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(dependencies/json)

# Find required packages
find_package(OpenSSL REQUIRED)
find_package(PkgConfig REQUIRED)
#find_package(PDFIO REQUIRED)

pkg_check_modules(SQLITE3 REQUIRED sqlite3)
#pkg_check_modules(PDFIO REQUIRED pdfio)

# Add the executable
#add_executable(tldr_cpp file-watcher.cpp)
file(GLOB_RECURSE SRC_FILES ${SOURCE_DIR}/*.cpp ${SOURCE_DIR}/*.h)
add_executable(tldr_cpp ${SRC_FILES})



# Include directories
target_include_directories(tldr_cpp PRIVATE
        ${OPENSSL_INCLUDE_DIR}
        ${SQLITE3_INCLUDE_DIRS}
#        ${PDFIO_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(tldr_cpp PRIVATE
        ${OPENSSL_LIBRARIES}
        ${SQLITE3_LIBRARIES}
#        ${PDFIO_LIBRARIES}
        nlohmann_json::nlohmann_json
        stdc++fs
)

