# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(tldr_cpp)

set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
include_directories(${SOURCE_DIR})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force static libraries
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)

# Set position independent code globally
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#set(JSON_BuildTests OFF CACHE INTERNAL "")

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(POPPLER REQUIRED poppler-cpp)
find_package(PostgreSQL REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(CURL REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(SQLiteCpp REQUIRED)


# Create static library
add_library(tldr STATIC
    ${SOURCE_DIR}/lib_tldr/lib_tldr.cpp
    ${SOURCE_DIR}/lib_tldr/lib_tldr.h
    ${SOURCE_DIR}/lib_tldr/constants.h
    ${SOURCE_DIR}/lib_tldr/db/sqlite_database.cpp
    ${SOURCE_DIR}/lib_tldr/db/sqlite_database.h
    ${SOURCE_DIR}/lib_tldr/db/database.h
    ${SOURCE_DIR}/lib_tldr/db/postgres_database.cpp
    ${SOURCE_DIR}/lib_tldr/db/postgres_database.h
    ${SOURCE_DIR}/lib_tldr/db/connection_pool.h
    ${SOURCE_DIR}/lib_tldr/tldr_api.cpp
    ${SOURCE_DIR}/lib_tldr/tldr_api.h
)

# Include directories for the library
target_include_directories(tldr PRIVATE
    ${PostgreSQL_INCLUDE_DIRS}
    ${PQXX_INCLUDE_DIRS}
    ${SQLite3_INCLUDE_DIRS}
    /opt/homebrew/opt/libpq/include
    /opt/homebrew/opt/libpqxx/include
    /opt/homebrew/opt/poppler/include
    /opt/homebrew/opt/curl/include
    /opt/homebrew/opt/nlohmann-json/include
    /opt/homebrew/opt/sqlitecpp/include
    /Users/manu/dev/UW/cap_prj/ref_projects/llama.cpp/include
    /Users/manu/dev/UW/cap_prj/ref_projects/llama.cpp/ggml/include
)

# Set POSITION_INDEPENDENT_CODE to allow linking into Swift
set_property(TARGET tldr PROPERTY POSITION_INDEPENDENT_CODE ON)

# Always prefer static libraries
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a" CACHE STRING "Library suffixes")

# Link libraries for the static library with explicit static library paths
target_link_libraries(tldr PUBLIC
        /opt/homebrew/opt/libpq/lib/libpq.a
        /opt/homebrew/opt/libpqxx/lib/libpqxx.a
        /opt/homebrew/opt/poppler/lib/libpoppler.a
        /opt/homebrew/opt/curl/lib/libcurl.a
         /Users/manu/dev/UW/cap_prj/ref_projects/llama.cpp/build/libs/libcommon.a
         /Users/manu/dev/UW/cap_prj/ref_projects/llama.cpp/build/libs/libggml-base.a
         /Users/manu/dev/UW/cap_prj/ref_projects/llama.cpp/build/libs/libggml-blas.a
         /Users/manu/dev/UW/cap_prj/ref_projects/llama.cpp/build/libs/libggml-cpu.a
         /Users/manu/dev/UW/cap_prj/ref_projects/llama.cpp/build/libs/libggml-metal.a
         /Users/manu/dev/UW/cap_prj/ref_projects/llama.cpp/build/libs/libggml.a
         /Users/manu/dev/UW/cap_prj/ref_projects/llama.cpp/build/libs/libllama.a
         /Users/manu/dev/UW/cap_prj/ref_projects/llama.cpp/build/libs/libllava_static.a
        "-framework Accelerate"
        "-framework Metal"
        "-framework Foundation"
)

target_include_directories(tldr PUBLIC ${SOURCE_DIR})

# Add the executable
add_executable(tldr_cpp ${SOURCE_DIR}/main.cpp)

# Link the static library with the executable
target_link_libraries(tldr_cpp PUBLIC tldr
        /opt/homebrew/opt/libpq/lib/libpq.a
        /opt/homebrew/opt/libpqxx/lib/libpqxx.a
        /opt/homebrew/opt/poppler/lib/libpoppler.a
        /opt/homebrew/opt/curl/lib/libcurl.a
)

target_link_options(tldr_cpp PUBLIC
        -L/opt/homebrew/opt/libpq/lib/
        -L/opt/homebrew/opt/libpqxx/lib/
        -L/opt/homebrew/opt/poppler/lib/
        -lpq
        -lpoppler-cpp
        -lcurl
)
# Include directories for the executable
target_include_directories(tldr_cpp PRIVATE ${SOURCE_DIR})
