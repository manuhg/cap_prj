# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(tldr_cpp)

set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
include_directories(${SOURCE_DIR})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(JSON_BuildTests OFF CACHE INTERNAL "")

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(POPPLER REQUIRED poppler-cpp)
find_package(PostgreSQL REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(CURL REQUIRED)
find_package(SQLite3 REQUIRED)
find_package(SQLiteCpp REQUIRED)

find_package(PkgConfig)
if (PkgConfig_FOUND)
    pkg_check_modules(PQXX IMPORTED_TARGET libpqxx)
    pkg_check_modules(POPPLER REQUIRED poppler-cpp)
endif ()

# Add linker flags for library paths
link_directories(
    /opt/homebrew/opt/libpq/lib
    /opt/homebrew/opt/libpqxx/lib
    /opt/homebrew/opt/poppler/lib
)

# Create static library
add_library(lib_tldr STATIC
    ${SOURCE_DIR}/lib_tldr/lib_tldr.cpp
    ${SOURCE_DIR}/lib_tldr/lib_tldr.h
    ${SOURCE_DIR}/lib_tldr/constants.h
    ${SOURCE_DIR}/lib_tldr/db/sqlite_database.cpp
    ${SOURCE_DIR}/lib_tldr/db/sqlite_database.h
    ${SOURCE_DIR}/lib_tldr/db/database.h
    ${SOURCE_DIR}/lib_tldr/db/postgres_database.cpp
    ${SOURCE_DIR}/lib_tldr/db/postgres_database.h
    ${SOURCE_DIR}/lib_tldr/db/connection_pool.h
    ${SOURCE_DIR}/lib_tldr/tldr.cpp
    ${SOURCE_DIR}/lib_tldr/tldr.h
)

# Include directories for the library
target_include_directories(lib_tldr PRIVATE
    ${PostgreSQL_INCLUDE_DIRS}
    ${PQXX_INCLUDE_DIRS}
    ${SQLite3_INCLUDE_DIRS}
    /opt/homebrew/opt/libpq/include
    /opt/homebrew/opt/libpqxx/include
    /opt/homebrew/opt/poppler/include
    /opt/homebrew/opt/curl/include
    /opt/homebrew/opt/nlohmann-json/include
)

# Link libraries for the static library
target_link_libraries(lib_tldr PRIVATE
    ${PostgreSQL_LIBRARIES}
    ${PQXX_LIBRARIES}
    SQLite::SQLite3
    SQLiteCpp
    nlohmann_json::nlohmann_json
    CURL::libcurl
    pqxx
    pq
    poppler-cpp
)
target_include_directories(lib_tldr PUBLIC ${SOURCE_DIR})

# Add the executable
add_executable(tldr_cpp ${SOURCE_DIR}/main.cpp)

# Link the static library with the executable
target_link_libraries(tldr_cpp PRIVATE lib_tldr)

# Include directories for the executable
target_include_directories(tldr_cpp PRIVATE ${SOURCE_DIR})
