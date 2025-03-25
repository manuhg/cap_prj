# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(tldr_cpp)

set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/c++)
include_directories(${SOURCE_DIR})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(JSON_BuildTests OFF CACHE INTERNAL "")

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(PostgreSQL REQUIRED)


find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(PQXX IMPORTED_TARGET libpqxx)
    pkg_check_modules(POPPLER REQUIRED poppler-cpp)
endif()

# Include directories
include_directories()

# Add the executable
file(GLOB_RECURSE SRC_FILES ${SOURCE_DIR}/*.cpp ${SOURCE_DIR}/*.h)
add_executable(tldr_cpp ${SRC_FILES}
        src/c++/main.cpp
        src/c++/main.h)



# Include directories
target_include_directories(tldr_cpp PRIVATE
        ${PostgreSQL_INCLUDE_DIRS}
        ${PQXX_INCLUDE_DIRS}
        /opt/homebrew/opt/libpq/include
        /opt/homebrew/opt/libpqxx/include
        /opt/homebrew/opt/poppler/include
)

target_link_options(tldr_cpp PRIVATE
        -L/opt/homebrew/opt/libpq/lib/
        -L/opt/homebrew/opt/libpqxx/lib/
        -L/opt/homebrew/opt/poppler/lib/
        -lpq
        -lpoppler-cpp
)
## Link libraries
target_link_libraries(tldr_cpp PRIVATE
        ${PostgreSQL_LIBRARIES} ${PQXX_LIBRARIES}
)


# Add the test executable
add_executable(test_pdf tests/test_pdf.cpp)



# Include directories for the test
target_include_directories(test_pdf PRIVATE
        ${PostgreSQL_INCLUDE_DIRS}
        ${PQXX_INCLUDE_DIRS}
        /opt/homebrew/opt/libpq/include
        /opt/homebrew/opt/libpqxx/include
        /opt/homebrew/opt/poppler/include
)

# Link libraries for the test
target_link_options(test_pdf PRIVATE
        -L/opt/homebrew/opt/libpq/lib/
        -L/opt/homebrew/opt/libpqxx/lib/
        -L/opt/homebrew/opt/poppler/lib/
        -lpq
        -lpoppler-cpp
)
target_link_libraries(test_pdf PRIVATE
        ${PostgreSQL_LIBRARIES} ${PQXX_LIBRARIES}
)