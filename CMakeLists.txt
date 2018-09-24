cmake_minimum_required(VERSION 3.12)
project(hpsc)

# Requires Open MPI (or equivalent) to be installed. On macOS, you can simply use `brew install openmpi` for this.
find_package(MPI REQUIRED)

# Build gflags anywhere with -DREGISTER_BUILD_DIR=ON and this will work (no need to install).
#find_package(gflags REQUIRED)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -msse3") # -fopenmp")

# Assignment 1 Executables
set(COMMON_SOURCES src/common/utils.cpp src/common/utils.h)
add_executable(globsum src/a01/globsum.cpp ${COMMON_SOURCES})
target_compile_options(globsum PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(globsum ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
