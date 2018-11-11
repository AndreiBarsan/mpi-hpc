cmake_minimum_required(VERSION 3.5)
project(hpsc)

# Requires Open MPI (or equivalent) to be installed. On macOS, you can simply use `brew install openmpi` for this.
find_package(MPI REQUIRED)

# Build gflags anywhere with -DREGISTER_BUILD_DIR=ON and this will work (no need to install).
find_package(gflags REQUIRED)
# TODO(andreib): Set appropriate flag if found or not found.
find_package(Eigen3 REQUIRED NO_MODULE)

include_directories(SYSTEM ${MPI_INCLUDE_PATH} .)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -msse3 -Wall -Wextra -Wno-reorder") # -fopenmp")

# Assignment 1 Executables
set(COMMON_SOURCES src/common/utils.cpp)
string(STRIP ${COMMON_SOURCES} COMMON_SOURCES)
add_executable(globsum src/a01/globsum.cpp ${COMMON_SOURCES})
add_executable(fanout src/a01/fanout.cpp ${COMMON_SOURCES})
add_executable(ring src/a01/ring.cpp ${COMMON_SOURCES})

add_executable(spline_problem src/a02/spline_problem.cpp ${COMMON_SOURCES} src/a02/matrix.h src/a02/serial_numerical.h src/a02/parallel_numerical.h)

string(STRIP "${MPI_CXX_LIBRARIES}" MPI_CXX_LIBRARIES)
string(STRIP "${MPI_CXX_LINK_FLAGS}" MPI_CXX_LINK_FLAGS)

target_compile_options(globsum PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(globsum ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} gflags stdc++fs)

target_compile_options(fanout PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(fanout ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} gflags stdc++fs)

target_compile_options(ring PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(ring ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} gflags stdc++fs)

target_compile_options(spline_problem PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(spline_problem ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} gflags Eigen3::Eigen stdc++fs)
