cmake_minimum_required(VERSION 3.5)
project(hpsc)

set(CMAKE_CXX_STANDARD 17)

# Requires Open MPI (or equivalent) to be installed. On macOS, you can simply use `brew install openmpi` for this.
find_package(MPI REQUIRED)

# Build gflags anywhere with -DREGISTER_BUILD_DIR=ON and this will work (no need to install).
find_package(gflags REQUIRED)
find_package(Eigen3 NO_MODULE)
if(Eigen3_FOUND)
    message("Eigen3 found: ${Eigen3_VERSION} located here: ${Eigen3_DIR}")
    add_compile_definitions(DEBUG_WITH_EIGEN)
else()
    message("Eigen3 not found. Will not perform error checking.")
endif()

include_directories(SYSTEM ${MPI_INCLUDE_PATH} src)

set(CMAKE_BUILD_TYPE Debug)
# Hardcore optimization; note that -Ofast can be even more aggressive. Slow compilation but holy moly is it fast!
# Coarsely speaking, I noticed a ~20x speedup on my workstation from no opt flags to hardcore opt! (Xeon E5-2687W v4)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -Wall -Wextra -Wno-reorder -std=c++17")
# Notmal optimzation
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1 -march=native -Wall -Wextra -Wno-reorder -std=c++17")
# No optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -Wextra -Wno-reorder -std=c++17")

set(COMMON_SOURCES
        src/common/utils.cpp
        src/common/matrix.cpp
        src/common/parallel_numerical.cpp
        src/common/serial_numerical.cpp
        src/common/mpi_eigen_helpers.cpp
        src/common/mpi_stopwatch.cpp)

# Assignment 1
add_executable(globsum src/a01/globsum.cpp ${COMMON_SOURCES})
add_executable(fanout src/a01/fanout.cpp ${COMMON_SOURCES})
add_executable(ring src/a01/ring.cpp ${COMMON_SOURCES})
# Assignment 2
add_executable(spline_problem ${COMMON_SOURCES} src/a02/spline_problem.cpp)
# Assignment 3
add_executable(spline_2d_problem ${COMMON_SOURCES} src/a03/spline_2d_problem.cpp src/a03/deboor_common.cpp)
add_executable(mpi_eigen_helpers_tests ${COMMON_SOURCES} src/common/mpi_eigen_helpers_test.cpp)

# This fixes some strange CMake errors caused by unusual spaces in these variables.
string(STRIP "${MPI_CXX_LIBRARIES}" MPI_CXX_LIBRARIES)
string(STRIP "${MPI_CXX_LINK_FLAGS}" MPI_CXX_LINK_FLAGS)

set(LIBS gflags Eigen3::Eigen) #stdc++fs)

target_compile_options(globsum PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(globsum ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} ${LIBS})

target_compile_options(fanout PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(fanout ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} ${LIBS})

target_compile_options(ring PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(ring ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} ${LIBS})

target_compile_options(spline_problem PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(spline_problem ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} ${LIBS})

target_compile_options(spline_2d_problem PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(spline_2d_problem ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} ${LIBS})

# Tests
target_compile_options(mpi_eigen_helpers_tests PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(mpi_eigen_helpers_tests ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} ${LIBS})
