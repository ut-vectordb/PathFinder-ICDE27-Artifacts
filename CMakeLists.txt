cmake_minimum_required(VERSION 3.15)
project(ANNS)
option(BYPASS_B_TREE "Use direct vector/attribute arrays for read-only experiments" OFF)
if(BYPASS_B_TREE)
    add_compile_definitions(BYPASS_B_TREE)
endif()

# cmake setting
set(CMAKE_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# add_compile_options(-fsanitize=address)
# add_link_options(-fsanitize=address)

# Boost
# set(Boost_INCLUDEDIRS "specify if needed")
# set(Boost_ROOT "specify if needed")
# set(Boost_DIR "specify if needed")
find_package(Boost COMPONENTS REQUIRED program_options filesystem)
message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIR})
else()
    message(FATAL_ERROR "no Boost support")
endif()

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP support")
endif()

# build options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wno-sign-compare -Wno-deprecated-declarations")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG -fsanitize=address")   # address sanitizer cannot handle memory usage beyond 64GB, so doesn't work with WIT.
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG -O3")

# -ffast-math breaks correctness, so we do not use it
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -DNDEBUG -O3 -march=native -mtune=native")

find_package(TBB REQUIRED)

# third-party dependencies
add_subdirectory(third_party)

# modules
add_subdirectory(modules/core)
add_subdirectory(modules/vamana)
add_subdirectory(modules/hnsw)
add_subdirectory(modules/tablestorage)
add_subdirectory(modules/querytype)
add_subdirectory(modules/irangegraph)
add_subdirectory(modules/queryplanning)
add_subdirectory(modules/baselines)

# applications, tools, and tests
add_subdirectory(app)
add_subdirectory(tools)
add_subdirectory(tests/unit)
