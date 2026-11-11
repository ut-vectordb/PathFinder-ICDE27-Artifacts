set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

foreach(tool fvecs_to_bin generate_base_labels generate_query_labels
             compute_groundtruth compute_groundtruth_rangefilter)
    add_executable(${tool} ${tool}.cpp)
    target_link_libraries(${tool} Core Boost::program_options)
endforeach()

# baseline_builders/ is a separate CMake project; it builds against the
# upstream CGIF and PathSeer code (see its CMakeLists.txt).
