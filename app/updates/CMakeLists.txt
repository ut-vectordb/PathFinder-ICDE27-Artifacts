set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

set(UPDATES_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src/updates)
set(LIBS Core Boost::program_options IRG Hnsw TableStorage QueryType Baselines TBB::tbb)

# Shared search/insert/prune functions for the online-update benchmarks.
add_library(RunSearchOnline STATIC ${UPDATES_SRC_DIR}/run_search_online.cpp)
target_include_directories(RunSearchOnline PUBLIC ${UPDATES_SRC_DIR})
target_link_libraries(RunSearchOnline PUBLIC ${LIBS})

add_library(OnlineUpdateBaselineCommon STATIC ${UPDATES_SRC_DIR}/online_update_baseline_common.cpp)
target_link_libraries(OnlineUpdateBaselineCommon PUBLIC RunSearchOnline)

add_executable(build_partial_pathfinder ${UPDATES_SRC_DIR}/build_partial_pathfinder.cpp)
target_link_libraries(build_partial_pathfinder RunSearchOnline)

add_executable(trim_saved_tree ${UPDATES_SRC_DIR}/trim_saved_tree.cpp)
target_link_libraries(trim_saved_tree RunSearchOnline)

add_executable(online_update_pathfinder_test ${UPDATES_SRC_DIR}/online_update_pathfinder_test.cpp)
target_link_libraries(online_update_pathfinder_test RunSearchOnline QueryPlanning)

foreach(method vamana hnsw navix favor)
    add_executable(online_update_${method}_baseline ${UPDATES_SRC_DIR}/online_update_${method}_baseline.cpp)
    target_link_libraries(online_update_${method}_baseline OnlineUpdateBaselineCommon)
endforeach()
