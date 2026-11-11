set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

set(EXP_HELPERS
    ${CMAKE_SOURCE_DIR}/app/src/run_search.cpp
    ${CMAKE_SOURCE_DIR}/app/src/run_search.h
)
set(LIBS Core Boost::program_options IRG Hnsw TableStorage QueryPlanning QueryType)

add_executable(test_btree               test_btree.cpp)
target_link_libraries(test_btree Boost::program_options TableStorage)
add_executable(test_concurrent_edge_map test_concurrent_edge_map.cpp)
target_link_libraries(test_concurrent_edge_map Core TBB::tbb)
add_executable(icde_scan_test           icde_scan_test.cpp)
target_link_libraries(icde_scan_test TableStorage)
add_executable(icde_search_test         icde_search_test.cpp)
target_link_libraries(icde_search_test Baselines)

function(add_test_target name)
    add_executable(${name} ${ARGN})
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/app/src)
    target_link_libraries(${name} PUBLIC ${LIBS})
endfunction()

add_test_target(dec24_fanout_factors_build          dec24_fanout_factors_build.cpp ${EXP_HELPERS})
add_test_target(dec24_fanout_factors_build_for_time dec24_fanout_factors_build_for_time.cpp ${EXP_HELPERS})
add_test_target(dec24_fanout_factors_test           dec24_fanout_factors_test.cpp ${EXP_HELPERS})
add_test_target(dec24_fanout_factors_test_disj      dec24_fanout_factors_test_disj.cpp ${EXP_HELPERS})
add_test_target(dec24_irange_build                  dec24_irange_build.cpp ${EXP_HELPERS})
