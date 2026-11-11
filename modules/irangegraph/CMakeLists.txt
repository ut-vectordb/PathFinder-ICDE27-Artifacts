file(GLOB_RECURSE IRG_INTERNAL_HEADERS internal/*.h internal/*.hpp)
add_library(IRG
    src/irangegraph.cpp
    src/attribute_tree.cpp
    ${IRG_INTERNAL_HEADERS}
)
target_include_directories(IRG
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/irangegraph
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/internal
)
target_link_libraries(IRG
    PUBLIC Core
    PUBLIC Hnsw
    PUBLIC TableStorage
    PUBLIC OpenMP::OpenMP_CXX
    PUBLIC TBB::tbb
)
