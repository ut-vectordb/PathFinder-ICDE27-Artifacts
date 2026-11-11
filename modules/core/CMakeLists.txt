add_library(Core
    src/distance.cpp
    src/storage.cpp
    src/utils.cpp
    src/search_queue.cpp
    src/filtered_scan.cpp
    src/trie.cpp
)
target_include_directories(Core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(Core PUBLIC Boost::filesystem TBB::tbb tsl)
