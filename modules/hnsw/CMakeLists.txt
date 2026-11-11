add_library(Hnsw src/hnsw.cpp)
target_include_directories(Hnsw PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(Hnsw PUBLIC Core Boost::filesystem hnswlib)
