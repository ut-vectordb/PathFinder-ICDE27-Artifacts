add_subdirectory(btreeolc)

add_library(hnswlib INTERFACE)
target_include_directories(hnswlib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/hnswlib)

add_library(tsl INTERFACE)
target_include_directories(tsl INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
