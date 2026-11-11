add_library(TableStorage src/tablestorage.cpp src/tablestorage_utils.cpp)
target_include_directories(TableStorage
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/tablestorage
)
target_link_libraries(TableStorage PUBLIC Core btreeolc)
