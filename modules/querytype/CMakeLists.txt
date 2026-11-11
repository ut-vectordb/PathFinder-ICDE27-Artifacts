add_library(QueryType src/querytype.cpp)
target_include_directories(QueryType
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/querytype
)
