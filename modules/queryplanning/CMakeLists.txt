add_library(QueryPlanning src/queryplanning.cpp)
target_include_directories(QueryPlanning PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(QueryPlanning
    PUBLIC Core
    PUBLIC IRG
    PUBLIC Hnsw
    PUBLIC QueryType
    PUBLIC OpenMP::OpenMP_CXX
)
