add_library(Vamana src/vamana.cpp)
target_include_directories(Vamana PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(Vamana PUBLIC Core Boost::filesystem)
