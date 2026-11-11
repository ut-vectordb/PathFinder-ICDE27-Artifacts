add_library(btreeolc btreeolc.cpp btreeolc.hpp EBR.hpp)
target_include_directories(btreeolc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
