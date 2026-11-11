# Header-only search ports of FAVOR, CGIF, and PathSeer.
add_library(Baselines INTERFACE)
target_include_directories(Baselines INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
