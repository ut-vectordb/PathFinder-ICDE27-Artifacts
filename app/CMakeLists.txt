set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

set(APP_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(EXP_HELPERS src/run_search.cpp src/run_search.h)
set(ICDE_SEARCH_HELPERS src/icde_search.cpp src/icde_search.h)
set(LIBS
    Core
    Boost::program_options
    IRG
    Hnsw
    TableStorage
    QueryPlanning
    QueryType
    Baselines
)

function(add_exp_target name)
    add_executable(${name} ${ARGN})
    target_include_directories(${name} PRIVATE ${APP_SRC_DIR})
    target_link_libraries(${name} PUBLIC ${LIBS})
endfunction()

# PathFinder and the baselines (OOR HNSW/Vamana, NaviX, pre-filtering, ACORN,
# and the FAVOR/CGIF/PathSeer search ports), on conjunctive and disjunctive filters.
add_exp_target(jul14_demo               src/jul14_demo.cpp ${EXP_HELPERS} ${ICDE_SEARCH_HELPERS})
add_exp_target(jul14_demo_disjunctive   src/jul14_demo_disjunctive.cpp ${EXP_HELPERS} ${ICDE_SEARCH_HELPERS})
add_exp_target(jul14_demo_irangegraph   src/jul14_demo_irangegraph.cpp ${EXP_HELPERS})
add_exp_target(jul14_demo_alpha         src/jul14_demo_alpha.cpp ${EXP_HELPERS})
add_exp_target(jul14_demo_index_borrowing src/jul14_demo_index_borrowing.cpp ${EXP_HELPERS})
add_exp_target(prefilter_benchmark      src/prefilter_benchmark.cpp)

add_subdirectory(updates)
