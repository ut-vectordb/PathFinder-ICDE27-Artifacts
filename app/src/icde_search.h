#pragma once
#include "run_search.h"
struct IcdeSearchOptions {
    std::string algorithm, pathseer_graph, cgif_clusters;
    bool favor_cache_sampling = false;
    bool favor_oor_results = false;
    float favor_delta_d = std::numeric_limits<float>::quiet_NaN();
};
// Pred is QueryType (conjunctive) or DisjunctiveQueryType (DNF); both expose
// check(const float* attr), which is all the ports use.
template<class Pred>
void test_search_icde(TableStorage&, std::shared_ptr<ANNS::IStorage>,
                     const std::vector<Pred>&, const std::string& hnsw_prefix,
                     int k,const std::vector<int>& efs,std::pair<ANNS::IdxType,float>* gt,
                     const IcdeSearchOptions&);
