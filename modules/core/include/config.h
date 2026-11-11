#ifndef ANNS_CONFIG_H
#define ANNS_CONFIG_H

#include <cstdint>
#include <cstddef>


namespace ANNS {

    // type for storing the id of vectors
    using IdxType = uint32_t;

    // type for storing the label of vectors
    // using LabelType = uint16_t;
    using LabelType = uint32_t;     // for serf...

    // type for marks which are refreshed in each search
    using MarkType = uint16_t;

    enum DataType {
        FLOAT = 0,
        UINT8 = 1,
        INT8 = 2
    };

    enum Metric {
        L2 = 0,
        INNER_PRODUCT = 1,
        COSINE = 2
    };

    // default parameters
    namespace default_paras {
        const uint32_t NUM_THREADS = 1;

        // general graph indices
        const IdxType MAX_DEGREE = 64;
        const IdxType L_BUILD = 100;
        const IdxType L_SEARCH = 100;

        // for vamana
        const IdxType MAX_CANDIDATE_SIZE = 750;
        const float ALPHA = 1.2;
        const float GRAPH_SLACK_FACTOR = 1.3;

        // for hnsw
        const int HNSW_EF_CON = 200;
        const int HNSW_EF_SEARCH = 64;

        // for serf
        // default values can be grabbed from serf_halfbound.cc and serf_arbitrary.cc. They are the same. See "string indexk_str = "8";"
        const int INDEXK = 8;
        const int EF_CON = 100;
        const int EF_MAX = 500;
        const int EF_SEARCH = 64;   // string ef_search_str = "16,64,256";

        // for Unified Navigating Graph
        const IdxType NUM_ENTRY_POINTS = 16;
        const IdxType NUM_CROSS_EDGES = 6;
    }
}

#endif // ANNS_CONFIG_H