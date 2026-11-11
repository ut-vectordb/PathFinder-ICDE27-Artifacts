#ifndef IRANGEGRAPH_H
#define IRANGEGRAPH_H

#include "storage.h"
#include "distance.h"
#include "graph.h"
#include "search_cache.h"

#include "hnsw/hnsw.h"

// from iRangeGraph github repo:
#include "construction.h"
#include "iRG_search.h"
#include "iRG_search_multi.h"


namespace ANNS {

    class iRG {
        public:
            iRG(bool verbose=true) : _verbose(verbose) {}
            ~iRG() = default;

            void build(std::shared_ptr<IStorage> base_storage,
                std::shared_ptr<DistanceHandler> distance_handler, 
                IdxType M, int ef_con, int n_threads);

            void build_node(iRangeGraph::TreeNode *u);

            iRangeGraph::TreeNode *find_child_containing(iRangeGraph::TreeNode *cur_node, IdxType pid) {
                for (int i = 0; i < cur_node->childs.size(); ++i) {
                    if (cur_node->childs[i]->lbound <= pid && cur_node->childs[i]->rbound >= pid) {
                        // only one child will satisfy this
                        return cur_node->childs[i];
                    }
                }
                return cur_node;
            };
            std::vector<IdxType> select_edge(IdxType pid, IdxType L_closed, IdxType R_closed, VisitedSet &visited_set);
            IdxType iterate_to_fixed_point__rangefilter_by_id(
                const char *query,
                IdxType l_closed, IdxType r_closed,
                std::shared_ptr<SearchCache> search_cache,
                std::vector<IdxType> *entry_pts = nullptr);

            std::vector<IdxType> select_edge_by_attr(
                IdxType pid, IdxType L_closed, IdxType R_closed,
                float *attrs, float L_attr, float R_attr,
                VisitedSet &visited_set);
            IdxType iterate_to_fixed_point__rangefilter_by_attr(
                const char *query,
                float *attrs, float L_attr, float R_attr,
                std::shared_ptr<SearchCache> search_cache,
                std::vector<IdxType> *entry_pts = nullptr);

            // stats and I/O
            void statistics();
            void save(const std::string& index_name);
            void load(const std::string& index_name,
                std::shared_ptr<IStorage> base_storage,
                std::shared_ptr<DistanceHandler> distance_handler, 
                IdxType M, int ef_con, int n_threads);

        // private:
            // data
            std::shared_ptr<IStorage> _base_storage, _query_storage;
            std::shared_ptr<DistanceHandler> _distance_handler;

            std::vector<Hnsw> indices;
            std::vector<std::shared_ptr<Graph>> graphs;

            iRangeGraph::SegmentTree *tree;

            // build parameters
            uint32_t _num_threads;
            int _M = 8;
            int _ef_con = 100;

            // for logs
            bool _verbose;
    };
}

#endif // IRANGEGRAPH_H