#ifndef HNSW_H
#define HNSW_H

#include "storage.h"
#include "distance.h"
#include "graph.h"
#include "search_cache.h"

// #include "base_hnsw/hnswalg.h"
// #include "base_hnsw/hnswlib.h"
// namespace hnswlib = base_hnsw;


namespace ANNS {

    struct HNSWUpperLayer {
        IdxType enterpoint_node_;
        int maxlevel_;
        size_t n_links_per_element_;        // = (1 + M)
        std::vector<int> element_levels_;   // max level of node (inclusive; min = 0 inclusive)
        std::unordered_map<IdxType, std::vector<IdxType>> edgelist; // = L * (1+M)

        void save(const std::string& filename) const;
        void load(const std::string& filename);

        size_t get_index_size() const;
    };

    class Hnsw {
            
        public:
            Hnsw(bool verbose = true) { _verbose = verbose; };
            Hnsw(std::shared_ptr<IStorage> base_storage, std::shared_ptr<DistanceHandler> distance_handler,
                   std::shared_ptr<Graph> graph, IdxType entry_point) 
                        : _base_storage(base_storage), _distance_handler(distance_handler),
                        _graph(graph), _entry_point(entry_point), _verbose(false) {}
            ~Hnsw() = default;

            void build_HNSWLib(std::shared_ptr<IStorage> base_storage,
                       std::shared_ptr<DistanceHandler> distance_handler, 
                       std::shared_ptr<Graph> graph,
                       int M, int ef_construction,
                       int num_threads=1,
                       bool randomize=false,
                       bool merge_upper_levels=false);

            void build_NSW(std::shared_ptr<IStorage> base_storage,
                       std::shared_ptr<DistanceHandler> distance_handler, 
                       std::shared_ptr<Graph> graph,
                       int M, int ef_construction,
                       int num_threads=1);

            void search(std::shared_ptr<IStorage> base_storage, std::shared_ptr<IStorage> query_storage, 
                        std::shared_ptr<DistanceHandler> distance_handler, IdxType K, IdxType ef_search, 
                        uint32_t num_threads, std::pair<IdxType, float>* results, std::vector<IdxType>& num_cmps);

            void search_containment_warmup(
                std::shared_ptr<IStorage> base_storage,
                std::shared_ptr<IStorage> query_storage,
                std::shared_ptr<DistanceHandler> distance_handler, IdxType K,
                IdxType ef_search, uint32_t num_threads,
                std::pair<IdxType, float> *results,
                std::vector<IdxType> &num_cmps);

            std::map<std::pair<IdxType, IdxType>, int>
            search_containment_profiling(
                std::shared_ptr<IStorage> base_storage,
                std::shared_ptr<IStorage> query_storage,
                std::shared_ptr<DistanceHandler> distance_handler, IdxType K,
                IdxType ef_search, uint32_t num_threads,
                std::pair<IdxType, float> *results,
                std::vector<IdxType> &num_cmps);

            enum InsertType { NO_PRUNE, VAMANA_PRUNE, VAMANA_PRUNE_KEEP_ADDED };
            void select_and_insert(
                std::map<std::pair<ANNS::IdxType, ANNS::IdxType>, int> ve, int top_k,
                InsertType type=NO_PRUNE);

            void search_containment(
                std::shared_ptr<IStorage> base_storage,
                std::shared_ptr<IStorage> query_storage,
                std::shared_ptr<DistanceHandler> distance_handler, IdxType K,
                IdxType ef_search,
                bool enable_2hop,
                uint32_t num_threads,
                std::pair<IdxType, float> *results,
                std::vector<IdxType> &num_cmps);

            // search the graph
            IdxType get_entry_point() { return _entry_point; }
            IdxType iterate_to_fixed_point(const char* query, std::shared_ptr<SearchCache> search_cache, 
                                           bool record_expanded = false, IdxType target_id = -1, std::vector<IdxType> *entrypts = nullptr);
            IdxType iterate_to_fixed_point__infilter(
                const char *query,
                const std::vector<LabelType> &query_label,
                std::shared_ptr<SearchCache> search_cache,
                bool resort_to_2hop=false);

            IdxType iterate_to_fixed_point__rangefilter_by_id(
                const char *query,
                IdxType l_closed, IdxType r_closed,
                std::shared_ptr<SearchCache> search_cache);

            void iterate_to_fixed_point__rangefilter_by_id(
                const char *query,
                IdxType l_closed, IdxType r_closed,
                int &num_cmps, int &num_expanded, int &num_inserted,
                std::vector<double> &timers,
                std::shared_ptr<SearchCache> search_cache);

            IdxType iterate_to_fixed_point__infilter_bitmap(
                const char *query, char *bitmap,
                std::shared_ptr<SearchCache> search_cache);

            IdxType iterate_to_fixed_point__infilter_attr_mapped(
                const char *query, std::vector<std::vector<float>> &attributes,
                std::vector<float>& predicate,
                std::shared_ptr<SearchCache> search_cache,
                std::vector<IdxType> *entrypts = nullptr);

            void NSW_prune_neighbors(IdxType id,
                                   std::vector<Candidate> &candidates,
                                   std::vector<IdxType> &pruned_list,
                                   int maxdeg=-1);

            void vamana_prune_neighbors(IdxType id,
                                   std::vector<Candidate> &candidates,
                                   std::vector<IdxType> &pruned_list,
                                   int maxdeg=-1);

            IdxType iterate_to_fixed_point__infilter_adaptive(
                const char *query, const std::vector<LabelType> &query_label,
                std::shared_ptr<SearchCache> search_cache);

            IdxType iterate_to_fixed_point__infilter_profiling(
                const char *query, const std::vector<LabelType> &query_label,
                std::shared_ptr<SearchCache> search_cache,
                std::map<std::pair<IdxType, IdxType>, int> &virt_edges);

            // stats and I/O
            void statistics();
            void save(const std::string& index_path_prefix);
            void load(const std::string& index_path_prefix, std::shared_ptr<Graph> graph);

        // private:
            // data
            std::shared_ptr<IStorage> _base_storage, _query_storage;
            std::shared_ptr<DistanceHandler> _distance_handler;

            // build parameters
            uint32_t _num_threads;
            int _M;
            int _ef_construction;
            bool _randomize;
            bool _merge_upper_levels;

            float fixed_alpha = 1.0;

            // build the graph
            IdxType _entry_point;
            std::shared_ptr<Graph> _graph;
            std::shared_ptr<HNSWUpperLayer> _upper_layer;

            void link_HNSWLib(void * _alg_hnsw, bool randomize=false, bool merge_upper_levels=false);

            // for logs
            bool _verbose;
    };
}

#endif // HNSW_H