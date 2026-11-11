/**
 * @file hnsw.cpp
 * @author Tianming Wu
 * @brief Wrapper for HNSWLib, exposes interface for UNG
 * @date 2025-01-15
 */

#include <omp.h>
#include <set>
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <boost/filesystem.hpp>
#include "utils.h"
#include "hnsw/hnsw.h"

// for rdtsc:
#include <x86intrin.h>

#include "hnswlib.h"
// namespace hnswlib = base_hnsw;

namespace fs = boost::filesystem;



namespace ANNS {

    void Hnsw::build_NSW(std::shared_ptr<IStorage> base_storage,
                     std::shared_ptr<DistanceHandler> distance_handler,
                     std::shared_ptr<Graph> graph,
                     int M, int ef_construction,
                     int num_threads) {
        _num_threads = num_threads;

        if (_verbose) {
            std::cout << "Building NSW index ..." << std::endl;
            std::cout << "- num_threads: " << num_threads << std::endl;
            std::cout << "- M: " << M << std::endl;
            std::cout << "- ef_construction: " << ef_construction << std::endl;
        }

        _base_storage = base_storage;
        _distance_handler = distance_handler;
        _graph = graph;
        _M = M;
        _ef_construction = ef_construction;
        _randomize = false;
        int dim = base_storage->get_dim();
        int num_points = base_storage->get_num_points();

        // link the graph.
        if (_verbose) {
            std::cout << "Building index containing " << num_points << " Points, dim = " << dim << "." << std::endl;
        }

        std::random_device rd;
        std::default_random_engine g(rd());

        // for NSW, the user has to choose entrypoints themselves
        _entry_point = -1;

        SearchCacheList search_cache_list(num_threads + 5, num_points, _ef_construction);

        int interval = std::clamp(num_points / 100, 1, 10000);

        // if (num_points < 1000)
        //     num_threads = 1;
        omp_set_num_threads(num_threads);
        #pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads)
        for (int i = 1; i < num_points; i++) {
            if (_verbose && i % interval == 0)
                std::cout << "\n" << (100.0 * i) / num_points << "%" << std::flush;

            IdxType pid = i;
            std::shared_ptr<SearchCache> search_cache = search_cache_list.get_free_cache();
            std::vector<IdxType> entry_pts;

            // starting point choice like iRangeGraph codebase
            std::uniform_int_distribution<int> u_start(0, i - 1);
            for (int k = 0; k < std::min(3, i); k++) {
                int enterpid = u_start(g);
                entry_pts.emplace_back(enterpid);
            }

            char *query = base_storage->get_vector(i);

            this->iterate_to_fixed_point(query, search_cache, true, i, &entry_pts);

            std::vector<IdxType> new_neighbors;
            if (this->fixed_alpha == 1.0) {
                this->NSW_prune_neighbors(i, search_cache->expanded_list, new_neighbors, this->_M);
            } else {
                this->vamana_prune_neighbors(i, search_cache->expanded_list, new_neighbors, this->_M);
            }
            {
                std::lock_guard<std::mutex> lock(_graph->neighbor_locks[i]);
                this->_graph->neighbors[i] = new_neighbors;
            }

            // insert the reversed edge
            for (int dst : new_neighbors) {
                bool need_prune = false;
                std::vector<Candidate> candidates;

                // try to add edge dst -> src
                {
                    std::lock_guard<std::mutex> lock(_graph->neighbor_locks[dst]);
                    auto& dst_neighbors = _graph->neighbors[dst];
                    if (std::find(dst_neighbors.begin(), dst_neighbors.end(), i) == dst_neighbors.end()) {
                        if (dst_neighbors.size() < M) 
                            dst_neighbors.push_back(i);
                        else {
                            candidates.reserve(dst_neighbors.size() + 1);
                            for (auto& neighbor : dst_neighbors)
                                candidates.emplace_back(neighbor, 0);
                            candidates.emplace_back(i, 0);
                            need_prune = true;
                        }
                    }
                }

                // prune the neighbors of dst
                if (need_prune) {
                    for (auto& candidate : candidates)
                        candidate.distance = _distance_handler->compute(_base_storage->get_vector(dst), 
                                                                        _base_storage->get_vector(candidate.id), dim);
                    std::vector<IdxType> new_dst_neighbors;
                    if (this->fixed_alpha == 1.0) {
                        NSW_prune_neighbors(dst, candidates, new_dst_neighbors, _M);
                    } else {
                        vamana_prune_neighbors(dst, candidates, new_dst_neighbors, _M);
                    }
                    {
                        std::lock_guard<std::mutex> lock(_graph->neighbor_locks[dst]);
                        _graph->neighbors[dst] = new_dst_neighbors;
                    }
                }
            }

            search_cache_list.release_cache(search_cache);
        }

        if (_verbose)
            this->statistics();
    }

    void Hnsw::build_HNSWLib(std::shared_ptr<IStorage> base_storage,    // this implements get_vector(id)
                     std::shared_ptr<DistanceHandler> distance_handler,
                     std::shared_ptr<Graph> graph,      // this provides .neighbors = std::vector<IdxType>* neighbors

                    int M,          // Tightly connected with internal dimensionality of the data
                                    // strongly affects the memory consumption
                    int ef_construction,// Controls index search speed/build speed tradeoff
                    int num_threads,
                    bool randomize,
                    bool merge_upper_levels
    ) {

        _num_threads = num_threads;
        
        if (_verbose) {
            std::cout << "Building Hnsw index ..." << std::endl;
            std::cout << "- num_threads: " << num_threads << std::endl;
            std::cout << "- M: " << M << std::endl;
            std::cout << "- ef_construction: " << ef_construction << std::endl;
        }

        _base_storage = base_storage;
        _distance_handler = distance_handler;
        _graph = graph;

        _M = M;
        _ef_construction = ef_construction;
        _randomize = randomize;
        _merge_upper_levels = merge_upper_levels;

        int dim = base_storage->get_dim();               // Dimension of the elements
        int max_elements = base_storage->get_num_points() + 10;   // Maximum number of elements, should be known beforehand

        ///////////////////

        if (_verbose)
            std::cout << "Linking the graph ..." << std::endl;
        // Initing index
        // hnswlib::L2Space space(dim);
        hnswlib::SpaceInterface<float> *space;
        if (base_storage->get_data_type() == FLOAT) {
            space = new hnswlib::L2Space(dim);
        } else if (base_storage->get_data_type() == UINT8) {
            space = new hnswlib::L2Space_U8_pretend_to_be_float(dim);
        }
        hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(space, max_elements, M, ef_construction);
        auto link_start = std::chrono::high_resolution_clock::now();
        link_HNSWLib(alg_hnsw, _randomize, _merge_upper_levels);
        auto link_end = std::chrono::high_resolution_clock::now();
        auto link_time = std::chrono::duration_cast<std::chrono::milliseconds>(link_end - link_start).count();
        std::cout << "@@@ HNSWLib build time = " << link_time / 1000.00 << " seconds." << std::endl;

        if (_verbose)
            std::cout << "Getting entry point ..." << std::endl;
        auto HNSWLib_entry = alg_hnsw->getExternalLabel(alg_hnsw->enterpoint_node_);
        _entry_point = HNSWLib_entry;
        if (_verbose)
            std::cout << "Entry_point = " << _entry_point << std::endl;

        if (_verbose)
            std::cout << "Removing HNSWLib object." << std::endl;
        delete alg_hnsw;
        delete space;

        if (_verbose)
            std::cout << "Finish." << std::endl << SEP_LINE;

        // call statistics!
        this->statistics();
    }


    void Hnsw::link_HNSWLib(void * _alg_hnsw, bool randomize, bool merge_upper_levels) {
        hnswlib::HierarchicalNSW<float> * hnsw = (hnswlib::HierarchicalNSW<float> *)_alg_hnsw;
        auto num_points = _base_storage->get_num_points();
        auto dim = _base_storage->get_dim();

        // Add data to index
        std::cout << "Building index containing " << num_points << " Points, dim = " << dim << "." << std::endl;
        std::vector<int> randomizer(num_points);
        for (int i = 0; i < num_points; ++i) { randomizer[i] = i; }
        std::random_device rd;
        std::mt19937 g(rd());
        if (randomize)
            std::shuffle(randomizer.begin(), randomizer.end(), g);

        omp_set_num_threads(_num_threads);
#pragma omp parallel for schedule(dynamic, 1)
        for (int u = 0; u < num_points; u++) {
            if (num_points > 20 && u % (num_points / 10) == 0) { // Print progress every 10%
                std::cout << "Build Progress: " << (u * 100.00) / num_points << "%\n";
            }

            int u_ = randomizer[u];
            const char* query = _base_storage->get_vector(u_);
            hnsw->addPoint(query, u_);
        }

        std::cout << "\n\nFinished HNSWLib HNSW graph construction. "
                        "Transferring data into ANNS::Graph..."
                    << std::endl;

        // Transfer data to ANNS::Graph
        if (!merge_upper_levels) {
            // put the data into `std::shared_ptr<HNSWUpperLayer> _upper_layer`
            _upper_layer = std::make_shared<HNSWUpperLayer>();
            _upper_layer->maxlevel_ = hnsw->maxlevel_;
            _upper_layer->n_links_per_element_ = hnsw->size_links_per_element_ / sizeof(hnswlib::tableint);
            _upper_layer->element_levels_.resize(num_points);
            for (int u = 0; u < num_points; u++) {
                hnswlib::tableint internal_id = hnsw->label_lookup_[u];
                int L = hnsw->element_levels_[internal_id];
                _upper_layer->element_levels_[u] = L;
                if (L > 0)
                    _upper_layer->edgelist.emplace(u, L * _upper_layer->n_links_per_element_);
            }
            _upper_layer->enterpoint_node_ = hnsw->getExternalLabel(hnsw->enterpoint_node_);
        }

        for (int u = 0; u < num_points; u++) {
            if (num_points > 20 && u % (num_points / 10) == 0) { // Print progress every 10%
                std::cout << "Transfer Progress: " << (u * 100) / num_points << "%\n";
            }

            std::vector<IdxType> neighbors;
            std::set<IdxType> added;

            // the following code retrieves the neighbor data from HNSWLib object, and stores it in ANNS::Graph object

            hnswlib::tableint internal_id = hnsw->label_lookup_[u];
            hnswlib::linklistsizeint *data;
            hnswlib::tableint *datal;
            int size;

            // copy level 0 of HNSW into Graph, ignoring Hierarchy
            data = hnsw->get_linklist0(internal_id);
            size = hnsw->getListCount(data);
            datal = (hnswlib::tableint *)(data + 1);
            for (int j = 0; j < size; j++) {
                hnswlib::tableint cand = datal[j];
                if (added.count(cand) == 0) {
                    neighbors.push_back(cand);
                    added.insert(cand);
                }
            }

            // add other levels into neighbor[]. Then use std::unique to dedup.
            if (merge_upper_levels) {
                int elemLevel = hnsw->element_levels_[internal_id];
                for (int level = elemLevel; level > 0; level--) {
                    data = hnsw->get_linklist(internal_id, level);
                    size = hnsw->getListCount(data);
                    datal = (hnswlib::tableint *)(data + 1);
                    for (int j = 0; j < size; j++) {
                        hnswlib::tableint cand = datal[j];
                        if (added.count(cand) == 0) {
                            neighbors.push_back(cand);
                            added.insert(cand);
                        }
                    }
                }
            } else if (int elemLevel = hnsw->element_levels_[internal_id]; elemLevel > 0) {
                auto &storage_location = _upper_layer->edgelist.at(u);
                for (int level = elemLevel; level > 0; level--) {
                    data = hnsw->get_linklist(internal_id, level);
                    size = hnsw->getListCount(data);
                    datal = (hnswlib::tableint *)(data + 1);
                    storage_location[_upper_layer->n_links_per_element_ * (level-1)] = size;
                    for (int j = 0; j < size; j++) {
                        hnswlib::tableint cand = datal[j];
                        IdxType v = hnsw->getExternalLabel(cand);
                        storage_location[_upper_layer->n_links_per_element_ * (level-1) + 1 + j] = v;
                    }
                }
            }

            // convert HNSWLib internal node ID to actual node ID
            for (int j = 0; j < neighbors.size(); j++) {
                neighbors[j] = hnsw->getExternalLabel(neighbors[j]);
            }

            // now that we have the neighbors, we can add them to the Graph object
            {
                std::lock_guard<std::mutex> lock(_graph->neighbor_locks[u]);
                _graph->neighbors[u] = neighbors;
            }
        }

    }

    size_t HNSWUpperLayer::get_index_size() const {
        size_t ret = 0;
        if (!element_levels_.empty()) {
            ret += sizeof(element_levels_[0]) * element_levels_.size();
        }
        for (const auto& [node, neighbors] : edgelist) {
            (void)node;
            if (!neighbors.empty()) {
                ret += neighbors.size() * sizeof(neighbors[0]);
            }
        }
        return ret;
    }

    void HNSWUpperLayer::save(const std::string& filename) const {
        std::ofstream ofs(filename);
        if (!ofs) {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        // Scalars
        ofs << enterpoint_node_ << "\n";
        ofs << maxlevel_ << "\n";
        ofs << n_links_per_element_ << "\n";

        // element_levels_
        ofs << element_levels_.size() << "\n";
        for (size_t i = 0; i < element_levels_.size(); ++i) {
            ofs << element_levels_[i];
            if (i + 1 < element_levels_.size()) ofs << " ";
        }
        ofs << "\n";

        // edgelist
        ofs << edgelist.size() << "\n";
        for (const auto& [node, neighbors] : edgelist) {
            ofs << node << " " << neighbors.size();
            for (const auto& nb : neighbors) {
                ofs << " " << nb;
            }
            ofs << "\n";
        }
    }

    void HNSWUpperLayer::load(const std::string& filename) {
        std::ifstream ifs(filename);
        if (!ifs) {
            throw std::runtime_error("Failed to open file for reading: " + filename);
        }

        // Scalars
        ifs >> enterpoint_node_;
        ifs >> maxlevel_;
        ifs >> n_links_per_element_;

        // element_levels_
        size_t num_levels = 0;
        ifs >> num_levels;
        element_levels_.resize(num_levels);
        for (size_t i = 0; i < num_levels; ++i) {
            ifs >> element_levels_[i];
        }

        // edgelist
        size_t num_nodes = 0;
        ifs >> num_nodes;
        edgelist.clear();

        for (size_t i = 0; i < num_nodes; ++i) {
            IdxType node;
            size_t num_neighbors;
            ifs >> node >> num_neighbors;

            auto& neighbors = edgelist[node];
            neighbors.resize(num_neighbors);
            for (size_t j = 0; j < num_neighbors; ++j) {
                ifs >> neighbors[j];
            }
        }
    }


    IdxType Hnsw::iterate_to_fixed_point(const char* query, std::shared_ptr<SearchCache> search_cache, 
                                           bool record_expanded, IdxType target_id, std::vector<IdxType> *entrypts) {
        auto dim = _base_storage->get_dim();
        auto& search_queue = search_cache->search_queue;
        auto& visited_set = search_cache->visited_set;
        auto& expanded_list = search_cache->expanded_list;
        search_queue.clear();
        visited_set.clear();
        expanded_list.clear();
        std::vector<IdxType> neighbors;
        
        // entry point
        IdxType num_cmps = 0;
        if (entrypts == nullptr && _entry_point != -1) {
            search_queue.insert(_entry_point, _distance_handler->compute(query, _base_storage->get_vector(_entry_point), dim));
            visited_set.set(_entry_point);
            num_cmps = 1;
        } else if (entrypts == nullptr) {
            if (record_expanded) {
                throw std::runtime_error("It is ill-advised to choose entry pts randomly when building the graph, because there is a high chance that one node will choose un-populated nodes as neighbors and result in poor connectivity.");
            }
            for (int i = 0; i < 15; i++) {
                IdxType pid = rand() % _base_storage->get_num_points();
                search_queue.insert(pid, _distance_handler->compute(query, _base_storage->get_vector(pid), dim));
                visited_set.set(pid);
                ++num_cmps;
            }
        } else {
            for (int i = 0; i < entrypts->size(); i++) {
                search_queue.insert(entrypts->at(i), _distance_handler->compute(query, _base_storage->get_vector(entrypts->at(i)), dim));
                visited_set.set(entrypts->at(i));
                ++num_cmps;
            }
        }

        // greedily expand closest nodes
        while (search_queue.has_unexpanded_node()) {
            const Candidate& cur = search_queue.get_closest_unexpanded();
            if (record_expanded && target_id != cur.id)
                expanded_list.push_back(cur);

            // iterate neighbors
            {
                std::lock_guard<std::mutex> lock(_graph->neighbor_locks[cur.id]);
                neighbors = _graph->neighbors[cur.id];
            }
            for (auto i=0; i<neighbors.size(); ++i) {

                // prefetch
                // if (i+1 < neighbors.size()) {
                //     visited_set.prefetch(neighbors[i+1]);
                //     _base_storage->prefetch_vec_by_id(neighbors[i+1]);
                // }

                // skip if visited
                auto& neighbor = neighbors[i];
                if (visited_set.check(neighbor)) 
                    continue;
                visited_set.set(neighbor);

                // push to search queue
                search_queue.insert(neighbor, _distance_handler->compute(query, _base_storage->get_vector(neighbor), dim));
                num_cmps++;
            }
        }
        return num_cmps;
    }

// #define OPTIM_STD_SLOW

    IdxType Hnsw::iterate_to_fixed_point__infilter(
            const char *query,
            const std::vector<LabelType> &query_label,
            std::shared_ptr<SearchCache> search_cache,
            bool resort_to_2hop) {
        auto dim = _base_storage->get_dim();
        auto& search_queue = search_cache->search_queue;
        auto& visited_set = search_cache->visited_set;
        search_queue.clear();
        visited_set.clear();

        // entry point
        IdxType num_cmps = 0;

        search_queue.insert(_entry_point, _distance_handler->compute(query, _base_storage->get_vector(_entry_point), dim));
        visited_set.set(_entry_point);
        num_cmps++;

        // more entry points ...
        // throw std::invalid_argument("This part of code is not renovated");
        // int countr = 5;
        // for (auto n : _graph->neighbors[_entry_point]) {
        //     auto &cur_label_set = _base_storage->get_label_set(n);
        //     bool cur_good = std::includes(cur_label_set.begin(), cur_label_set.end(), query_label.begin(), query_label.end());
        //     if (cur_good) {
        //         search_queue.insert(n, _distance_handler->compute(query, _base_storage->get_vector(n), dim));
        //         num_cmps++;

        //         countr--;
        //         if (countr <= 0) break;
        //     }
        // }

        // greedily expand closest nodes
        while (search_queue.has_unexpanded_node()) {
            const Candidate& cur = search_queue.get_closest_unexpanded();
            // if (cur.distance > search_queue.lower_bound()) break;    // SearchQueue automatically handles this

            // iterate neighbors
            std::vector<IdxType> &neighbors = _graph->neighbors[cur.id];
            int num_viable_1hop = 0;

            std::vector<IdxType> good_neighbors;
            // std::vector<uint8_t> good_nbr_mask;
            // good_nbr_mask.resize(neighbors.size());
            for (auto i=0; i<neighbors.size(); ++i) {
                if (visited_set.check(neighbors[i])) { continue; }
                auto& neighbor = neighbors[i];
                auto &nbr_label_set = _base_storage->get_label_set(neighbor);
                bool nbr_good = std::includes(nbr_label_set.begin(), nbr_label_set.end(), query_label.begin(), query_label.end());
                // good_nbr_mask[i] = nbr_good;
                if (nbr_good) good_neighbors.push_back(neighbor);
            }
            // for (auto i=0; i<neighbors.size(); ++i) {
            for (auto i_g = 0; i_g < good_neighbors.size(); i_g++) {
                // skip if visited
                auto neighbor = good_neighbors[i_g];
                // if (visited_set.check(neighbors[i])) { continue; }
                if (visited_set.check(neighbor)) { continue; }

                // prefetch
                // if (i_g+1 < good_neighbors.size()) {
                //     visited_set.prefetch(good_neighbors[i_g+1]);
                //     _base_storage->prefetch_vec_by_id(good_neighbors[i_g+1]);
                // }

                // auto& neighbor = neighbors[i];

                // in-filtering  --  only visit neighbor if it passes the filter.
                bool nbr_good = true;
                if (nbr_good) {
                    visited_set.set(neighbor);
                    // push to search queue
                    float dist = _distance_handler->compute(query, _base_storage->get_vector(neighbor), dim);
                    if (search_queue.can_insert(dist)) {
                        num_viable_1hop++;
                        search_queue.insert(neighbor, dist);
                    } else {
                        // this neighbor is no good: it's too far to be accepted into the search queue.
                    }
                    num_cmps++;
                }
            }

            if (resort_to_2hop && num_viable_1hop == 0)
            for (auto i=0; i<neighbors.size(); ++i) {
                auto& neighbor = neighbors[i];
                std::vector<IdxType> &twohop = _graph->neighbors[neighbor];
                for (auto j = 0; j < twohop.size(); j++) {
                    if (visited_set.check(twohop[j])) { continue; }
                    auto &nj = twohop[j];
                    auto &nj_label_set = _base_storage->get_label_set(nj);
                    bool nj_good = std::includes(nj_label_set.begin(), nj_label_set.end(), query_label.begin(), query_label.end());

                    if (nj_good) {
                        visited_set.set(nj);
                        float dist = _distance_handler->compute(query, _base_storage->get_vector(nj), dim);
                        if (search_queue.can_insert(dist)) {
                            search_queue.insert(nj, dist);
                        }
                        num_cmps++;
                    }
                }
            }
        }
        // std::cout << num_cmps << std::endl;
        return num_cmps;
    }

    IdxType Hnsw::iterate_to_fixed_point__rangefilter_by_id(
            const char *query,
            IdxType l, IdxType r,
            std::shared_ptr<SearchCache> search_cache) {
        int n_cmps, n_expanded, n_inserted;
        std::vector<double> timers;
        this->iterate_to_fixed_point__rangefilter_by_id(query,l,r,n_cmps,n_expanded,n_inserted,timers,search_cache);
        return n_cmps;
    }


// #define timeNow() std::chrono::high_resolution_clock::now()
// #define durationSec(a) (std::chrono::duration_cast<std::chrono::nanoseconds>((a)).count() / 1.0e9)
#define TSC_nominal_freq (2.6e9)
inline uint64_t timeNow() { _mm_lfence(); uint64_t tsc = __rdtsc(); _mm_lfence(); return tsc; }
// inline uint64_t timeNow() { return __rdtsc(); }
#define durationSec(a) ((a) / TSC_nominal_freq)
// __rdtsc()
// _mm_lfence();  // optionally wait for earlier insns to retire before reading the clock
// uint64_t tsc = __rdtsc();
// _mm_lfence();  // optionally block later instructions until rdtsc retires

// #define USE_HASHMAP
// HASHMAP returns very long time consumption in insertion

    void Hnsw::iterate_to_fixed_point__rangefilter_by_id(
            const char *query,
            IdxType l, IdxType r,
            int &num_cmps,
            int &num_expanded,
            int &num_inserted,
            std::vector<double> &timers,
            std::shared_ptr<SearchCache> search_cache) {

        timers.clear();
        timers.resize(10, 0.00);
auto start0 = timeNow();
        /**
         * 0. total
         * 1. total time in the main loop
         * 2. accessing neighbors
         * 3. check visited
         * 4. prefetch
         * 5. computing the similarity scores
         * 6. set visited
         * 7. maintaining the efSearch queue
         * 
         * 9. known overhead due to bad code
         */
        auto dim = _base_storage->get_dim();
        int ef_search = search_cache->search_queue.capacity();

        typedef std::pair<float, IdxType> pfi;
        std::priority_queue<pfi> search_queue_minus;  // top = best one, insert with minus sign
        std::priority_queue<pfi> best_candidates;     // top = worst one; size is fixed
#ifndef USE_HASHMAP
        auto& visited_set = search_cache->visited_set;// meaning: already computed distance, already tried insert
        visited_set.clear();
        visited_set.set(_entry_point);
#else
        std::unordered_set<IdxType> visited_hashmap;
        visited_hashmap.insert(_entry_point);
#endif

        num_cmps = 1;
        num_expanded = 0;
        num_inserted = 1;
        auto vec = _base_storage->get_vector(_entry_point);
        float dist = _distance_handler->compute(query, vec, dim);
        search_queue_minus.emplace(-dist, _entry_point);
        best_candidates.emplace(dist, _entry_point);

auto start1 = timeNow();
        while (search_queue_minus.size()) {
            pfi cur = search_queue_minus.top();
            search_queue_minus.pop();
            if (best_candidates.size() >= ef_search && -cur.first > best_candidates.top().first) {
                break;
            }

            std::vector<IdxType> &neighbors = _graph->neighbors[cur.second];

            ++num_expanded;

auto start2 = timeNow();    // 0 ≈ 1 ≈ 2
            for (int i = 0; i < neighbors.size(); i++) {
auto start3 = timeNow();
                auto& nbr = neighbors[i];
#ifndef USE_HASHMAP
                bool check_result = visited_set.check(nbr);
#else
                bool check_result = (visited_hashmap.count(nbr) > 0);
#endif
auto start4 = timeNow();
timers[3] += durationSec(start4 - start3);
                if (check_result) { continue; }

                // if (i+1 < neighbors.size()) {
                //     visited_set.prefetch(neighbors[i+1]);
                //     _base_storage->prefetch_vec_by_id(neighbors[i+1]);
                // }
auto start5 = timeNow();
timers[4] += durationSec(start5 - start4);

                if (!(l <= nbr && nbr <= r)) { continue; }

                float dist = _distance_handler->compute(query, _base_storage->get_vector(nbr), dim);
auto start6 = timeNow();
timers[5] += durationSec(start6 - start5);
#ifndef USE_HASHMAP
                visited_set.set(nbr);
#else
                visited_hashmap.insert(nbr);
#endif
                num_cmps++;
auto start7 = timeNow();
timers[6] += durationSec(start7 - start6);
                // if (search_queue.can_insert(dist)) { search_queue.insert(nbr, dist); }
                if (best_candidates.size() < ef_search || dist < best_candidates.top().first) {
                    num_inserted++; // NOTE: this is less than num_cmps
                    best_candidates.emplace(dist, nbr);
                    search_queue_minus.emplace(-dist, nbr);
                    while (best_candidates.size() > ef_search) best_candidates.pop();   // worst ones pop first
                }
timers[7] += durationSec(timeNow() - start7);
            }
timers[2] += durationSec(timeNow() - start2);
        }
timers[1] = durationSec(timeNow() - start1);

auto start9 = timeNow();
        search_cache->search_queue.clear();
        while (best_candidates.size()) {
            search_cache->search_queue.insert(best_candidates.top().second, best_candidates.top().first);
            best_candidates.pop();
        }
timers[9] = durationSec(timeNow() - start9);

timers[0] = durationSec(timeNow() - start0);
    }

    IdxType Hnsw::iterate_to_fixed_point__infilter_bitmap(
            const char *query,
            char *bitmap,
            std::shared_ptr<SearchCache> search_cache) {

        auto dim = _base_storage->get_dim();
        auto& search_queue = search_cache->search_queue;
        auto& visited_set = search_cache->visited_set;
        search_queue.clear();
        visited_set.clear();

        IdxType num_cmps = 1;
        auto vec = _base_storage->get_vector(_entry_point);
        float dist = _distance_handler->compute(query, vec, dim);
        search_queue.insert(_entry_point, dist);
        visited_set.set(_entry_point);

        while (search_queue.has_unexpanded_node()) {
            const Candidate& cur = search_queue.get_closest_unexpanded();
            std::vector<IdxType> &neighbors = _graph->neighbors[cur.id];

            for (int i = 0; i < neighbors.size(); i++) {
                auto& nbr = neighbors[i];
                if (visited_set.check(nbr)) { continue; }

                if (!(bitmap[i])) { continue; }

                visited_set.set(nbr);
                float dist = _distance_handler->compute(query, _base_storage->get_vector(nbr), dim);
                num_cmps++;
                if (search_queue.can_insert(dist)) { search_queue.insert(nbr, dist); }
            }
        }
        return num_cmps;
    }

    IdxType Hnsw::iterate_to_fixed_point__infilter_attr_mapped(
        const char *query, std::vector<std::vector<float>> &attributes, std::vector<float> &predicate, std::shared_ptr<SearchCache> search_cache, std::vector<IdxType> *entrypts)
    {

        std::shared_ptr<ANNS::MappedStorage<float>> storage_mapped = std::dynamic_pointer_cast<ANNS::MappedStorage<float>>(_base_storage);
        if (!storage_mapped) {
            throw "storage conversion failed!"s;
        }

        auto dim = _base_storage->get_dim();
        auto& search_queue = search_cache->search_queue;
        auto& visited_set = search_cache->visited_set;
        search_queue.clear();
        visited_set.clear();

        IdxType num_cmps = 0;
        if (entrypts == nullptr) {
            search_queue.insert(_entry_point, _distance_handler->compute(query, _base_storage->get_vector(_entry_point), dim));
            visited_set.set(_entry_point);
            num_cmps = 1;
        } else {
            for (int i = 0; i < entrypts->size(); i++) {
                search_queue.insert(entrypts->at(i), _distance_handler->compute(query, _base_storage->get_vector(entrypts->at(i)), dim));
                visited_set.set(entrypts->at(i));
                ++num_cmps;
            }
        }

        auto predicate_check = [&](ANNS::IdxType nbr) {
            int mapped_id = storage_mapped->map_new_to_old(nbr);
            for (int i = 0; i < attributes.size(); i += 1) {
                if (!(predicate[i*2] <= attributes[i][mapped_id]
                    && attributes[i][mapped_id] < predicate[i*2+1])) {
                    return false;
                }
            }
            return true;
        };

        while (search_queue.has_unexpanded_node()) {
            const Candidate& cur = search_queue.get_closest_unexpanded();
            std::vector<IdxType> &neighbors = _graph->neighbors[cur.id];

            for (int i = 0; i < neighbors.size(); i++) {
                auto& nbr = neighbors[i];
                if (visited_set.check(nbr)) { continue; }

                bool pass = predicate_check(nbr);

                if (!pass) { continue; }

                visited_set.set(nbr);
                float dist = _distance_handler->compute(query, _base_storage->get_vector(nbr), dim);
                num_cmps++;
                if (search_queue.can_insert(dist)) { search_queue.insert(nbr, dist); }
            }
        }

        return num_cmps;
    }

    void Hnsw::NSW_prune_neighbors(IdxType id, std::vector<Candidate>& candidates, std::vector<IdxType>& pruned_list, int maxdeg) {
        // auto dim = _base_storage->get_dim();
        pruned_list.clear();
        if (maxdeg == -1) maxdeg = _M;
        pruned_list.reserve(maxdeg);

        // init candidates
        auto closest = [](const Candidate &a, const Candidate &b) { return a.distance < b.distance; };
        std::sort(candidates.begin(), candidates.end(), closest);

        std::vector<IdxType> discarded_list;

        // prune neighbors
        for (int i = 0; i < candidates.size() && pruned_list.size() < maxdeg; ++i) {
            if (candidates[i].id == id) continue;

            bool good = true;

            for (int j = 0; j < pruned_list.size(); j++) {
                int id1 = candidates[i].id;
                int id2 = pruned_list[j];
                float curdist = _distance_handler->compute(_base_storage->get_vector(id1), _base_storage->get_vector(id2), _base_storage->get_dim());
                if (curdist < candidates[i].distance) {
                    good = false;
                    discarded_list.push_back(candidates[i].id);
                    break;
                }
            }

            if (good)
                pruned_list.push_back(candidates[i].id);
        }

        // bool keepPrunedConnections = true;
        // if (keepPrunedConnections && pruned_list.size() < maxdeg) {
        //     for (int i = 0; i < discarded_list.size() && pruned_list.size() < maxdeg; ++i) {
        //         pruned_list.push_back(discarded_list[i]);
        //     }
        // }
    }

    void Hnsw::vamana_prune_neighbors(IdxType id, std::vector<Candidate>& candidates, std::vector<IdxType>& pruned_list, int maxdeg) {
        auto dim = _base_storage->get_dim();
        pruned_list.clear();
        if (maxdeg == -1) maxdeg = _M;
        pruned_list.reserve(maxdeg);

        // float fixed_alpha = 1.2;

        // init candidates
        std::sort(candidates.begin(), candidates.end(),
                    [](const Candidate &a, const Candidate &b) { return a.distance < b.distance; });

        IdxType _max_candidate_size = 750;
        auto candidate_size = std::min((IdxType)(candidates.size()), _max_candidate_size);
        // auto candidate_size = (IdxType)(candidates.size());

        // init occlude factor
        std::vector<float> occlude_factor;
        occlude_factor.clear();
        occlude_factor.insert(occlude_factor.end(), candidate_size, 0.0f);

        // prune neighbors
        float cur_alpha = 1;
        while (cur_alpha <= fixed_alpha && pruned_list.size() < maxdeg) {
            for (auto i = 0; i < candidate_size && pruned_list.size() < maxdeg; ++i) {
                if (occlude_factor[i] > cur_alpha)
                    continue;

                // set to float::max so that is not considered again
                occlude_factor[i] = std::numeric_limits<float>::max();
                if (candidates[i].id != id)
                    pruned_list.push_back(candidates[i].id);

                // update occlude factor for the following candidates
                for (auto j = i + 1; j < candidate_size; ++j) {
                    if (occlude_factor[j] > fixed_alpha)
                        continue;
                    auto distance_ij = _distance_handler->compute(
                        _base_storage->get_vector(candidates[i].id),
                        _base_storage->get_vector(candidates[j].id), dim);
                    occlude_factor[j] = (distance_ij == 0) ?
                                        std::numeric_limits<float>::max()
                                        : std::max(occlude_factor[j], candidates[j].distance / distance_ij);
                }
            }
            cur_alpha *= 1.2f;
        }
    }

    // this function modifies the graph structure on the search trail
    IdxType Hnsw::iterate_to_fixed_point__infilter_adaptive(
            const char *query,
            const std::vector<LabelType> &query_label,
            std::shared_ptr<SearchCache> search_cache) {
        auto dim = _base_storage->get_dim();
        auto& search_queue = search_cache->search_queue;
        auto& visited_set = search_cache->visited_set;
        search_queue.clear();
        visited_set.clear();

        // for adaptively updating the graph
        std::vector<IdxType> compared_list;     // all nodes put into search queue
        std::vector<IdxType> compared_1hop_list;
        std::vector<IdxType> compared_2hop_list;
        std::vector<IdxType> expanded_list;     // all nodes put into search queue AND expanded
        int n_insert_1hop = 0, n_insert_2hop = 0;
        // std::priority_queue<Candidate> top_compared_list;

        float entry_dist = _distance_handler->compute(query, _base_storage->get_vector(_entry_point), dim);
        search_queue.insert(_entry_point, entry_dist);
        compared_list.push_back(_entry_point);
        compared_1hop_list.push_back(_entry_point);
        // top_compared_list.emplace(_entry_point, entry_dist);
        IdxType num_cmps = 1;
        visited_set.set(_entry_point);

        while (search_queue.has_unexpanded_node()) {
            const Candidate& cur = search_queue.get_closest_unexpanded();
            expanded_list.push_back(cur.id);

            std::vector<IdxType> &neighbors = _graph->neighbors[cur.id];
            auto &cur_label_set = _base_storage->get_label_set(cur.id);

            // all 1-hops
            int num_viable_1hop = 0;

            for (auto i=0; i<neighbors.size(); ++i) {
                auto& neighbor = neighbors[i];
                auto &nbr_label_set = _base_storage->get_label_set(neighbor);
                bool nbr_good = std::includes(nbr_label_set.begin(), nbr_label_set.end(), query_label.begin(), query_label.end());

                if (nbr_good && !visited_set.check(neighbor)) {
                    visited_set.set(neighbor);
                    float dist = _distance_handler->compute(query, _base_storage->get_vector(neighbor), dim);
                    if (search_queue.can_insert(dist)) {
                        num_viable_1hop++;
                        n_insert_1hop++;
                        search_queue.insert(neighbor, dist);
                    } else {
                        // this neighbor is no good: it's too far to be accepted into the search queue.
                    }
                    compared_list.push_back(neighbor);
                    compared_1hop_list.push_back(neighbor);
                    // top_compared_list.emplace(neighbor, dist);
                    num_cmps++;
                }
            }


            // all 2-hops
            if (num_viable_1hop == 0) for (auto i=0; i<neighbors.size(); ++i) {
                auto& neighbor = neighbors[i];
                std::vector<IdxType> &twohop = _graph->neighbors[neighbor];
                for (auto j = 0; j < twohop.size(); j++) {
                    auto &nj = twohop[j];
                    auto &nj_label_set = _base_storage->get_label_set(nj);
                    bool nj_good = std::includes(nj_label_set.begin(), nj_label_set.end(), query_label.begin(), query_label.end());

                    if (nj_good && !visited_set.check(nj)) {
                        visited_set.set(nj);
                        float dist = _distance_handler->compute(query, _base_storage->get_vector(nj), dim);
                        if (search_queue.can_insert(dist)) {
                            n_insert_2hop++;
                            search_queue.insert(nj, dist);
                            // this edge helps! Put it in a separate data structure.
                        } else {
                            // this neighbor is no good: it's too far to be accepted into the search queue.
                        }
                        compared_list.push_back(nj);
                        compared_2hop_list.push_back(nj);
                        // top_compared_list.emplace(nj, dist);
                        num_cmps++;
                    }
                }
            }
        }
        // std::cout << expanded_list.size() << ", " << num_cmps << std::endl;
        /** expanded_list is very small!!  Far smaller than num_cmps.
         * expanded_list.size()  \approx  ef_search + 3
         * also, expanded_list is guaranteed to contain every final result (in search_queue) although may not be in order.
         */

        std::cout << "exp=" << expanded_list.size()
         << ", " << "c1hop=" << compared_1hop_list.size()
         << ", " << "c2hop=" << compared_2hop_list.size()
         << ", " << "ins1hop=" << n_insert_1hop
         << ", " << "ins2hop=" << n_insert_2hop
         << ", " << "cmps=" << num_cmps
        //  << ", " << "aug=" << augment_list.size()
         << std::endl;

        {
            int skip = 0;
            // NOTE: easy parallelization foreach "u". However, Don't do it too soon
            for (int u : expanded_list) {
                auto &adj_u = _graph->neighbors[u];
                std::vector<Candidate> cand_u;
                std::set<IdxType> cands;
                auto vec_u = _base_storage->get_vector(u);
                for (int v : adj_u) {
                    cand_u.emplace_back(v, _distance_handler->compute(vec_u, _base_storage->get_vector(v), dim));
                    cands.insert(v);
                }

                for (int v : expanded_list) {
                    if (v == u) { continue; }
                    if (cands.count(v) == 0) {
                        cand_u.emplace_back(v, _distance_handler->compute(vec_u, _base_storage->get_vector(v), dim));
                        cands.insert(v);
                    }
                }

                // use Vamana pruning: cap at maxdeg edges
                std::vector<IdxType> final_adj;
                vamana_prune_neighbors(u, cand_u, final_adj);
                _graph->neighbors[u] = std::move(final_adj);
            }
        }
        return num_cmps;
    }


    // this function records all "virtual" edges that are only obtained through 2-hop
    IdxType Hnsw::iterate_to_fixed_point__infilter_profiling(
        const char *query,
        const std::vector<LabelType> &query_label,
        std::shared_ptr<SearchCache> search_cache,
        std::map<std::pair<IdxType, IdxType>, int> &virt_edges) {

        auto dim = _base_storage->get_dim();
        auto& search_queue = search_cache->search_queue;
        auto& visited_set = search_cache->visited_set;
        search_queue.clear();
        visited_set.clear();

        // for adaptively updating the graph
        std::vector<IdxType> expanded_list;     // all nodes put into search queue AND expanded
        int n_insert_1hop = 0, n_insert_2hop = 0;

        float entry_dist = _distance_handler->compute(query, _base_storage->get_vector(_entry_point), dim);
        search_queue.insert(_entry_point, entry_dist);
        IdxType num_cmps = 1;
        visited_set.set(_entry_point);

        while (search_queue.has_unexpanded_node()) {
            const Candidate& cur = search_queue.get_closest_unexpanded();
            expanded_list.push_back(cur.id);

            std::vector<IdxType> &neighbors = _graph->neighbors[cur.id];
            auto &cur_label_set = _base_storage->get_label_set(cur.id);

            // all 1-hops
            int num_viable_1hop = 0;

            for (auto i=0; i<neighbors.size(); ++i) {
                auto& neighbor = neighbors[i];
                auto &nbr_label_set = _base_storage->get_label_set(neighbor);
                bool nbr_good = std::includes(nbr_label_set.begin(), nbr_label_set.end(), query_label.begin(), query_label.end());

                if (nbr_good && !visited_set.check(neighbor)) {
                    visited_set.set(neighbor);
                    float dist = _distance_handler->compute(query, _base_storage->get_vector(neighbor), dim);
                    if (search_queue.can_insert(dist)) {
                        num_viable_1hop++; n_insert_1hop++;
                        search_queue.insert(neighbor, dist);
                    }
                    num_cmps++;
                }
            }

            // all 2-hops
            if (num_viable_1hop == 0) for (auto i=0; i<neighbors.size(); ++i) {
                auto& neighbor = neighbors[i];
                std::vector<IdxType> &twohop = _graph->neighbors[neighbor];
                for (auto j = 0; j < twohop.size(); j++) {
                    auto &nj = twohop[j];
                    auto &nj_label_set = _base_storage->get_label_set(nj);
                    bool nj_good = std::includes(nj_label_set.begin(), nj_label_set.end(), query_label.begin(), query_label.end());

                    if (nj_good && !visited_set.check(nj)) {
                        visited_set.set(nj);
                        float dist = _distance_handler->compute(query, _base_storage->get_vector(nj), dim);
                        if (search_queue.can_insert(dist)) {
                            n_insert_2hop++;
                            auto virtual_edge = std::make_pair(cur.id, nj); // cur.id -> neighbor -> nj
                            virt_edges[virtual_edge]++;
                            search_queue.insert(nj, dist);
                        }
                        num_cmps++;
                    }
                }
            }
        }
        std::cout << "exp=" << expanded_list.size()
         << ", " << "ins1hop=" << n_insert_1hop
         << ", " << "ins2hop=" << n_insert_2hop
         << ", " << "cmps=" << num_cmps
         << std::endl;

        return num_cmps;
    }

    void Hnsw::statistics() {
        size_t num_points = _base_storage->get_num_points();
        std::cout << "Number of points: " << num_points << std::endl;

        size_t num_edges = 0;
        IdxType min_degree = std::numeric_limits<IdxType>::max(), max_degree = 0;
        for (auto id=0; id<num_points; ++id) {
            num_edges += _graph->neighbors[id].size();
            min_degree = std::min(min_degree, (IdxType)_graph->neighbors[id].size());
            max_degree = std::max(max_degree, (IdxType)_graph->neighbors[id].size());
        }
        std::cout << "Number of edges: " << num_edges << std::endl;
        std::cout << "Min degree: " << min_degree << std::endl;
        std::cout << "Max degree: " << max_degree << std::endl;

        float avg_degree = (float)num_edges / num_points;
        std::cout << "Average degree: " << avg_degree << std::endl;
    }



    void Hnsw::save(const std::string& index_path_prefix) {
        fs::create_directories(index_path_prefix);
        std::cout << "Saving index to " << index_path_prefix << " ..." << std::endl;

        // save meta data
        std::map<std::string, std::string> meta_data;
        // meta_data["max_degree"] = std::to_string(_max_degree);
        // meta_data["Lbuild"] = std::to_string(_Lbuild);
        // meta_data["alpha"] = std::to_string(_alpha);
        // meta_data["max_candidate_size"] = std::to_string(_max_candidate_size);
        meta_data["num_threads"] = std::to_string(_num_threads);
        meta_data["M"] = std::to_string(_M);
        meta_data["ef_construction"] = std::to_string(_ef_construction);
        meta_data["build_num_threads"] = std::to_string(_num_threads);
        meta_data["entry_point"] = std::to_string(_entry_point);
        std::string meta_filename = (fs::path(index_path_prefix) / "meta").string();
        write_kv_file(meta_filename, meta_data);

        // save graph data
        std::string graph_filename = (fs::path(index_path_prefix) / "graph").string();
        _graph->save(graph_filename);

        // print
        std::cout << "- Index saved." << std::endl;
    }

    

    void Hnsw::load(const std::string& index_path_prefix, std::shared_ptr<Graph> graph) {
        if (_verbose) {
            std::cout << "Loading index from " << index_path_prefix << " ..." << std::endl;
        }
            
        // load meta data
        std::string meta_filename = (fs::path(index_path_prefix) / "meta").string();
        auto meta_data = parse_kv_file(meta_filename);
        // std::cout << "Trying stoi from meta_data[\"entry_point\"], which is " << meta_data["entry_point"] << std::endl;
        _entry_point = std::stoul(meta_data["entry_point"]);    // allow "-1"
        // std::cout << "Trying stoi from meta_data[\"M\"], which is " << meta_data["M"] << std::endl;
        _M = std::stoi(meta_data["M"]);
        // std::cout << "Trying stoi from meta_data[\"ef_construction\"], which is " << meta_data["ef_construction"] << std::endl;
        _ef_construction = std::stoi(meta_data["ef_construction"]);

        // load graph data
        std::string graph_filename = (fs::path(index_path_prefix) / "graph").string();
        _graph = graph;
        _graph->load(graph_filename);

        if (_verbose) {
            std::cout << "- Index loaded." << std::endl;
        }
    }



    void Hnsw::search(std::shared_ptr<IStorage> base_storage, std::shared_ptr<IStorage> query_storage, 
                        std::shared_ptr<DistanceHandler> distance_handler, IdxType K, IdxType ef_search, 
                        uint32_t num_threads, std::pair<IdxType, float>* results, std::vector<IdxType>& num_cmps) {
        auto num_points = base_storage->get_num_points();
        auto num_queries = query_storage->get_num_points();
        _base_storage = base_storage;
        _query_storage = query_storage;
        _distance_handler = distance_handler;

        // preparation
        if (K > ef_search) {
            std::cerr << "Error: K should be less than or equal to Lsearch" << std::endl;
            exit(-1);
        }
        SearchCacheList search_cache_list(num_threads, num_points, ef_search);

        // run queries
        omp_set_num_threads(num_threads);
        #pragma omp parallel for schedule(dynamic, 1)
        for (auto id = 0; id < num_queries; ++id) {
            auto search_cache = search_cache_list.get_free_cache(); 
            const char* query = _query_storage->get_vector(id);
            num_cmps[id] = iterate_to_fixed_point(query, search_cache);

            // write results then clean
            for (auto k=0; k<K; ++k) {
                if (k < search_cache->search_queue.size()) {
                    results[id*K+k].first = search_cache->search_queue[k].id;
                    results[id*K+k].second = search_cache->search_queue[k].distance;
                } else
                    results[id*K+k].first = -1;
            }
            search_cache_list.release_cache(search_cache);
        }
    }


    // search, but do in-filtering with label containment.
    void Hnsw::search_containment_warmup(std::shared_ptr<IStorage> base_storage, std::shared_ptr<IStorage> query_storage, 
                        std::shared_ptr<DistanceHandler> distance_handler, IdxType K, IdxType ef_search, 
                        uint32_t num_threads, std::pair<IdxType, float>* results, std::vector<IdxType>& num_cmps) {
        // warmup length is gotten automatically from query_storage length
        auto num_points = base_storage->get_num_points();
        auto num_queries = query_storage->get_num_points();
        _base_storage = base_storage;
        _query_storage = query_storage; // use "_query_storage->get_label_set(id)" to get the filter.
        _distance_handler = distance_handler;

        // preparation
        if (K > ef_search) {
            std::cerr << "Error: K should be less than or equal to Lsearch" << std::endl;
            exit(-1);
        }
        SearchCacheList search_cache_list(num_threads, num_points, ef_search);

        // run queries
        // each query will serve to update the graph so that later queries run better.
        // that's why this must run sequentially.
        for (auto id = 0; id < num_queries; ++id) {
            auto search_cache = search_cache_list.get_free_cache(); 
            const char* query = _query_storage->get_vector(id);
            auto &query_label  = _query_storage->get_label_set(id);

            /**
             * My goal here is: after the warm-up phase with 2hop expansion,
             * the regular query phase uses only 1hop neighbors (probably with a larger ef_Search),
             * and favorably reach better cmps economics than 2hop expansion.
             * (recall > 85% at cmps < 3000)
             */
            {   // "warm up"
                num_cmps[id] = iterate_to_fixed_point__infilter_adaptive(query, query_label, search_cache);
            }

            // write results then clean
            for (auto k=0; k<K; ++k) {
                if (k < search_cache->search_queue.size()) {
                    results[id*K+k].first = search_cache->search_queue[k].id;
                    results[id*K+k].second = search_cache->search_queue[k].distance;
                } else
                    results[id*K+k].first = -1;
            }
            search_cache_list.release_cache(search_cache);
        }
    }


    std::map<std::pair<IdxType, IdxType>, int>
    Hnsw::search_containment_profiling(std::shared_ptr<IStorage> base_storage, std::shared_ptr<IStorage> query_storage, 
        std::shared_ptr<DistanceHandler> distance_handler, IdxType K, IdxType ef_search, 
        uint32_t num_threads, std::pair<IdxType, float>* results, std::vector<IdxType>& num_cmps) {

        // warmup length is gotten automatically from query_storage length
        auto num_points = base_storage->get_num_points();
        auto num_queries = query_storage->get_num_points();
        _base_storage = base_storage;
        _query_storage = query_storage; // use "_query_storage->get_label_set(id)" to get the filter.
        _distance_handler = distance_handler;

        // preparation
        if (K > ef_search) {
            std::cerr << "Error: K should be less than or equal to Lsearch" << std::endl;
            exit(-1);
        }
        SearchCacheList search_cache_list(num_threads, num_points, ef_search);

        std::map<std::pair<IdxType, IdxType>, int> virt_edges;

        omp_set_num_threads(num_threads);
        #pragma omp parallel for schedule(dynamic, 1)
        for (auto id = 0; id < num_queries; ++id) {
            auto search_cache = search_cache_list.get_free_cache(); 
            const char* query = _query_storage->get_vector(id);
            auto &query_label  = _query_storage->get_label_set(id);

            {
                // profiles virtual 2-hop edges into `virt_edges`
                std::map<std::pair<IdxType, IdxType>, int> my_virt_edges;
                num_cmps[id] = iterate_to_fixed_point__infilter_profiling(query, query_label, search_cache, my_virt_edges);

                // merge maps
                #pragma omp critical
                for (auto item : my_virt_edges) {
                    virt_edges[item.first] += item.second;
                }
            }

            // write results then clean
            for (auto k=0; k<K; ++k) {
                if (k < search_cache->search_queue.size()) {
                    results[id*K+k].first = search_cache->search_queue[k].id;
                    results[id*K+k].second = search_cache->search_queue[k].distance;
                } else
                    results[id*K+k].first = -1;
            }
            search_cache_list.release_cache(search_cache);
        }

        return virt_edges;
    }

    void Hnsw::select_and_insert(
            std::map<std::pair<ANNS::IdxType, ANNS::IdxType>, int> ve, int top_k,
            InsertType type) {
        std::cout << "Profiling returns a list of " << ve.size() << " edges." << std::endl;
        if (ve.size() == 0) return;

        struct edge_cand {
            IdxType start, end;
            int hotness;
            bool operator < (const edge_cand &other) { return hotness > other.hotness; }
        };
        std::vector<edge_cand> edges_vec;
        for (auto e : ve) {
            edges_vec.push_back(edge_cand({e.first.first, e.first.second, e.second}));
        }
        std::sort(edges_vec.begin(), edges_vec.end());

        if (top_k == -1)
            top_k = edges_vec.size() + 1;

        std::cout << "Maximum count = " << edges_vec[0].hotness << "." << std::endl;
        if (top_k < edges_vec.size())
        std::cout << "Cut-off count = " << edges_vec[top_k].hotness << "." << std::endl;
        std::cout << "Minumum count = " << edges_vec.rbegin()->hotness << "." << std::endl;

        {
            // dump the list into a file for plotting
            std::ofstream f("hotness.txt");
            for (auto c : edges_vec) {
                f << c.hotness << std::endl;
            }
        }

        int inserted = 0;
        int pruned = 0;

        if (top_k > 0)
        if (type == NO_PRUNE) {
            for (auto e : edges_vec) {
                auto &neighbors = _graph->neighbors[e.start];
                neighbors.push_back(e.end);
                ++inserted;
                if (inserted >= top_k) break;
            }
        } else if (type == VAMANA_PRUNE) {
            int dim = _base_storage->get_dim();

            // first write all edges into the "cands" vector
            std::map<IdxType, std::vector<Candidate>> cands;
            for (auto e : edges_vec) {
                auto vec_u = _base_storage->get_vector(e.start);

                if (cands.count(e.start) == 0) {
                    // first encounter: put all edges from base graph into candidate list
                    auto &adj_u = _graph->neighbors[e.start];
                    for (int v : adj_u) {
                        auto vec_v = _base_storage->get_vector(v);
                        cands[e.start].emplace_back(v, _distance_handler->compute(vec_u, vec_v, dim));
                    }
                }

                cands[e.start].emplace_back(e.end, _distance_handler->compute(vec_u, _base_storage->get_vector(e.end), dim));
                ++inserted;
                if (inserted >= top_k) break;
            }

            for (auto u : cands) {
                auto size1 = u.second.size();   // includes new edges and original edge list
                // int keepsize = _graph->neighbors[u.first].size();
                int keepsize = std::max<int>(_graph->neighbors[u.first].size(), _M);
                vamana_prune_neighbors(u.first, u.second, _graph->neighbors[u.first], keepsize);
                pruned += (size1 - _graph->neighbors[u.first].size());
            }
        } else if (type == VAMANA_PRUNE_KEEP_ADDED) {
            // new edges go directly into g->neighbors, while original
            // neighbors go into cand_u and get pruned, then appended
            // into g->neighbors if maximum degree not reached
            int dim = _base_storage->get_dim();

            std::map<IdxType, std::vector<IdxType>> cands_inserted;
            for (auto e : edges_vec) {
                cands_inserted[e.start].push_back(e.end);
                // force insert this edge, regardless of deg constraint
                ++inserted;
                if (inserted >= top_k) break;
            }

            for (auto u : cands_inserted) { // u: pair(IdxType, vector<IdxType>)
                auto n_ins = u.second.size();
                std::vector<Candidate> orig_edges;
                auto &adj_u = _graph->neighbors[u.first];
                // int keepsize = adj_u.size();
                int keepsize = std::max<int>(adj_u.size(), _M);
                if (n_ins < keepsize) {
                    {
                        // put all edges from base graph into candidate list
                        auto vec_u = _base_storage->get_vector(u.first);
                        for (int v : adj_u) {
                            auto vec_v = _base_storage->get_vector(v);
                            orig_edges.emplace_back(v, _distance_handler->compute(vec_u, vec_v, dim));
                        }
                    }
                    std::vector<IdxType> pruned_orig;
                    {
                        auto size1 = orig_edges.size();
                        vamana_prune_neighbors(u.first, orig_edges, pruned_orig, keepsize - n_ins);
                        pruned += (size1 - pruned_orig.size());
                    }
                    _graph->neighbors[u.first].resize(n_ins + pruned_orig.size());
                    std::copy(u.second.begin(), u.second.end(), _graph->neighbors[u.first].begin());
                    std::copy(pruned_orig.begin(), pruned_orig.end(), _graph->neighbors[u.first].begin() + n_ins);
                } else {
                    // we would discard all original edges altogether!
                    pruned += _graph->neighbors[u.first].size();
                    _graph->neighbors[u.first].resize(n_ins);
                    std::copy(u.second.begin(), u.second.end(), _graph->neighbors[u.first].begin());
                }
            }
        } else {
            throw type;
        }

        std::cout << "Inserted " << inserted << " edges, pruned " << pruned << " edges." << std::endl;
    }


    void Hnsw::search_containment(std::shared_ptr<IStorage> base_storage, std::shared_ptr<IStorage> query_storage, 
                        std::shared_ptr<DistanceHandler> distance_handler, IdxType K, IdxType ef_search,
                        bool enable_2hop, 
                        uint32_t num_threads, std::pair<IdxType, float>* results, std::vector<IdxType>& num_cmps) {
        auto num_points = base_storage->get_num_points();
        auto num_queries = query_storage->get_num_points();
        _base_storage = base_storage;
        _query_storage = query_storage; // use "_query_storage->get_label_set(id)" to get the filter.
        _distance_handler = distance_handler;

        // preparation
        if (K > ef_search) {
            std::cerr << "Error: K should be less than or equal to Lsearch" << std::endl;
            exit(-1);
        }
        SearchCacheList search_cache_list(num_threads, num_points, ef_search);

        // run queries
        omp_set_num_threads(num_threads);
        #pragma omp parallel for schedule(dynamic, 1)
        for (auto id = 0; id < num_queries; ++id) {
            auto search_cache = search_cache_list.get_free_cache(); 
            const char* query = _query_storage->get_vector(id);
            auto &query_label = _query_storage->get_label_set(id);
            num_cmps[id] = iterate_to_fixed_point__infilter(query, query_label, search_cache, enable_2hop);

            // write results then clean
            for (auto k=0; k<K; ++k) {
                if (k < search_cache->search_queue.size()) {
                    results[id*K+k].first = search_cache->search_queue[k].id;
                    results[id*K+k].second = search_cache->search_queue[k].distance;
                } else
                    results[id*K+k].first = -1;
            }
            search_cache_list.release_cache(search_cache);
        }
    }
}
