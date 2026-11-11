#include "online_update_baseline_common.h"
#include "mixed_calibration.h"
#include "mixed_workload.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <thread>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <unordered_set>

#include <omp.h>

#include "baselines/search.h"
#include "irangegraph/attribute_tree.h"
#include "tablestorage/tablestorage_utils.h"
#include "utils.h"
#include "artifact_paths.h"

namespace ANNS {

using namespace std;

namespace {

struct SearchRunStats {
    vector<pair<IdxType, float>> results;
    int n_cmps = 0;
};

char *get_live_vec_or_null(TableStorage &table_storage, IdxType id) {
    void *row = table_storage.find(id);
    if (row == nullptr) return nullptr;
    return table_storage.get_vector(row);
}

const float *get_live_attr_or_null(TableStorage &table_storage, IdxType id) {
    void *row = table_storage.find(id);
    if (row == nullptr) return nullptr;
    return table_storage.get_attr(row);
}

void init_table_storage_schema(shared_ptr<IStorage> base_storage,
                               int na,
                               TableStorage &table_storage) {
    table_storage.nb = base_storage->get_num_points();
    table_storage.na = na;
    table_storage.n_dim = base_storage->get_dim();
    table_storage.table_schema.resize(na, TableStorage::NUMERICAL);
    table_storage.dim_byte_count = 4;
    table_storage.vector_bytes = (size_t)table_storage.dim_byte_count * table_storage.n_dim;
    table_storage.attr_offset = TableStorage::roundUp(table_storage.vector_bytes, 4);
    table_storage.aligned_record_size =
        TableStorage::roundUp(table_storage.attr_offset + na * sizeof(float), 32);
}

vector<IdxType> live_ids_from_table(TableStorage &table_storage) {
    auto keys = table_storage.keys();
    return vector<IdxType>(keys.begin(), keys.end());
}

void search_flat_oor(
        ConcurrentEdgeMap &edgemap,
        const vector<IdxType> &entry_points,
        const char *query,
        function<char*(IdxType)> get_vec,
        shared_ptr<DistanceHandler> distance_handler,
        int dim,
        function<bool(IdxType)> predicate,
        SearchCacheList &cache_list,
        int K,
        SearchRunStats &stats) {
    stats.results.clear();
    stats.n_cmps = 0;

    auto cache = cache_list.get_free_cache();
    auto &sq = cache->search_queue;
    auto &vs = cache->visited_set;
    sq.clear();
    vs.clear();

    priority_queue<pair<float, IdxType>> top_candidates;
    auto record_top = [&](IdxType pid, float dist) {
        top_candidates.emplace(dist, pid);
        while ((int)top_candidates.size() > K) top_candidates.pop();
    };

    int seeded = 0;
    ConcurrentEdgeMap::InPlaceReadGuard rguard;
    for (IdxType entry : entry_points) {
        if (seeded >= 5) break;
        if (vs.check(entry)) continue;
        if (!edgemap.try_read_lock(entry, rguard)) continue;
        rguard.release();
        char *entry_vec = get_vec(entry);
        if (entry_vec == nullptr) continue;
        vs.set(entry);
        float dist = distance_handler->compute(query, entry_vec, dim);
        ++stats.n_cmps;
        sq.insert(entry, dist, -1);
        if (predicate(entry)) record_top(entry, dist);
        ++seeded;
    }

    while (sq.has_unexpanded_node()) {
        const Candidate cur = sq.get_closest_unexpanded();
        vector<IdxType> edges;
        if (!edgemap.try_read_lock(cur.id, rguard)) continue;
        edges = *rguard.edges;
        rguard.release();

        for (IdxType nbr : edges) {
            if (vs.check(nbr)) continue;
            char *nbr_vec = get_vec(nbr);
            if (nbr_vec == nullptr) continue;
            vs.set(nbr);
            float dist = distance_handler->compute(query, nbr_vec, dim);
            ++stats.n_cmps;
            sq.insert(nbr, dist, cur.id);
            if (predicate(nbr)) record_top(nbr, dist);
        }
    }

    while (!top_candidates.empty()) {
        auto [dist, id] = top_candidates.top();
        top_candidates.pop();
        stats.results.emplace_back(id, dist);
    }
    sort(stats.results.begin(), stats.results.end(),
         [](auto &a, auto &b) { return a.second < b.second; });

    cache_list.release_cache(cache);
}

void search_hnsw_oor(
        ConcurrentEdgeMap &edgemap,
        shared_ptr<HNSWUpperLayer> upper_layer,
        const vector<IdxType> &fallback_entry_points,
        const char *query,
        function<char*(IdxType)> get_vec,
        shared_ptr<DistanceHandler> distance_handler,
        int dim,
        function<bool(IdxType)> predicate,
        SearchCacheList &cache_list,
        int K,
        SearchRunStats &stats) {
    if (!upper_layer) {
        search_flat_oor(edgemap, fallback_entry_points, query, get_vec,
                        distance_handler, dim, predicate, cache_list, K, stats);
        return;
    }

    stats.results.clear();
    stats.n_cmps = 0;

    auto cache = cache_list.get_free_cache();
    auto &sq = cache->search_queue;
    auto &vs = cache->visited_set;
    sq.clear();
    vs.clear();

    priority_queue<pair<float, IdxType>> top_candidates;
    auto record_top = [&](IdxType pid, float dist) {
        top_candidates.emplace(dist, pid);
        while ((int)top_candidates.size() > K) top_candidates.pop();
    };

    auto compdist = [&](IdxType pid) {
        char *vec = get_vec(pid);
        if (vec == nullptr) return numeric_limits<float>::infinity();
        ++stats.n_cmps;
        return distance_handler->compute(query, vec, dim);
    };

    auto node_exists = [&](IdxType id) {
        ConcurrentEdgeMap::InPlaceReadGuard guard;
        bool ok = edgemap.try_read_lock(id, guard);
        if (ok) guard.release();
        return ok;
    };

    IdxType curr = upper_layer->enterpoint_node_;
    if (!node_exists(curr)) {
        curr = static_cast<IdxType>(-1);
        for (IdxType entry : fallback_entry_points) {
            if (node_exists(entry)) {
                curr = entry;
                break;
            }
        }
        if (curr == static_cast<IdxType>(-1)) {
            cache_list.release_cache(cache);
            return;
        }
    }

    float curdist = compdist(curr);
    for (int level = upper_layer->maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            auto it = upper_layer->edgelist.find(curr);
            if (it == upper_layer->edgelist.end()) break;
            auto &all_layers = it->second;
            IdxType offset = upper_layer->n_links_per_element_ * (level - 1);
            if (offset >= all_layers.size()) break;
            int size = all_layers[offset];
            IdxType *neighbors = const_cast<IdxType *>(&all_layers[offset + 1]);
            for (int i = 0; i < size; i++) {
                IdxType cand = neighbors[i];
                if (!node_exists(cand)) continue;
                float d = compdist(cand);
                if (d < curdist) {
                    curdist = d;
                    curr = cand;
                    changed = true;
                }
            }
        }
    }

    vs.set(curr);
    sq.insert(curr, curdist, -1);
    if (predicate(curr)) record_top(curr, curdist);

    ConcurrentEdgeMap::InPlaceReadGuard rguard;
    while (sq.has_unexpanded_node()) {
        const Candidate cur = sq.get_closest_unexpanded();
        vector<IdxType> edges;
        if (!edgemap.try_read_lock(cur.id, rguard)) continue;
        edges = *rguard.edges;
        rguard.release();

        for (IdxType nbr : edges) {
            if (vs.check(nbr)) continue;
            vs.set(nbr);
            float dist = compdist(nbr);
            sq.insert(nbr, dist, cur.id);
            if (predicate(nbr)) record_top(nbr, dist);
        }
    }

    while (!top_candidates.empty()) {
        auto [dist, id] = top_candidates.top();
        top_candidates.pop();
        stats.results.emplace_back(id, dist);
    }
    sort(stats.results.begin(), stats.results.end(),
         [](auto &a, auto &b) { return a.second < b.second; });

    cache_list.release_cache(cache);
}

void search_navix(
        ConcurrentEdgeMap &edgemap,
        const vector<IdxType> &entry_points,
        const char *query,
        function<char*(IdxType)> get_vec,
        shared_ptr<DistanceHandler> distance_handler,
        int dim,
        function<bool(IdxType)> predicate,
        SearchCacheList &cache_list,
        int K,
        int M,
        int n_entry,
        SearchRunStats &stats) {
    stats.results.clear();
    stats.n_cmps = 0;

    auto cache = cache_list.get_free_cache();
    auto &sq = cache->search_queue;
    auto &vs = cache->visited_set;
    sq.clear();
    vs.clear();

    priority_queue<pair<float, IdxType>> top_candidates;
    auto record_top = [&](IdxType pid, float dist) {
        top_candidates.emplace(dist, pid);
        while ((int)top_candidates.size() > K) top_candidates.pop();
    };

    auto compdist = [&](IdxType pid) {
        char *vec = get_vec(pid);
        if (vec == nullptr) return numeric_limits<float>::infinity();
        ++stats.n_cmps;
        return distance_handler->compute(query, vec, dim);
    };

    int seeded = 0;
    ConcurrentEdgeMap::InPlaceReadGuard rguard;
    for (IdxType entry : entry_points) {
        if (seeded >= n_entry) break;
        if (vs.check(entry)) continue;
        if (!edgemap.try_read_lock(entry, rguard)) continue;
        rguard.release();
        char *entry_vec = get_vec(entry);
        if (entry_vec == nullptr) continue;
        vs.set(entry);
        float dist = distance_handler->compute(query, entry_vec, dim);
        ++stats.n_cmps;
        sq.insert(entry, dist, -1);
        if (predicate(entry)) record_top(entry, dist);
        ++seeded;
    }

    while (sq.has_unexpanded_node()) {
        const Candidate cur = sq.get_closest_unexpanded();

        vector<IdxType> neighbors;
        if (!edgemap.try_read_lock(cur.id, rguard)) continue;
        neighbors = *rguard.edges;
        rguard.release();
        if (neighbors.empty()) continue;

        int visited_passing = 0;
        int pass_count = 0;
        vector<int> pass_mask(neighbors.size(), 0);
        for (size_t i = 0; i < neighbors.size(); i++) {
            IdxType nbr = neighbors[i];
            bool pass = predicate(nbr);
            if (pass) visited_passing++;
            if (vs.check(nbr)) continue;
            pass_mask[i] = pass ? 1 : 0;
            if (pass_mask[i]) pass_count++;
        }

        double sel_local = (double)pass_count / neighbors.size();
        double esv = sel_local * (M + 1) * M;
        double leniency = 3.0;

        if (sel_local >= 0.4) {
            for (size_t i = 0; i < neighbors.size(); i++) {
                if (!pass_mask[i]) continue;
                IdxType nbr = neighbors[i];
                if (vs.check(nbr)) continue;
                vs.set(nbr);
                float dist = compdist(nbr);
                record_top(nbr, dist);
                sq.insert(nbr, dist);
            }
        } else if (esv < M * leniency) {
            int filter_nbrs_to_find = neighbors.size();
            for (size_t i = 0; i < neighbors.size(); i++) {
                IdxType nbr = neighbors[i];
                if (!vs.check(nbr) && pass_mask[i]) {
                    vs.set(nbr);
                    float dist = compdist(nbr);
                    record_top(nbr, dist);
                    sq.insert(nbr, dist);
                }

                vector<IdxType> edges2;
                if (!edgemap.try_read_lock(nbr, rguard)) continue;
                edges2 = *rguard.edges;
                rguard.release();

                for (IdxType nbr2 : edges2) {
                    if (visited_passing >= filter_nbrs_to_find) break;
                    if (vs.check(nbr2)) continue;
                    if (!predicate(nbr2)) continue;
                    ++visited_passing;
                    vs.set(nbr2);
                    float dist = compdist(nbr2);
                    record_top(nbr2, dist);
                    sq.insert(nbr2, dist);
                }
                if (visited_passing >= filter_nbrs_to_find) break;
            }
        } else {
            priority_queue<pair<float, IdxType>> nbrs_to_explore;
            for (size_t i = 0; i < neighbors.size(); i++) {
                IdxType nbr = neighbors[i];
                if (vs.check(nbr)) continue;
                vs.set(nbr);
                float dist = compdist(nbr);
                nbrs_to_explore.emplace(-dist, nbr);
                if (pass_mask[i]) {
                    record_top(nbr, dist);
                    sq.insert(nbr, dist);
                }
            }

            int filter_nbrs_to_find = neighbors.size();
            while (!nbrs_to_explore.empty()) {
                auto [neg_dist, nbr] = nbrs_to_explore.top();
                nbrs_to_explore.pop();
                if (visited_passing >= filter_nbrs_to_find) break;

                vector<IdxType> edges2;
                if (!edgemap.try_read_lock(nbr, rguard)) continue;
                edges2 = *rguard.edges;
                rguard.release();

                for (IdxType nbr2 : edges2) {
                    bool pass = predicate(nbr2);
                    if (pass) visited_passing++;
                    if (vs.check(nbr2)) continue;
                    if (!pass) continue;
                    vs.set(nbr2);
                    float dist = compdist(nbr2);
                    record_top(nbr2, dist);
                    sq.insert(nbr2, dist);
                }
            }
        }
    }

    int ptr = top_candidates.size();
    stats.results.resize(ptr);
    --ptr;
    for (; ptr >= 0; --ptr) {
        auto [dist, id] = top_candidates.top();
        top_candidates.pop();
        stats.results[ptr] = {id, dist};
    }

    cache_list.release_cache(cache);
}

// =====================================================================
//  FAVOR on the online-update index
//  Graph: the same stock-HNSW bottom layer (ConcurrentEdgeMap, updated in place by
//  insert_point/delete_point exactly as for the HNSW baseline) plus the static upper
//  layer. Search: icde::favor from baselines/search.h reads the live edge map.
// =====================================================================

// Upstream FAVOR addPoint (include/favor.h): when the level-0 ef_construction pool is
// full, local_delta_d += 5 * (farthest - 10th nearest) / ((ef_c - 10) * max_elements).
// Returns the term before the division by max_elements (capacity).
double favor_insert_term(vector<float> &pool, int ef_con) {
    if (ef_con <= 10 || (int)pool.size() < ef_con) return 0.0;
    partial_sort(pool.begin(), pool.begin() + ef_con, pool.end());
    return 5.0 * (double(pool[ef_con - 1]) - double(pool[9])) / double(ef_con - 10);
}

// Query-local scratch; release each edge-list lock before computing distances.
// This benchmark's upper layer is immutable, while level zero is updated in place.
struct FavorLiveGraph {
    BaselineIndex &index;
    icde::Id entry = -1;
    int max_level = 0;
    mutable vector<icde::Id> scratch;
    FavorLiveGraph(BaselineIndex &idx, const vector<IdxType> &fallback) : index(idx) {
        if (!index.upper_layer) throw runtime_error("FAVOR needs the HNSW upper layer");
        const auto &up = *index.upper_layer;
        ConcurrentEdgeMap::InPlaceReadGuard guard;
        auto exists = [&](IdxType id) {
            bool found = index.edgemap.try_read_lock(id, guard);
            if (found) guard.release();
            return found;
        };
        entry = up.enterpoint_node_;
        max_level = up.maxlevel_;
        if (!exists(entry)) {
            entry = -1;
            for (auto id : fallback) if (exists(id)) { entry = id; break; }
            if (entry < 0) throw runtime_error("FAVOR: no live entry point");
            max_level = entry < (int)up.element_levels_.size() ? up.element_levels_[entry] : 0;
        }
        scratch.reserve(2 * up.n_links_per_element_);
    }
    size_t size() const { return index.nb; }
    icde::Span neighbors(icde::Id id, int level = 0) const {
        scratch.clear();
        if (level == 0) {
            ConcurrentEdgeMap::InPlaceReadGuard guard;
            if (index.edgemap.try_read_lock(id, guard)) {
                scratch.assign(guard.edges->begin(), guard.edges->end());
                guard.release();
            }
        } else {
            const auto &up = *index.upper_layer;
            auto it = up.edgelist.find(id);
            size_t start = size_t(level - 1) * up.n_links_per_element_;
            if (it != up.edgelist.end() && start < it->second.size()) {
                int count = it->second[start];
                scratch.assign(it->second.begin() + start + 1,
                               it->second.begin() + start + 1 + count);
            }
        }
        return {scratch.data(), scratch.data() + scratch.size()};
    }
    int split(icde::Id id, int level = 0) const { return neighbors(id, level).size(); }
};

// Same procedure as calibrate_favor in experiments/icde_search.cpp (10,000 shuffled
// samples, seed 20261102, unfiltered ef=200 searches excluding the sample itself),
// restricted to the ids that are live in the initial index.
double favor_calibrate_mean(BaselineIndex &index, const vector<IdxType> &entries, uint32_t num_threads) {
    const int breadth = 200;
    vector<IdxType> ids = index.initial_ids;
    mt19937 rng(20261102);
    shuffle(ids.begin(), ids.end(), rng);
    ids.resize(min<size_t>(10000, ids.size()));
    vector<unique_ptr<icde::Visited>> visits;
    for (unsigned t = 0; t < num_threads; ++t) visits.emplace_back(new icde::Visited(index.nb));
    vector<double> terms(ids.size(), 0.0);
    int full = 0;
    #pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads) reduction(+: full)
    for (size_t s = 0; s < ids.size(); ++s) {
        int sample = ids[s];
        const char *query = get_live_vec_or_null(index.table_storage, sample);
        icde::Access a;
        a.distance = [&](int id) {
            char *v = get_live_vec_or_null(index.table_storage, id);
            return v ? index.distance_handler->compute(query, v, index.dim) : numeric_limits<float>::infinity();
        };
        FavorLiveGraph graph(index, entries);
        auto entry = icde::descend(graph, a);
        icde::Heap queue, top;
        auto &seen = *visits[omp_get_thread_num()];
        seen.clear(); seen.set(entry.second);
        queue.push({-entry.first, entry.second});
        if (entry.second != sample) top.push(entry);
        while (!queue.empty()) {
            auto cur = queue.top(); queue.pop();
            if ((int)top.size() == breadth && -cur.first > top.top().first) break;
            for (int id : graph.neighbors(cur.second)) {
                if (seen.has(id)) continue;
                seen.set(id);
                float d = a.distance(id);
                if ((int)top.size() < breadth || d < top.top().first) {
                    queue.push({-d, id});
                    if (id != sample) { top.push({d, id}); if ((int)top.size() > breadth) top.pop(); }
                }
            }
        }
        if ((int)top.size() == breadth) {
            auto hits = icde::finish(top, breadth);
            terms[s] = 5.0 * (hits.back().first - hits[9].first) / (breadth - 10);
            ++full;
        }
    }
    double mean = accumulate(terms.begin(), terms.end(), 0.0) / terms.size();
    cout << setprecision(10) << "FAVOR calibration (initial graph): samples=" << terms.size()
         << " complete_pools=" << full << " efConstruction=" << breadth
         << " seed=20261102 mean=" << mean << endl;
    return mean;
}

// One FAVOR query: predicates through TableStorage (live rows), low-selectivity branch
// as a true B-tree leaf scan in batches of 4096 (same as experiments/icde_search.cpp).
vector<icde::Hit> favor_search_one(BaselineIndex &index, const vector<IdxType> &entries, const char *query,
                                   const function<bool(IdxType)> &predicate, const QueryType &qt,
                                   int K, int ef, float delta_d, icde::Visited &seen, int &n_cmps) {
    icde::Access a;
    a.distance = [&](int id) {
        char *v = get_live_vec_or_null(index.table_storage, id);
        if (v == nullptr) return numeric_limits<float>::infinity();
        ++n_cmps;
        return index.distance_handler->compute(query, v, index.dim);
    };
    a.predicate = [&](int id) { return predicate(id); };
    a.scan = [&](const auto &visit) {
        using Tree = decltype(index.table_storage.tree);
        vector<typename Tree::KeyValuePair> rows;
        rows.reserve(4096);
        uint32_t low = 0;
        for (;;) {
            index.table_storage.tree.scan(TableStorage::KeyWrapper(low), TableStorage::KeyWrapper(UINT32_MAX),
                                          true, true, 4096, rows);
            if (rows.empty()) break;
            for (const auto &row : rows) {
                auto *record = row.second.getValue();
                if (qt.check(index.table_storage.get_attr(record))) {
                    function<float()> dist = [&] {
                        ++n_cmps;
                        return index.distance_handler->compute(query, index.table_storage.get_vector(record), index.dim);
                    };
                    visit(row.first.getValue(), dist);
                }
            }
            auto last = rows.back().first.getValue();
            if (rows.size() < 4096 || last == UINT32_MAX) break;
            low = last + 1;
        }
    };
    FavorLiveGraph graph(index, entries);
    return icde::favor(graph, a, seen, K, ef, delta_d);
}

} // namespace

BaselineDatasetConfig get_baseline_dataset_config(const string &dataset) {
    string data_root = pf_path("data/");
    string attr_root = pf_path("workloads/mar30_new_mixed_attributes/step0_attributes/");
    string index_root = pf_path("data/index_files/apr1_online_update/");
    string tsv_root = pf_path("workloads/mar30_new_mixed_attributes/step2/");

    BaselineDatasetConfig cfg;
    cfg.dataset = dataset;
    if (dataset == "sift3") {
        cfg.base_bin_file = data_root + "sift/sift_base.bin";
        cfg.query_bin_file = data_root + "sift/sift_query.bin";
        cfg.attr_bin_file = attr_root + "SIFT-mixed_attrs.bin";
        cfg.query_tsv_file = tsv_root + "pred_sift_a012_mid.tsv";
        cfg.index_prefix_pathfinder = index_root + "sift3-pathfinder/";
        cfg.M = 32;
        cfg.query_tsv_files = {
            {"low",  tsv_root + "pred_sift_a012_low.tsv"},
            {"mid",  tsv_root + "pred_sift_a012_mid.tsv"},
            {"high", tsv_root + "pred_sift_a012_high.tsv"},
        };
    } else if (dataset == "gist3") {
        cfg.base_bin_file = data_root + "gist/gist_base.bin";
        cfg.query_bin_file = data_root + "gist/gist_query.bin";
        cfg.attr_bin_file = attr_root + "GIST-mixed_attrs.bin";
        cfg.query_tsv_file = tsv_root + "pred_gist_a012_mid.tsv";
        cfg.index_prefix_pathfinder = index_root + "gist3-pathfinder/";
        cfg.M = 64;
        cfg.query_tsv_files = {
            {"low",  tsv_root + "pred_gist_a012_low.tsv"},
            {"mid",  tsv_root + "pred_gist_a012_mid.tsv"},
            {"high", tsv_root + "pred_gist_a012_high.tsv"},
        };
    } else if (dataset == "redcaps") {
        string home_redcaps = pf_path("datasets/redcaps/");
        string redcaps_tsv_root = pf_path("workloads/oct20_tsv_two_three_mixed/");
        cfg.base_bin_file = home_redcaps + "redcaps-embeddings.bin";
        cfg.query_bin_file = home_redcaps + "redcaps-query.bin";
        cfg.attr_bin_file = home_redcaps + "redcaps-attributes-subreddit-time-upvote.bin";
        cfg.query_tsv_file = redcaps_tsv_root + "redcaps_two_three_mid.tsv";
        cfg.index_prefix_pathfinder = index_root + "redcaps-pathfinder/";
        cfg.M = 64;
        cfg.query_tsv_files = {
            {"low",  redcaps_tsv_root + "redcaps_two_three_low.tsv"},
            {"mid",  redcaps_tsv_root + "redcaps_two_three_mid.tsv"},
            {"high", redcaps_tsv_root + "redcaps_two_three_high.tsv"},
        };
    } else {
        throw runtime_error("Unknown dataset: " + dataset);
    }
    return cfg;
}

BaselineArtifacts get_baseline_artifacts(const BaselineDatasetConfig &cfg,
                                         const string &method) {
    BaselineArtifacts out;
    string prefix = cfg.index_prefix_pathfinder;
    if (method == "vamana") {
        out.graph_path = prefix + "root_vamana.txt";
        out.mapping_path = prefix + "initial_ids_vamana.bin";
    } else if (method == "hnsw") {
        out.graph_path = prefix + "root.txt";
        out.mapping_path = prefix + "initial_ids_hnsw.bin";
        out.upper_path = prefix + "root_upper.txt";
    } else {
        throw runtime_error("Unknown baseline method: " + method);
    }
    return out;
}

shared_ptr<BaselineIndex> load_baseline_index(const BaselineDatasetConfig &cfg,
                                              const string &method) {
    auto out = make_shared<BaselineIndex>();
    out->base_storage = create_storage("float");
    out->base_storage->load_from_file(cfg.base_bin_file, "");
    out->query_storage = create_storage("float");
    out->query_storage->load_from_file(cfg.query_bin_file, "");
    out->nb = out->base_storage->get_num_points();
    out->nq = out->query_storage->get_num_points();
    out->dim = out->base_storage->get_dim();
    out->distance_handler = get_distance_handler("float", "L2");
    out->full_attr = read_binary_float32(cfg.attr_bin_file);
    out->na = out->full_attr.size() / out->nb;
    out->M = cfg.M;

    auto [query_types, _] = read_queries_from_tsv(cfg.query_tsv_file);
    out->query_types = std::move(query_types);

    auto artifacts = get_baseline_artifacts(cfg, method);
    out->initial_ids = load_id_mapping(artifacts.mapping_path);
    int n_initial = out->initial_ids.size();

    init_table_storage_schema(out->base_storage, out->na, out->table_storage);
    for (IdxType gid : out->initial_ids) {
        const float *attr_ptr = out->full_attr.data() + (size_t)gid * out->na;
        const float *vec_ptr = reinterpret_cast<const float *>(out->base_storage->get_vector(gid));
        if (!out->table_storage.insert(gid, vec_ptr, attr_ptr)) {
            throw runtime_error("Failed to insert initial row into TableStorage");
        }
    }

    auto graph = make_shared<Graph>(n_initial);
    graph->load(artifacts.graph_path);

    AttrTreeNode node(0, 100, 0, n_initial);
    node.g = graph;
    node.size = n_initial;
    for (int i = 0; i < n_initial; i++) {
        node.vertex_table[out->initial_ids[i]] = i;
    }
    node.translate_edge_destination();
    out->edgemap.reserve(out->nb);
    for (const auto &kv : node.vertex_table) {
        IdxType global_id = kv.first;
        IdxType internal_id = kv.second;
        out->edgemap.emplace(global_id, std::move(node.g->neighbors[internal_id]));
    }

    if (method == "hnsw") {
        out->upper_layer = make_shared<HNSWUpperLayer>();
        out->upper_layer->load(artifacts.upper_path);
        auto &upper = *out->upper_layer;
        upper.enterpoint_node_ = out->initial_ids.at(upper.enterpoint_node_);
        vector<int> old_levels = std::move(upper.element_levels_);
        upper.element_levels_.assign(out->nb, 0);
        for (int i = 0; i < n_initial; i++) {
            upper.element_levels_[out->initial_ids[i]] = old_levels[i];
        }
        unordered_map<IdxType, vector<IdxType>> new_edgelist;
        for (auto &[node_id, neighbors] : upper.edgelist) {
            auto &dst = new_edgelist[out->initial_ids.at(node_id)];
            dst.resize(neighbors.size());
            for (size_t j = 0; j < neighbors.size(); j++) {
                dst[j] = out->initial_ids.at(neighbors[j]);
            }
        }
        upper.edgelist = std::move(new_edgelist);
    }

    online_distance_handler = out->distance_handler;
    online_nb = out->nb;
    online_dim = out->dim;
    online_M = out->M;
    online_vamana_alpha = 1.2f;

    return out;
}

vector<IdxType> sample_entry_points(ConcurrentEdgeMap &edgemap, int n) {
    vector<IdxType> entries;
    for (auto it = edgemap.begin(); it != edgemap.end(); ++it) {
        auto kv = *it;
        entries.push_back(kv.first);
        if ((int)entries.size() >= n) break;
    }
    return entries;
}

void compute_filtered_ground_truth(
        TableStorage &table_storage,
        shared_ptr<IStorage> query_storage,
        shared_ptr<DistanceHandler> distance_handler,
        int dim,
        const vector<IdxType> &live_ids,
        function<bool(IdxType, int)> predicate,
        int nq, int K,
        vector<vector<pair<IdxType, float>>> &gt_out) {
    gt_out.resize(nq);
    #pragma omp parallel for schedule(dynamic, 4)
    for (int q = 0; q < nq; q++) {
        auto query = query_storage->get_vector(q);
        priority_queue<pair<float, IdxType>> heap;
        for (IdxType pid : live_ids) {
            if (!predicate(pid, q)) continue;
            char *vec = get_live_vec_or_null(table_storage, pid);
            if (vec == nullptr) continue;
            float d = distance_handler->compute(query, vec, dim);
            heap.emplace(d, pid);
            if ((int)heap.size() > K) heap.pop();
        }
        gt_out[q].resize(heap.size());
        int idx = heap.size() - 1;
        while (!heap.empty()) {
            gt_out[q][idx--] = {heap.top().second, heap.top().first};
            heap.pop();
        }
    }
}

UpdateStats run_batched_updates(BaselineIndex &index,
                                int n_batches,
                                int batch_size,
                                int ef_con,
                                uint32_t num_threads,
                                vector<IdxType> &insert_ids,
                                vector<IdxType> &delete_ids,
                                vector<IdxType> &entry_points,
                                FavorUpdateState *favor) {
    omp_set_num_threads(num_threads);
    SearchCacheList insert_cache_list(num_threads + 5, index.nb, ef_con);

    cout << "\n=== Phase 1: " << n_batches << " batches ===" << endl;
    cout << "  batch | ins_ms | del_ms | ins_QPS | del_QPS | ins_dc | del_dc" << endl;

    vector<double> batch_ins_ms(n_batches), batch_del_ms(n_batches);
    vector<long> batch_ins_dc(n_batches), batch_del_dc(n_batches);

    for (int b = 0; b < n_batches; b++) {
        int start = b * batch_size;
        int end = start + batch_size;

        long ins_dc_total = 0;
        auto t_ins_start = chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic, 4) reduction(+: ins_dc_total)
        for (int i = start; i < end; i++) {
            IdxType gid = insert_ids[i];
            char *vec = index.base_storage->get_vector(gid);
            const float *attr = index.full_attr.data() + (size_t)gid * index.na;
            if (!index.table_storage.insert(gid, reinterpret_cast<const float *>(vec), attr)) {
                throw runtime_error("Failed to insert row into TableStorage during update");
            }
            long before = online_dist_comps;
            if (favor) {
                // FAVOR: stock-HNSW insertion + upstream's delta_d growth (timed, as upstream).
                vector<float> pool;
                insert_point(index.edgemap, gid, vec, index.table_storage, insert_cache_list, entry_points, &pool);
                double term = favor_insert_term(pool, favor->ef_con);
                lock_guard<mutex> lk(favor->mu);
                favor->delta_d += term / favor->capacity;
                favor->terms_added += term > 0;
            } else {
                insert_point(index.edgemap, gid, vec, index.table_storage, insert_cache_list, entry_points);
            }
            ins_dc_total += online_dist_comps - before;
        }
        auto t_ins_end = chrono::high_resolution_clock::now();
        double ins_ms = chrono::duration_cast<chrono::microseconds>(t_ins_end - t_ins_start).count() / 1e3;

        long del_dc_total = 0;
        auto t_del_start = chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic, 4) reduction(+: del_dc_total)
        for (int i = start; i < end; i++) {
            IdxType gid = delete_ids[i];
            char *vec = get_live_vec_or_null(index.table_storage, gid);
            if (vec == nullptr) continue;
            long before = online_dist_comps;
            delete_point(index.edgemap, gid, vec, index.table_storage, insert_cache_list, entry_points);
            index.table_storage.remove(gid);
            del_dc_total += online_dist_comps - before;
        }
        auto t_del_end = chrono::high_resolution_clock::now();
        double del_ms = chrono::duration_cast<chrono::microseconds>(t_del_end - t_del_start).count() / 1e3;

        batch_ins_ms[b] = ins_ms;
        batch_del_ms[b] = del_ms;
        batch_ins_dc[b] = ins_dc_total;
        batch_del_dc[b] = del_dc_total;

        entry_points = sample_entry_points(index.edgemap, 100);

        if (b % 10 == 0 || b == n_batches - 1) {
            cout << "  " << setw(5) << (b + 1)
                 << " | " << setw(7) << fixed << setprecision(1) << ins_ms
                 << " | " << setw(7) << del_ms
                 << " | " << setw(7) << setprecision(0) << batch_size / (ins_ms / 1e3)
                 << " | " << setw(7) << batch_size / (del_ms / 1e3)
                 << " | " << setw(6) << setprecision(0) << (double)ins_dc_total / batch_size
                 << " | " << setw(6) << (double)del_dc_total / batch_size
                 << endl;
        }
    }

    UpdateStats stats;
    stats.avg_ins_ms = accumulate(batch_ins_ms.begin(), batch_ins_ms.end(), 0.0) / n_batches;
    stats.avg_del_ms = accumulate(batch_del_ms.begin(), batch_del_ms.end(), 0.0) / n_batches;
    stats.avg_ins_dc = (double)accumulate(batch_ins_dc.begin(), batch_ins_dc.end(), 0L) /
                       ((double)n_batches * batch_size);
    stats.avg_del_dc = (double)accumulate(batch_del_dc.begin(), batch_del_dc.end(), 0L) /
                       ((double)n_batches * batch_size);
    stats.avg_ins_qps = batch_size / (stats.avg_ins_ms / 1e3);
    stats.avg_del_qps = batch_size / (stats.avg_del_ms / 1e3);
    stats.avg_update_qps = (2.0 * batch_size) / ((stats.avg_ins_ms + stats.avg_del_ms) / 1e3);

    cout << "\n=== Update throughput (averaged over " << n_batches << " batches) ===" << endl;
    cout << "  Avg insert batch: " << fixed << setprecision(1) << stats.avg_ins_ms
         << " ms -> " << setprecision(0) << stats.avg_ins_qps << " inserts/s, "
         << setprecision(1) << stats.avg_ins_dc << " dist comps/insert" << endl;
    cout << "  Avg delete batch: " << setprecision(1) << stats.avg_del_ms
         << " ms -> " << setprecision(0) << stats.avg_del_qps << " deletes/s, "
         << setprecision(1) << stats.avg_del_dc << " dist comps/delete" << endl;

    return stats;
}

EvalStats evaluate_filtered_queries(BaselineIndex &index,
                                    BaselineSearchMode mode,
                                    const string &title,
                                    const vector<IdxType> &live_ids,
                                    int K,
                                    int ef_search,
                                    uint32_t num_threads,
                                    int navix_n_entry) {
    cout << "\n=== " << title << " ===" << endl;
    cout << "  Live points: " << live_ids.size() << " / " << index.nb << endl;

    int nq_eval = min(index.nq, (int)index.query_types.size());
    vector<bool> is_live(index.nb, false);
    for (IdxType id : live_ids) is_live[id] = true;

    auto predicate_2arg = [&](IdxType pid, int q) -> bool {
        const float *attr = get_live_attr_or_null(index.table_storage, pid);
        return attr != nullptr && index.query_types[q].check(attr);
    };
    auto predicate_factory = [&](int q) -> function<bool(IdxType)> {
        return [&, q](IdxType pid) -> bool {
            const float *attr = get_live_attr_or_null(index.table_storage, pid);
            return is_live[pid] && attr != nullptr && index.query_types[q].check(attr);
        };
    };

    vector<vector<pair<IdxType, float>>> gt;
    {
        int gt_threads = omp_get_num_procs();
        omp_set_num_threads(gt_threads);
        compute_filtered_ground_truth(index.table_storage, index.query_storage,
                                      index.distance_handler, index.dim, live_ids,
                                      predicate_2arg, nq_eval, K, gt);
        omp_set_num_threads(num_threads);
    }

    SearchCacheList search_cache_list(num_threads + 5, index.nb, ef_search);
    vector<IdxType> entry_points = sample_entry_points(index.edgemap, 100);
    auto get_vec = [&](IdxType id) -> char* {
        return get_live_vec_or_null(index.table_storage, id);
    };

    int total_found = 0;
    long total_dist_comps = 0;
    auto t0 = chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic, 1) reduction(+: total_found, total_dist_comps)
    for (int q = 0; q < nq_eval; q++) {
        auto query = index.query_storage->get_vector(q);
        auto predicate = predicate_factory(q);
        SearchRunStats stats;
        if (mode == BaselineSearchMode::FlatOOR) {
            search_flat_oor(index.edgemap, entry_points, query, get_vec,
                            index.distance_handler, index.dim, predicate,
                            search_cache_list, K, stats);
        } else if (mode == BaselineSearchMode::HNSWOOR) {
            search_hnsw_oor(index.edgemap, index.upper_layer, entry_points, query,
                            get_vec, index.distance_handler, index.dim, predicate,
                            search_cache_list, K, stats);
        } else {
            search_navix(index.edgemap, entry_points, query, get_vec,
                         index.distance_handler, index.dim, predicate,
                         search_cache_list, K, index.M, navix_n_entry, stats);
        }

        unordered_set<IdxType> gt_set;
        for (auto &[id, dist] : gt[q]) gt_set.insert(id);
        for (auto &[id, dist] : stats.results) {
            if (gt_set.count(id)) total_found++;
        }
        total_dist_comps += stats.n_cmps;
    }

    auto t1 = chrono::high_resolution_clock::now();
    EvalStats out;
    out.wall_ms = chrono::duration_cast<chrono::microseconds>(t1 - t0).count() / 1e3;
    out.recall = 100.0 * total_found / (nq_eval * K);
    out.qps = nq_eval / (out.wall_ms / 1e3);
    out.avg_dist_comps = (double)total_dist_comps / nq_eval;
    out.live_points = live_ids.size();

    cout << "  Recall@" << K << " = " << fixed << setprecision(2) << out.recall << "%" << endl;
    cout << "  QPS = " << setprecision(0) << out.qps << endl;
    cout << "  Avg dist comps/query = " << setprecision(1) << out.avg_dist_comps << endl;
    cout << "  Search wall time = " << setprecision(1) << out.wall_ms << " ms" << endl;

    return out;
}

vector<IdxType> make_insert_ids(const BaselineIndex &index,
                                int total_updates,
                                int seed) {
    vector<bool> in_initial(index.nb, false);
    for (IdxType id : index.initial_ids) in_initial[id] = true;
    vector<IdxType> insert_ids;
    insert_ids.reserve(index.nb - index.initial_ids.size());
    for (int i = 0; i < index.nb; i++) {
        if (!in_initial[i]) insert_ids.push_back(i);
    }
    mt19937 rng(seed);
    shuffle(insert_ids.begin(), insert_ids.end(), rng);
    if ((int)insert_ids.size() < total_updates) {
        throw runtime_error("Not enough insert candidates");
    }
    insert_ids.resize(total_updates);
    return insert_ids;
}

vector<IdxType> make_delete_ids(const BaselineIndex &index,
                                int total_updates,
                                int seed) {
    vector<IdxType> delete_ids = index.initial_ids;
    mt19937 rng(seed);
    shuffle(delete_ids.begin(), delete_ids.end(), rng);
    if ((int)delete_ids.size() < total_updates) {
        throw runtime_error("Not enough delete candidates");
    }
    delete_ids.resize(total_updates);
    return delete_ids;
}

// Uses the production search/update paths, with disjoint one-use update IDs.
// Each worker shuffles q/i/d triplets, maintaining equal operation proportions.
int run_favor_stress(BaselineIndex &index, const BaselineDatasetConfig &dataset,
                     const BaselineRunConfig &cfg) {
    if (cfg.search_mode != BaselineSearchMode::FAVOR || cfg.num_threads == 0 ||
        cfg.stress_seconds <= 0 || cfg.stress_ef <= 0 || cfg.ef_con <= 0 ||
        !std::isfinite(cfg.stress_delta) || cfg.stress_delta < 0)
        throw runtime_error("Invalid FAVOR stress configuration");
    auto insert_ids = make_insert_ids(index, index.nb - index.initial_ids.size(), 42);
    auto delete_ids = make_delete_ids(index, index.initial_ids.size(), 42);
    auto entries = sample_entry_points(index.edgemap, 100);
    vector<vector<QueryType>> queries;
    for (const auto &item : dataset.query_tsv_files) {
        auto parsed = read_queries_from_tsv(item.second);
        queries.push_back(std::move(parsed.first));
        if (queries.back().empty()) throw runtime_error("Empty stress query set");
    }
    SearchCacheList cache(cfg.num_threads + 5, index.nb, cfg.ef_con);
    vector<unique_ptr<icde::Visited>> visited;
    for (unsigned t=0; t<cfg.num_threads; ++t) visited.emplace_back(new icde::Visited(index.nb));
    FavorUpdateState state;
    state.delta_d = cfg.stress_delta; state.capacity = index.nb; state.ef_con = cfg.ef_con;
    atomic<size_t> next_insert{0}, next_delete{0};
    array<atomic<size_t>,3> completed{};
    atomic<bool> failed{false}, finished{false};
    string error;
    mutex error_mutex;
    using Clock = chrono::steady_clock;
    auto start = Clock::now();
    auto deadline = start + chrono::seconds(cfg.stress_seconds);
    cout << "STRESS_START method=FAVOR threads=" << cfg.num_threads
         << " ef_search=" << cfg.stress_ef << " ef_con=" << cfg.ef_con
         << " seconds=" << cfg.stress_seconds << " ratio=1:1:1 initial_live="
         << index.initial_ids.size() << " delta_d=" << setprecision(12) << state.delta_d << endl;
    thread reporter([&] {
        int ticks=0;
        while (!finished.load()) {
            this_thread::sleep_for(chrono::seconds(1));
            if (++ticks % 10 == 0)
                cout << "STRESS_PROGRESS seconds=" << ticks << " queries=" << completed[0].load()
                     << " inserts=" << completed[1].load() << " deletes=" << completed[2].load() << endl;
        }
    });
    #pragma omp parallel num_threads(cfg.num_threads)
    {
        int tid = omp_get_thread_num();
        mt19937 rng(42 + tid);
        array<int,3> ops{0,1,2};
        try {
            while (!failed.load() && Clock::now() < deadline) {
                shuffle(ops.begin(), ops.end(), rng);
                for (int op : ops) {
                    if (failed.load() || Clock::now() >= deadline) break;
                    if (op == 0) {
                        auto &set = queries[uniform_int_distribution<size_t>(0,queries.size()-1)(rng)];
                        int q = uniform_int_distribution<int>(0,min<int>(index.nq,set.size())-1)(rng);
                        const auto &qt = set[q];
                        function<bool(IdxType)> predicate = [&](IdxType id) {
                            const float *attr = get_live_attr_or_null(index.table_storage,id);
                            return attr && qt.check(attr);
                        };
                        double delta;
                        { lock_guard<mutex> lock(state.mu); delta=state.delta_d; }
                        int dc=0;
                        favor_search_one(index,entries,index.query_storage->get_vector(q),predicate,qt,
                                         cfg.K,cfg.stress_ef,delta,*visited[tid],dc);
                    } else if (op == 1) {
                        size_t n=next_insert.fetch_add(1);
                        if (n>=insert_ids.size()) throw runtime_error("Stress insertion IDs exhausted");
                        auto id=insert_ids[n];
                        char *vec=index.base_storage->get_vector(id);
                        if (!index.table_storage.insert(id,reinterpret_cast<const float*>(vec),
                                index.full_attr.data()+size_t(id)*index.na))
                            throw runtime_error("Stress table insertion failed");
                        vector<float> pool;
                        insert_point(index.edgemap,id,vec,index.table_storage,cache,entries,&pool);
                        double term=favor_insert_term(pool,cfg.ef_con);
                        lock_guard<mutex> lock(state.mu);
                        state.delta_d+=term/state.capacity;
                    } else {
                        size_t n=next_delete.fetch_add(1);
                        if (n>=delete_ids.size()) throw runtime_error("Stress deletion IDs exhausted");
                        auto id=delete_ids[n];
                        char *vec=get_live_vec_or_null(index.table_storage,id);
                        if (!vec) throw runtime_error("Stress deletion row missing");
                        delete_point(index.edgemap,id,vec,index.table_storage,cache,entries);
                        index.table_storage.remove(id);
                    }
                    ++completed[op];
                }
            }
        } catch (const exception &ex) {
            lock_guard<mutex> lock(error_mutex);
            if (!failed.exchange(true)) error=ex.what();
        }
    }
    double elapsed=chrono::duration<double>(Clock::now()-start).count();
    finished=true; reporter.join();
    size_t total=completed[0]+completed[1]+completed[2];
    cout << "RESULT_STRESS method=FAVOR status=" << (failed?"FAILED":"OK")
         << " seconds=" << elapsed << " queries=" << completed[0] << " inserts=" << completed[1]
         << " deletes=" << completed[2] << " ops_per_second=" << total/elapsed << endl;
    if (failed) throw runtime_error(error);
    return 0;
}

int run_baseline_mixed(BaselineIndex &index, const BaselineDatasetConfig &dataset,
                        const BaselineRunConfig &cfg) {
    int updates = int64_t(cfg.mixed_ops)*cfg.update_percent/200;
    auto ins=make_insert_ids(index,updates,42);
    auto del=make_delete_ids(index,updates,42);
    auto entries=sample_entry_points(index.edgemap,100);
    vector<vector<QueryType>> sets;
    int per_set=1000;
    for (const auto &item:dataset.query_tsv_files) {
        auto parsed=read_queries_from_tsv(item.second);
        sets.push_back(std::move(parsed.first));
        per_set=min(per_set,min(index.nq,int(sets.back().size())));
    }
    SearchCacheList insert_cache(cfg.num_threads+5,index.nb,cfg.ef_con);
    SearchCacheList query_cache(cfg.num_threads+5,index.nb,cfg.mixed_ef);
    bool is_favor=cfg.search_mode==BaselineSearchMode::FAVOR;
    FavorUpdateState state;
    state.capacity=index.nb; state.ef_con=cfg.ef_con; state.delta_d=cfg.stress_delta;
    vector<unique_ptr<icde::Visited>> visits;
    if (is_favor) {
        if (!std::isfinite(state.delta_d) || state.delta_d<0)
            throw runtime_error("Mixed FAVOR requires saved initial delta_d via --stress_delta");
        for(unsigned t=0;t<cfg.num_threads;++t) visits.emplace_back(new icde::Visited(index.nb));
    }
    auto get_vec=[&](IdxType id){return get_live_vec_or_null(index.table_storage,id);};
    auto query=[&](int qid) {
        // Interleave selectivities so incomplete cycles remain balanced.
        const auto &qt=sets[qid%sets.size()][qid/sets.size()];
        const char *q=index.query_storage->get_vector(qid/sets.size());
        function<bool(IdxType)> pred=[&](IdxType id){
            const float *a=get_live_attr_or_null(index.table_storage,id); return a && qt.check(a);
        };
        SearchRunStats stats;
        if(cfg.search_mode==BaselineSearchMode::FlatOOR)
            search_flat_oor(index.edgemap,entries,q,get_vec,index.distance_handler,index.dim,pred,query_cache,cfg.K,stats);
        else if(cfg.search_mode==BaselineSearchMode::HNSWOOR)
            search_hnsw_oor(index.edgemap,index.upper_layer,entries,q,get_vec,index.distance_handler,index.dim,pred,query_cache,cfg.K,stats);
        else if(is_favor) {
            double delta; {lock_guard<mutex> lock(state.mu);delta=state.delta_d;}
            favor_search_one(index,entries,q,pred,qt,cfg.K,cfg.mixed_ef,delta,*visits[omp_get_thread_num()],stats.n_cmps);
        } else search_navix(index.edgemap,entries,q,get_vec,index.distance_handler,index.dim,pred,query_cache,cfg.K,index.M,cfg.navix_n_entry,stats);
    };
    auto insert=[&](int i){
        auto id=ins[i]; auto vec=index.base_storage->get_vector(id);
        if(!index.table_storage.insert(id,reinterpret_cast<const float*>(vec),index.full_attr.data()+size_t(id)*index.na))
            throw runtime_error("Mixed insertion failed");
        vector<float> pool;
        insert_point(index.edgemap,id,vec,index.table_storage,insert_cache,entries,is_favor?&pool:nullptr);
        if(is_favor){double term=favor_insert_term(pool,cfg.ef_con);lock_guard<mutex> lock(state.mu);state.delta_d+=term/state.capacity;}
    };
    auto remove=[&](int i){
        auto id=del[i]; auto vec=get_vec(id);
        if(!vec)throw runtime_error("Mixed deletion row missing");
        delete_point(index.edgemap,id,vec,index.table_storage,insert_cache,entries);
        if(!index.table_storage.remove(id))throw runtime_error("Mixed deletion failed");
    };
    return run_mixed_workload(cfg.method_name,cfg.mixed_ops,cfg.update_percent,cfg.num_threads,
                             cfg.mixed_ef,per_set*sets.size(),query,insert,remove);
}

int run_baseline_experiment(const BaselineRunConfig &cfg) {
    try {
        auto dataset_cfg = get_baseline_dataset_config(cfg.dataset);
        auto index = load_baseline_index(dataset_cfg, cfg.artifact_method);

        if (cfg.mixed_ops != 0) return run_baseline_mixed(*index, dataset_cfg, cfg);
        if (cfg.stress_seconds != 0) return run_favor_stress(*index, dataset_cfg, cfg);

        if (cfg.calibrate_only) {
            auto entries = sample_entry_points(index->edgemap, 100);
            const bool is_favor = cfg.search_mode == BaselineSearchMode::FAVOR;
            float delta = 0;
            vector<unique_ptr<icde::Visited>> visited;
            if (is_favor) {
                double mean = favor_calibrate_mean(*index, entries, cfg.num_threads);
                if (cfg.favor_initial != "capacity" && cfg.favor_initial != "full")
                    throw runtime_error("favor_initial must be capacity or full");
                delta = cfg.favor_initial == "capacity" ? mean * double(index->initial_ids.size()) / index->nb : mean;
                cout << setprecision(10) << "CALIBRATION_FAVOR delta_d=" << delta << " mode=" << cfg.favor_initial << endl;
                for (unsigned t = 0; t < cfg.num_threads; ++t) visited.emplace_back(new icde::Visited(index->nb));
            }
            auto get_vec = [&](IdxType id) { return get_live_vec_or_null(index->table_storage, id); };
            auto search = [&](const char *query, const QueryType &qt, int ef, SearchCacheList &cache,
                              vector<pair<IdxType, float>> &results) {
                function<bool(IdxType)> pred = [&](IdxType id) {
                    const float *a = get_live_attr_or_null(index->table_storage, id);
                    return a && qt.check(a);
                };
                SearchRunStats stats;
                if (cfg.search_mode == BaselineSearchMode::FlatOOR)
                    search_flat_oor(index->edgemap, entries, query, get_vec, index->distance_handler,
                                    index->dim, pred, cache, cfg.K, stats);
                else if (cfg.search_mode == BaselineSearchMode::HNSWOOR)
                    search_hnsw_oor(index->edgemap, index->upper_layer, entries, query, get_vec,
                                    index->distance_handler, index->dim, pred, cache, cfg.K, stats);
                else if (is_favor) {
                    auto hits = favor_search_one(*index, entries, query, pred, qt, cfg.K, ef, delta,
                                                *visited[omp_get_thread_num()], stats.n_cmps);
                    for (auto &hit : hits) stats.results.emplace_back(hit.second, hit.first);
                } else
                    search_navix(index->edgemap, entries, query, get_vec, index->distance_handler,
                                 index->dim, pred, cache, cfg.K, index->M, cfg.navix_n_entry, stats);
                results = std::move(stats.results);
            };
            return calibrate_mixed_queries(cfg.method_name, cfg.dataset, dataset_cfg.query_tsv_files,
                index->base_storage, index->query_storage, index->distance_handler, index->full_attr,
                index->na, index->initial_ids, cfg.K, cfg.num_threads, cfg.calibration_max_ef, search, cfg.calibration_target);
        }

        // Full benchmark runs 100 batches over 20% of nb; n_batches selects how many
        // of those 100 batches to actually run, so batch_size stays fixed regardless.
        constexpr int kFullBatches = 100;
        int batch_size = (int)(index->nb * 0.2) / kFullBatches;
        int total_updates = cfg.n_batches * batch_size;
        cout << "Auto batch_size = " << batch_size << " (nb=" << index->nb
             << ", " << kFullBatches << " full batches = 20%, running n_batches="
             << cfg.n_batches << "/" << kFullBatches << ")" << endl;

        auto insert_ids = make_insert_ids(*index, total_updates, 42);
        auto delete_ids = make_delete_ids(*index, total_updates, 42);
        auto entry_points = sample_entry_points(index->edgemap, 100);

        cout << "\n=== Update plan ===" << endl;
        cout << "  Batches: " << cfg.n_batches << " x (" << batch_size << " inserts + "
             << batch_size << " deletes)" << endl;
        cout << "  Total inserts: " << total_updates << endl;
        cout << "  Total deletes: " << total_updates << endl;

        const bool is_favor = cfg.search_mode == BaselineSearchMode::FAVOR;
        FavorUpdateState favor_state;
        if (is_favor) {
            // Initial delta_d on the initial (80%) stock-HNSW graph, outside all timing.
            double mean = favor_calibrate_mean(*index, entry_points, cfg.num_threads);
            favor_state.capacity = index->nb;
            favor_state.ef_con = cfg.ef_con;
            if (cfg.favor_initial == "capacity")
                favor_state.delta_d = mean * double(index->initial_ids.size()) / index->nb;
            else if (cfg.favor_initial == "full")
                favor_state.delta_d = mean;
            else
                throw runtime_error("favor_initial must be capacity or full");
            cout << setprecision(10) << "FAVOR_DELTA_D_INITIAL=" << favor_state.delta_d
                 << " mode=" << cfg.favor_initial << " calibrated_mean=" << mean
                 << " capacity=" << favor_state.capacity << endl;
            cout << "  (Phase 0 pre-update evaluation skipped for FAVOR)" << endl;
        } else {
            // Pre-update eval with ef_search=200
            auto pre_stats = evaluate_filtered_queries(*index, cfg.search_mode,
                                                       "Phase 0: Pre-update filtered evaluation",
                                                       index->initial_ids, cfg.K, 200,
                                                       cfg.num_threads, cfg.navix_n_entry);
        }
        uint32_t update_threads = cfg.update_threads ? cfg.update_threads : cfg.num_threads;
        cout << "  Update threads: " << update_threads << ", query threads: " << cfg.num_threads << endl;
        auto update_stats = run_batched_updates(*index, cfg.n_batches, batch_size, cfg.ef_con,
                                                update_threads, insert_ids, delete_ids, entry_points,
                                                is_favor ? &favor_state : nullptr);
        if (is_favor)
            cout << setprecision(10) << "FAVOR_DELTA_D_FINAL=" << favor_state.delta_d
                 << " terms_added=" << favor_state.terms_added << endl;

        // ---- update RESULT lines (inserts / deletes) ----
        cout << "\n=== Update Summary ===" << endl;
        cout << "  Method: " << cfg.method_name << endl;
        cout << "  Dataset: " << cfg.dataset << endl;
        cout << "  Initial: " << index->initial_ids.size() << " / " << index->nb << " ("
             << fixed << setprecision(1) << 100.0 * index->initial_ids.size() / index->nb << "%)" << endl;
        cout << "  Batches: " << cfg.n_batches << " x " << batch_size
             << " ins + " << batch_size << " del" << endl;
        cout << "RESULT_INSERT"
             << " method=" << cfg.method_name
             << " dataset=" << cfg.dataset
             << " n_batches=" << cfg.n_batches
             << " batch_size=" << batch_size
             << " ins_qps=" << setprecision(0) << update_stats.avg_ins_qps
             << " ins_dc=" << setprecision(1) << update_stats.avg_ins_dc
             << endl;
        cout << "RESULT_DELETE"
             << " method=" << cfg.method_name
             << " dataset=" << cfg.dataset
             << " n_batches=" << cfg.n_batches
             << " batch_size=" << batch_size
             << " del_qps=" << setprecision(0) << update_stats.avg_del_qps
             << " del_dc=" << setprecision(1) << update_stats.avg_del_dc
             << endl;

        // ---- Phase 2: Post-update query sweep (3 selectivities x multiple ef_search) ----
        vector<int> ef_search_values = {10, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000};
        if (cfg.ef_descending) reverse(ef_search_values.begin(), ef_search_values.end());

        vector<IdxType> live_ids = live_ids_from_table(index->table_storage);
        cout << "\n=== Phase 2: Post-update query sweep ===" << endl;
        cout << "  Live points: " << live_ids.size() << " / " << index->nb << endl;

        vector<bool> is_live(index->nb, false);
        for (IdxType id : live_ids) is_live[id] = true;
        vector<IdxType> entry_points_search = sample_entry_points(index->edgemap, 100);
        auto get_vec = [&](IdxType id) -> char* {
            return get_live_vec_or_null(index->table_storage, id);
        };
        vector<unique_ptr<icde::Visited>> favor_visited;
        if (is_favor) {
            for (unsigned t = 0; t < cfg.num_threads; ++t) favor_visited.emplace_back(new icde::Visited(index->nb));
            cout << "  FAVOR search: live edge lists" << endl;
        }

        for (const auto &[sel_label, tsv_path] : dataset_cfg.query_tsv_files) {
            cout << "\n--- Selectivity: " << sel_label << " (" << tsv_path << ") ---" << endl;

            auto [sel_query_types, sel_header] = read_queries_from_tsv(tsv_path);
            index->query_types = std::move(sel_query_types);
            int nq_eval = min(index->nq, (int)index->query_types.size());

            auto predicate_2arg = [&](IdxType pid, int q) -> bool {
                const float *attr = get_live_attr_or_null(index->table_storage, pid);
                return attr != nullptr && index->query_types[q].check(attr);
            };
            auto predicate_factory = [&](int q) -> function<bool(IdxType)> {
                return [&, q](IdxType pid) -> bool {
                    const float *attr = get_live_attr_or_null(index->table_storage, pid);
                    return is_live[pid] && attr != nullptr && index->query_types[q].check(attr);
                };
            };

            cout << "  Computing filtered ground truth for " << nq_eval << " queries..." << endl;
            vector<vector<pair<IdxType, float>>> gt;
            {
                int gt_threads = omp_get_num_procs();
                omp_set_num_threads(gt_threads);
                compute_filtered_ground_truth(index->table_storage, index->query_storage,
                                              index->distance_handler, index->dim, live_ids,
                                              predicate_2arg, nq_eval, cfg.K, gt);
                omp_set_num_threads(cfg.num_threads);
            }

            for (int ef_s : ef_search_values) {
                SearchCacheList sweep_cache(cfg.num_threads + 5, index->nb, ef_s);

                int total_found = 0;
                long total_dist_comps = 0;
                auto t0 = chrono::high_resolution_clock::now();

                #pragma omp parallel for schedule(dynamic, 1) reduction(+: total_found, total_dist_comps)
                for (int q = 0; q < nq_eval; q++) {
                    auto query = index->query_storage->get_vector(q);
                    auto predicate = predicate_factory(q);
                    SearchRunStats sstats;
                    if (cfg.search_mode == BaselineSearchMode::FlatOOR) {
                        search_flat_oor(index->edgemap, entry_points_search, query, get_vec,
                                        index->distance_handler, index->dim, predicate,
                                        sweep_cache, cfg.K, sstats);
                    } else if (cfg.search_mode == BaselineSearchMode::HNSWOOR) {
                        search_hnsw_oor(index->edgemap, index->upper_layer, entry_points_search,
                                        query, get_vec, index->distance_handler, index->dim,
                                        predicate, sweep_cache, cfg.K, sstats);
                    } else if (is_favor) {
                        auto hits = favor_search_one(*index, entry_points_search, query, predicate, index->query_types[q],
                                                     cfg.K, ef_s, float(favor_state.delta_d),
                                                     *favor_visited[omp_get_thread_num()], sstats.n_cmps);
                        for (auto &h : hits) sstats.results.emplace_back(h.second, h.first);
                    } else {
                        search_navix(index->edgemap, entry_points_search, query, get_vec,
                                     index->distance_handler, index->dim, predicate,
                                     sweep_cache, cfg.K, index->M, cfg.navix_n_entry, sstats);
                    }

                    unordered_set<IdxType> gt_set;
                    for (auto &[id, dist] : gt[q]) gt_set.insert(id);
                    for (auto &[id, dist] : sstats.results)
                        if (gt_set.count(id)) total_found++;
                    total_dist_comps += sstats.n_cmps;
                }

                auto t1 = chrono::high_resolution_clock::now();
                double search_ms = chrono::duration_cast<chrono::microseconds>(t1 - t0).count() / 1e3;
                double recall = 100.0 * total_found / (nq_eval * cfg.K);
                double qps = nq_eval / (search_ms / 1e3);
                double avg_dc = (double)total_dist_comps / nq_eval;

                cout << "  ef_search=" << ef_s
                     << " recall=" << fixed << setprecision(2) << recall
                     << " qps=" << setprecision(0) << qps
                     << " avg_dc=" << setprecision(1) << avg_dc << endl;

                cout << "RESULT_QUERY"
                     << " method=" << cfg.method_name
                     << " dataset=" << cfg.dataset
                     << " selectivity=" << sel_label
                     << " K=" << cfg.K
                     << " ef_search=" << ef_s
                     << " recall=" << fixed << setprecision(2) << recall
                     << " qps=" << setprecision(0) << qps
                     << " avg_dc=" << setprecision(1) << avg_dc
                     << endl;
            }
        }

    } catch (const exception &ex) {
        cerr << ex.what() << endl;
        return 1;
    }

    return 0;
}

} // namespace ANNS
