#include "mixed_calibration.h"
#include "mixed_workload.h"
/**
 * online_update_pathfinder_test.cpp
 *
 * Loads partial (80%) PathFinder indices built by build_partial_pathfinder,
 * then runs 100 batches of mixed insert/delete (2000 each per batch)
 * to stress-test the tree-structured index under updates.
 *
 * After all batches: measures filtered recall@K and query QPS.
 *
 * Supports mixed attribute schemas: NUMERICAL (segment tree) and
 * CATEGORICAL (hash collection), determined from TSV query file headers.
 */

// #define SKIP_DELETE

#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <filesystem>
#include <queue>
#include <sstream>
#include <functional>

#include <boost/program_options.hpp>
#include <omp.h>

#include "hnsw/hnsw.h"
#include "irangegraph/attribute_tree.h"
#include "tablestorage/tablestorage_utils.h"
#include "querytype/querytype.h"
#include "queryplanning/queryplanning.h"
#include "run_search_online.h"
#include "utils.h"
#include "artifact_paths.h"

namespace po = boost::program_options;
using namespace std;
using namespace ANNS;

static string pick_existing_path(const vector<string> &candidates) {
    for (const auto &path : candidates) {
        if (filesystem::exists(path))
            return path;
    }
    return candidates.empty() ? string() : candidates.front();
}

// =====================================================================
//  Helpers
// =====================================================================

/// Walk a NUMERICAL segment tree from root to leaf, collecting all non-root
/// nodes whose attribute range contains `val`.
static vector<AttrTreeNode *> find_containing_nodes_tree(
        AttrSegmentTree *tree, float val) {
    vector<AttrTreeNode *> result;
    AttrTreeNode *cur = tree->root;
    while (cur) {
        if (cur->depth > 0)  // skip root (edgemap is empty)
            result.push_back(cur);
        AttrTreeNode *next = nullptr;
        for (auto *c : cur->childs) {
            if (c->lbound <= val && val < c->rbound_open) {
                next = c;
                break;
            }
        }
        cur = next;
    }
    return result;
}

/// For a CATEGORICAL hash collection, find the single non-root node for
/// the given category value.  Returns empty if the value has no bucket.
static vector<AttrTreeNode *> find_containing_nodes_hash(
        AttrHashCollection *hash, float val) {
    vector<AttrTreeNode *> result;
    auto it = hash->hash_table.find(val);
    if (it != hash->hash_table.end() && it->second != hash->root_collection)
        result.push_back(it->second);
    return result;
}

/// Unified: find the non-root subgraph nodes that a point belongs to.
static vector<AttrTreeNode *> find_containing_nodes(
        shared_ptr<irg_attr> &index, float val) {
    if (index->schema == irg_attr::NUMERICAL)
        return find_containing_nodes_tree(index->tree, val);
    else
        return find_containing_nodes_hash(index->hash, val);
}

/// Sample a few entry points from a ConcurrentEdgeMap by iterating its
/// hash table (same strategy as search_graph_reference_OOR in run_search.cpp).
/// SINGLE-THREADED ONLY.  Call before the parallel update phase and cache.
static vector<IdxType> sample_entry_points(ConcurrentEdgeMap &edgemap, int n) {
    vector<IdxType> entries;
    for (auto it = edgemap.begin(); it != edgemap.end(); ++it) {
        auto kv = *it;
        entries.push_back(kv.first);
        if ((int)entries.size() >= n) break;
    }
    return entries;
}

static char *get_live_vec_or_null(TableStorage &table_storage, IdxType id) {
    void *row = table_storage.find(id);
    return row == nullptr ? nullptr : table_storage.get_vector(row);
}

static const float *get_live_attr_or_null(TableStorage &table_storage, IdxType id) {
    void *row = table_storage.find(id);
    return row == nullptr ? nullptr : table_storage.get_attr(row);
}

static void init_table_storage_schema(shared_ptr<IStorage> base_storage,
                                      int na,
                                      TableStorage &table_storage) {
    table_storage.nb = base_storage->get_num_points();
    table_storage.na = na;
    table_storage.n_dim = base_storage->get_dim();
    table_storage.dim_byte_count = 4;
    table_storage.vector_bytes = (size_t)table_storage.dim_byte_count * table_storage.n_dim;
    table_storage.attr_offset = TableStorage::roundUp(table_storage.vector_bytes, 4);
    table_storage.aligned_record_size = TableStorage::roundUp(
        table_storage.attr_offset + na * sizeof(float), 32);
    table_storage.table_schema.resize(na, TableStorage::NUMERICAL);
}

static vector<IdxType> live_ids_from_table(TableStorage &table_storage) {
    return table_storage.keys();
}

/// Filtered brute-force ground truth.
/// For each query, find top-K among `live_ids` that satisfy `predicate(pid, q)`.
static void compute_filtered_ground_truth(
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
        priority_queue<pair<float, IdxType>> heap;  // max-heap
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

/// Filtered beam search using PathFinderPlanner to select subgraphs,
/// then OOR post-filter beam search on each selected edgemap.
static void pathfinder_filtered_search(
        IQueryPlanner &planner,
        const char *query,
        TableStorage &table_storage,
        shared_ptr<DistanceHandler> distance_handler,
        int dim,
        function<bool(IdxType)> predicate,
        const QueryType &qt,
        SearchCacheList &cache_list,
        int K, int ef_search,
        vector<pair<IdxType, float>> &results_out,
        const unordered_map<AttrTreeNode*, vector<IdxType>> *fixed_entries = nullptr) {

    results_out.clear();

    auto plan = planner(qt);

    unordered_set<IdxType> seen;
    priority_queue<pair<float, IdxType>> top_candidates;  // max-heap

    auto record_top = [&](IdxType pid, float dist) {
        top_candidates.emplace(dist, pid);
        while ((int)top_candidates.size() > K) top_candidates.pop();
    };

    for (const auto &graphref : plan) {
        AttrTreeNode *ag_ptr = graphref.ag_ptr;
        if (ag_ptr->delegate_root)
            ag_ptr = ag_ptr->delegate_root;

        if (ag_ptr->edgelist_hashmap.size() == 0) continue;

        auto cache = cache_list.get_free_cache();
        auto &sq = cache->search_queue;
        auto &vs = cache->visited_set;
        sq.clear();
        vs.clear();

        // Mixed workloads use seeds captured before updates, avoiding iteration
        // over the hash map while other threads insert/erase its entries.
        vector<IdxType> seeds;
        if (fixed_entries) {
            auto it=fixed_entries->find(ag_ptr);
            if(it!=fixed_entries->end()) seeds=it->second;
        } else {
            for(auto it=ag_ptr->edgelist_hashmap.begin();it!=ag_ptr->edgelist_hashmap.end();++it) {
                seeds.push_back((*it).first);
                if(seeds.size()>=100)break;
            }
        }
        int cntr=5;
        for (IdxType entry : seeds) {
            if (vs.check(entry)) continue;
            char *entry_vec = get_live_vec_or_null(table_storage, entry);
            if (entry_vec == nullptr) continue;
            vs.set(entry);
            float dist = distance_handler->compute(query, entry_vec, dim);
            ++online_dist_comps;
            sq.insert(entry, dist);
            if (predicate(entry) && seen.insert(entry).second)
                record_top(entry, dist);
            if (--cntr <= 0) break;
        }

        ConcurrentEdgeMap::InPlaceReadGuard rguard;
        std::vector<IdxType> edges;
        while (sq.has_unexpanded_node()) {
            const Candidate cur = sq.get_closest_unexpanded();

            if (!ag_ptr->edgelist_hashmap.try_read_lock(cur.id, rguard)) continue;
            edges = *rguard.edges;
            rguard.release();  // don't hold bucket lock across distance compute

            for (IdxType nbr : edges) {
                if (vs.check(nbr)) continue;
                char *nbr_vec = get_live_vec_or_null(table_storage, nbr);
                if (nbr_vec == nullptr) continue;
                vs.set(nbr);
                float dist = distance_handler->compute(query, nbr_vec, dim);
                ++online_dist_comps;
                sq.insert(nbr, dist);
                if (predicate(nbr) && seen.insert(nbr).second)
                    record_top(nbr, dist);
            }
        }

        cache_list.release_cache(cache);
    }

    // Extract results
    while (!top_candidates.empty()) {
        auto [dist, id] = top_candidates.top();
        top_candidates.pop();
        results_out.emplace_back(id, dist);
    }
    sort(results_out.begin(), results_out.end(),
         [](auto &a, auto &b) { return a.second < b.second; });
    if ((int)results_out.size() > K) results_out.resize(K);
}

// =====================================================================
//  Shared context for all phases
// =====================================================================
struct TestContext {
    // Dataset
    shared_ptr<IStorage> base_storage, query_storage;
    shared_ptr<DistanceHandler> distance_handler;
    TableStorage *table_storage;
    int nb, nq, dim, na, K, ef_search, ef_con;
    uint32_t num_threads;
    int n_batches, batch_size, total_updates;

    // Attributes & queries
    vector<float> *full_attr;
    vector<QueryType> *query_types;
    vector<bool> *is_numerical;

    // Index structures
    vector<shared_ptr<irg_attr>> *trees;
    AttrTreeNode *rootnode;
    vector<IdxType> *initial_ids;
    int n_initial;

    // Entry point caches (mutable across phases)
    unordered_map<int, unordered_map<int, vector<IdxType>>> *node_entries;
    vector<IdxType> *root_entry_points;

    // ID lists
    vector<IdxType> *insert_ids;
    vector<IdxType> *delete_ids;

    // Search / insert caches
    SearchCacheList *insert_cache_list;
    SearchCacheList *search_cache_list;

    // Query planner
    PathFinderPlanner *planner;

    // Helpers
    function<vector<AttrTreeNode*>(shared_ptr<irg_attr>&)> get_all_nodes;
};

struct EvalStats {
    double recall = 0.0;
    double qps = 0.0;
    double avg_dist_comps = 0.0;
    double search_ms = 0.0;
    size_t live_points = 0;
};

struct UpdateStats {
    double avg_ins_ms = 0.0;
    double avg_del_ms = 0.0;
    double avg_ins_qps = 0.0;
    double avg_del_qps = 0.0;
    double avg_ins_dc = 0.0;
    double avg_del_dc = 0.0;
    double avg_update_qps = 0.0;
    vector<double> avg_del_dc_by_layer;
    vector<int> graph_count_by_layer;
};

static int compute_max_graph_layer(
        const vector<shared_ptr<irg_attr>> &trees,
        const function<vector<AttrTreeNode*>(shared_ptr<irg_attr>&)> &get_all_nodes) {
    int max_layer = 0;
    for (const auto &tree : trees) {
        auto nodes = get_all_nodes(const_cast<shared_ptr<irg_attr>&>(tree));
        for (auto *node : nodes) {
            max_layer = max(max_layer, node->depth);
        }
    }
    return max_layer;
}

static vector<int> compute_graph_count_by_layer(
        AttrTreeNode *rootnode,
        const vector<shared_ptr<irg_attr>> &trees,
        const function<vector<AttrTreeNode*>(shared_ptr<irg_attr>&)> &get_all_nodes,
        int max_layer) {
    vector<int> graph_count_by_layer(max_layer + 1, 0);
    graph_count_by_layer[0] = (rootnode != nullptr) ? 1 : 0;
    for (const auto &tree : trees) {
        auto nodes = get_all_nodes(const_cast<shared_ptr<irg_attr>&>(tree));
        for (auto *node : nodes) {
            if (node->depth <= 0) continue;
            graph_count_by_layer[node->depth]++;
        }
    }
    return graph_count_by_layer;
}

static string format_layer_dc_breakdown(
        const vector<double> &per_layer_dc,
        const vector<int> &graph_count_by_layer) {
    ostringstream oss;
    for (size_t layer = 0; layer < per_layer_dc.size(); layer++) {
        if (layer > 0) oss << " ";
        oss << "L" << layer;
        if (layer == 0) oss << "[root";
        else oss << "[" << graph_count_by_layer[layer] << "g";
        oss << "]=" << fixed << setprecision(0) << per_layer_dc[layer];
    }
    return oss.str();
}

// =====================================================================
//  Evaluate filtered recall and QPS on a given set of live IDs
// =====================================================================
static EvalStats evaluate_recall(TestContext &ctx, const string &label,
                                 const vector<IdxType> &live_ids) {
    cout << "\n=== " << label << " ===" << endl;
    cout << "  Live points: " << live_ids.size() << " / " << ctx.nb << endl;

    int nq_tsv = ctx.query_types->size();
    int nq_eval = min(ctx.nq, nq_tsv);
    int K = ctx.K;
    auto &query_types = *ctx.query_types;

    auto predicate_2arg = [&](IdxType pid, int q) -> bool {
        const float *attr = get_live_attr_or_null(*ctx.table_storage, pid);
        return attr != nullptr && query_types[q].check(attr);
    };
    auto predicate_factory = [&](int q) -> function<bool(IdxType)> {
        return [&, q](IdxType pid) -> bool {
            const float *attr = get_live_attr_or_null(*ctx.table_storage, pid);
            return attr != nullptr && query_types[q].check(attr);
        };
    };

    cout << "  Computing filtered ground truth for " << nq_eval << " queries..." << endl;
    vector<vector<pair<IdxType, float>>> gt;
    {
        int gt_threads = omp_get_num_procs();
        omp_set_num_threads(gt_threads);
        compute_filtered_ground_truth(
            *ctx.table_storage, ctx.query_storage, ctx.distance_handler, ctx.dim,
            live_ids, predicate_2arg, nq_eval, K, gt);
        omp_set_num_threads(ctx.num_threads);
    }

    cout << "  Running filtered search (ef_search=" << ctx.ef_search << ")..." << endl;
    int total_found = 0;
    long total_dist_comps = 0;
    auto t0 = chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic, 1) reduction(+: total_found, total_dist_comps)
    for (int q = 0; q < nq_eval; q++) {
        auto query = ctx.query_storage->get_vector(q);
        auto pred = predicate_factory(q);
        vector<pair<IdxType, float>> results;
        long before = online_dist_comps;
        pathfinder_filtered_search(
            *ctx.planner, query, *ctx.table_storage,
            ctx.distance_handler, ctx.dim,
            pred, query_types[q],
            *ctx.search_cache_list, K, ctx.ef_search, results);

        unordered_set<IdxType> gt_set;
        for (auto &[id, d] : gt[q]) gt_set.insert(id);
        int found = 0;
        for (auto &[id, d] : results)
            if (gt_set.count(id)) found++;
        total_found += found;
        total_dist_comps += online_dist_comps - before;
    }

    auto t1 = chrono::high_resolution_clock::now();
    double search_ms = chrono::duration_cast<chrono::microseconds>(t1 - t0).count() / 1e3;
    double recall = 100.0 * total_found / (nq_eval * K);
    double qps = nq_eval / (search_ms / 1e3);
    double avg_dist_comps = (double)total_dist_comps / nq_eval;

    cout << "  Recall@" << K << " = " << fixed << setprecision(2) << recall << "%" << endl;
    cout << "  QPS = " << setprecision(0) << qps << endl;
    cout << "  Avg dist comps/query = " << setprecision(1) << avg_dist_comps << endl;
    cout << "  Search wall time = " << setprecision(1) << search_ms << " ms" << endl;

    EvalStats stats;
    stats.recall = recall;
    stats.qps = qps;
    stats.avg_dist_comps = avg_dist_comps;
    stats.search_ms = search_ms;
    stats.live_points = live_ids.size();
    return stats;
}

// =====================================================================
//  Phase 1: Batched insert + delete
// =====================================================================
static UpdateStats phase1_batched_updates(TestContext &ctx) {
    int n_batches = ctx.n_batches, batch_size = ctx.batch_size;
    int na = ctx.na;
    auto &full_attr = *ctx.full_attr;
    auto &trees = *ctx.trees;
    auto &insert_ids = *ctx.insert_ids;
    auto &delete_ids = *ctx.delete_ids;
    auto &node_entries = *ctx.node_entries;
    auto &root_entry_points = *ctx.root_entry_points;
    auto &get_all_nodes = ctx.get_all_nodes;
    int max_layer = compute_max_graph_layer(trees, get_all_nodes);
    vector<int> graph_count_by_layer = compute_graph_count_by_layer(
        ctx.rootnode, trees, get_all_nodes, max_layer);
    string layer_legend = format_layer_dc_breakdown(
        vector<double>(max_layer + 1, 0.0), graph_count_by_layer);

    cout << "\n=== Phase 1: " << n_batches << " batches ===" << endl;
    cout << "        |   time duration   |     op per sec    |  distance comps | delete dc by layer" << endl;
    cout << "  batch |  ins_ms |  del_ms | ins_QPS | del_QPS | ins_dc | del_dc | " << layer_legend << endl;

    vector<double> batch_ins_ms(n_batches), batch_del_ms(n_batches);
    vector<long> batch_ins_dc(n_batches), batch_del_dc(n_batches);
    vector<vector<long>> batch_del_dc_by_layer(n_batches, vector<long>(max_layer + 1, 0));

    for (int b = 0; b < n_batches; b++) {
        int start = b * batch_size;
        int end = start + batch_size;

        // ---- insertions ----
        long ins_dc_total = 0;
        auto t_ins_start = chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule(dynamic, 4) reduction(+: ins_dc_total)
        for (int i = start; i < end; i++) {
            IdxType gid = insert_ids[i];
            const char *vec = ctx.base_storage->get_vector(gid);
            const float *attr = full_attr.data() + (size_t)gid * na;
            if (!ctx.table_storage->insert(gid, reinterpret_cast<const float *>(vec), attr)) {
                throw runtime_error("Failed to insert row into TableStorage during PathFinder update");
            }
            long before = online_dist_comps;

            insert_point(ctx.rootnode->edgelist_hashmap, gid, vec,
                         *ctx.table_storage, *ctx.insert_cache_list, root_entry_points);

            for (int a = 0; a < na; a++) {
                float val = attr[a];
                auto nodes = find_containing_nodes(trees[a], val);
                for (auto *node : nodes) {
                    auto it = node_entries[a].find(node->node_id);
                    if (it == node_entries[a].end()) continue;
                    insert_point(node->edgelist_hashmap, gid, vec,
                                 *ctx.table_storage, *ctx.insert_cache_list, it->second);
                }
            }
            ins_dc_total += online_dist_comps - before;
        }

        auto t_ins_end = chrono::high_resolution_clock::now();
        double ins_ms = chrono::duration_cast<chrono::microseconds>(
            t_ins_end - t_ins_start).count() / 1e3;

        // ---- deletions ----
        long del_dc_total = 0;
        vector<long> del_dc_by_layer(max_layer + 1, 0);
        auto t_del_start = chrono::high_resolution_clock::now();

        #ifndef SKIP_DELETE
        #pragma omp parallel
        {
            long del_dc_total_local = 0;
            vector<long> del_dc_by_layer_local(max_layer + 1, 0);

            #pragma omp for schedule(dynamic, 4) nowait
            for (int i = start; i < end; i++) {
                IdxType gid = delete_ids[i];
                char *vec = get_live_vec_or_null(*ctx.table_storage, gid);
                const float *attr = get_live_attr_or_null(*ctx.table_storage, gid);
                if (vec == nullptr || attr == nullptr) continue;
                long before = online_dist_comps;

                long before_graph = online_dist_comps;
                delete_point(ctx.rootnode->edgelist_hashmap, gid, vec,
                             *ctx.table_storage, *ctx.insert_cache_list, root_entry_points);
                del_dc_by_layer_local[0] += online_dist_comps - before_graph;

                for (int a = 0; a < na; a++) {
                    float val = attr[a];
                    auto nodes = find_containing_nodes(trees[a], val);
                    for (auto *node : nodes) {
                        auto it = node_entries[a].find(node->node_id);
                        if (it == node_entries[a].end()) continue;
                        before_graph = online_dist_comps;
                        delete_point(node->edgelist_hashmap, gid, vec,
                                     *ctx.table_storage, *ctx.insert_cache_list, it->second);
                        del_dc_by_layer_local[node->depth] += online_dist_comps - before_graph;
                    }
                }
                ctx.table_storage->remove(gid);
                del_dc_total_local += online_dist_comps - before;
            }

            #pragma omp critical
            {
                del_dc_total += del_dc_total_local;
                for (int layer = 0; layer <= max_layer; layer++) {
                    del_dc_by_layer[layer] += del_dc_by_layer_local[layer];
                }
            }
        }
        #endif

        auto t_del_end = chrono::high_resolution_clock::now();
        double del_ms = chrono::duration_cast<chrono::microseconds>(
            t_del_end - t_del_start).count() / 1e3;

        batch_ins_ms[b] = ins_ms;
        batch_del_ms[b] = del_ms;
        batch_ins_dc[b] = ins_dc_total;
        batch_del_dc[b] = del_dc_total;
        batch_del_dc_by_layer[b] = del_dc_by_layer;

        // Resample entry points from live edgemap nodes
        root_entry_points = sample_entry_points(ctx.rootnode->edgelist_hashmap, 100);
        for (int a = 0; a < na; a++) {
            auto all_nodes_vec = get_all_nodes(trees[a]);
            for (auto *node : all_nodes_vec) {
                if (node->edgelist_hashmap.size() == 0) continue;
                node_entries[a][node->node_id] = sample_entry_points(node->edgelist_hashmap, 100);
            }
        }

        if (b % 10 == 0 || b == n_batches - 1) {
            vector<double> per_delete_dc_by_layer(max_layer + 1, 0.0);
            for (int layer = 0; layer <= max_layer; layer++) {
                per_delete_dc_by_layer[layer] = (double)del_dc_by_layer[layer] / batch_size;
            }
            cout << "  " << setw(5) << (b + 1)
                 << " | " << setw(7) << fixed << setprecision(1) << ins_ms
                 << " | " << setw(7) << del_ms
                 << " | " << setw(7) << setprecision(0) << batch_size / (ins_ms / 1e3)
                 << " | " << setw(7) << batch_size / (del_ms / 1e3)
                 << " | " << setw(6) << setprecision(0) << (double)ins_dc_total / batch_size
                 << " | " << setw(6) << (double)del_dc_total / batch_size
                 << " | "
                 << format_layer_dc_breakdown(per_delete_dc_by_layer, graph_count_by_layer)
                 << endl;
        }
    }

    double avg_ins_ms = accumulate(batch_ins_ms.begin(), batch_ins_ms.end(), 0.0) / n_batches;
    double avg_del_ms = accumulate(batch_del_ms.begin(), batch_del_ms.end(), 0.0) / n_batches;
    double avg_ins_dc = (double)accumulate(batch_ins_dc.begin(), batch_ins_dc.end(), 0L)
                        / ((double)n_batches * batch_size);
    double avg_del_dc = (double)accumulate(batch_del_dc.begin(), batch_del_dc.end(), 0L)
                        / ((double)n_batches * batch_size);
    vector<double> avg_del_dc_by_layer(max_layer + 1, 0.0);
    for (int layer = 0; layer <= max_layer; layer++) {
        long total_layer_dc = 0;
        for (int b = 0; b < n_batches; b++) {
            total_layer_dc += batch_del_dc_by_layer[b][layer];
        }
        avg_del_dc_by_layer[layer] = (double)total_layer_dc / ((double)n_batches * batch_size);
    }

    cout << "\n=== Update throughput (averaged over " << n_batches << " batches) ===" << endl;
    cout << "  Avg insert batch: " << fixed << setprecision(1) << avg_ins_ms << " ms → "
         << setprecision(0) << batch_size / (avg_ins_ms / 1e3) << " inserts/s, "
         << setprecision(1) << avg_ins_dc << " dist comps/insert" << endl;
    cout << "  Avg delete batch: " << setprecision(1) << avg_del_ms << " ms → "
         << setprecision(0) << batch_size / (avg_del_ms / 1e3) << " deletes/s, "
         << setprecision(1) << avg_del_dc << " dist comps/delete" << endl;
    cout << "  Avg delete dist comps by layer (/delete): "
         << format_layer_dc_breakdown(avg_del_dc_by_layer, graph_count_by_layer) << endl;

    UpdateStats stats;
    stats.avg_ins_ms = avg_ins_ms;
    stats.avg_del_ms = avg_del_ms;
    stats.avg_ins_qps = batch_size / (avg_ins_ms / 1e3);
    stats.avg_del_qps = batch_size / (avg_del_ms / 1e3);
    stats.avg_ins_dc = avg_ins_dc;
    stats.avg_del_dc = avg_del_dc;
    stats.avg_update_qps = (2.0 * batch_size) / ((avg_ins_ms + avg_del_ms) / 1e3);
    stats.avg_del_dc_by_layer = std::move(avg_del_dc_by_layer);
    stats.graph_count_by_layer = std::move(graph_count_by_layer);
    return stats;
}

// =====================================================================
//  Phase 2: Build post-update live set, then evaluate
// =====================================================================
static EvalStats phase2_post_update_eval(TestContext &ctx) {
    vector<IdxType> live_ids = live_ids_from_table(*ctx.table_storage);
    return evaluate_recall(ctx, "Phase 2: Post-update filtered evaluation", live_ids);
}

// =====================================================================
//  Main
// =====================================================================
int main(int argc, char **argv) {
    uint32_t num_threads = 16;
    int ef_con = 200;
    int K = 10;
    string dataset = "sift3";
    string index_dir;
    int n_batches = 100;
    bool use_vamana_root = false;
    bool calibrate_only = false;
    int calibration_max_ef = 3000;
    int calibration_target = 0;
    int mixed_ops=0, update_percent=1, mixed_ef=30;

    try {
        po::options_description desc{"Online Update PathFinder Test"};
        desc.add_options()
            ("help,h", "Print help")
            ("mixed_ops", po::value<int>(&mixed_ops)->default_value(0))
            ("update_percent", po::value<int>(&update_percent)->default_value(1))
            ("mixed_ef", po::value<int>(&mixed_ef)->default_value(30))
            ("calibrate_only", po::bool_switch(&calibrate_only), "Calibrate initial mixed-query recall; do not update")
            ("calibration_max_ef", po::value<int>(&calibration_max_ef)->default_value(3000))
            ("calibration_target", po::value<int>(&calibration_target)->default_value(0), "Strict recall target; 0 evaluates 90 and 95")
            ("dataset", po::value<string>(&dataset)->default_value("sift3"))
            ("index_dir", po::value<string>(&index_dir), "Directory containing saved update indexes and ID mappings")
            ("num_threads", po::value<uint32_t>(&num_threads)->default_value(16))
            ("ef_con", po::value<int>(&ef_con)->default_value(200))
            ("n_batches", po::value<int>(&n_batches)->default_value(100))
            ("vamana_root", po::bool_switch(&use_vamana_root), "Use Vamana flat graph as root instead of HNSW")
            ;

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) { cout << desc; return 0; }
        po::notify(vm);
    } catch (const exception &ex) {
        cerr << ex.what() << endl;
        return 1;
    }

    omp_set_num_threads(num_threads);

    // ---- dataset paths ----
    string data_root = "data/";
    string attr_root = pf_path("workloads/mar30_new_mixed_attributes/step0_attributes/");
    string index_root = "data/index_files/"
                        "apr1_online_update/";

    string base_bin_file, query_bin_file, attr_bin_file, index_prefix;
    // query_tsv_files: one per selectivity level (low, mid, high) for post-update eval
    vector<pair<string, string>> query_tsv_files;  // {label, path}

    string tsv_root = pf_path("workloads/mar30_new_mixed_attributes/step2/");

    if (dataset == "sift3") {
        base_bin_file  = data_root + "sift/sift_base.bin";
        query_bin_file = data_root + "sift/sift_query.bin";
        attr_bin_file  = attr_root + "SIFT-mixed_attrs.bin";
        index_prefix   = index_root + "sift3-pathfinder/";
        online_M = 32;
        query_tsv_files = {
            {"low",  tsv_root + "pred_sift_a012_low.tsv"},
            {"mid",  tsv_root + "pred_sift_a012_mid.tsv"},
            {"high", tsv_root + "pred_sift_a012_high.tsv"},
        };
    } else if (dataset == "gist3") {
        base_bin_file  = data_root + "gist/gist_base.bin";
        query_bin_file = data_root + "gist/gist_query.bin";
        attr_bin_file  = attr_root + "GIST-mixed_attrs.bin";
        index_prefix   = index_root + "gist3-pathfinder/";
        online_M = 64;
        query_tsv_files = {
            {"low",  tsv_root + "pred_gist_a012_low.tsv"},
            {"mid",  tsv_root + "pred_gist_a012_mid.tsv"},
            {"high", tsv_root + "pred_gist_a012_high.tsv"},
        };
    } else if (dataset == "redcaps") {
        string home_redcaps = "data/redcaps/";
        string redcaps_tsv_root = "data/icde_inputs/tsv/";
        base_bin_file  = home_redcaps + "redcaps-embeddings.bin";
        query_bin_file = home_redcaps + "redcaps-query.bin";
        attr_bin_file  = home_redcaps + "redcaps-attributes-subreddit-time-upvote.bin";
        index_prefix   = index_root + "redcaps-pathfinder/";
        online_M = 64;
        query_tsv_files = {
            {"low",  redcaps_tsv_root + "redcaps_two_three_low.tsv"},
            {"mid",  redcaps_tsv_root + "redcaps_two_three_mid.tsv"},
            {"high", redcaps_tsv_root + "redcaps_two_three_high.tsv"},
        };
    } else {
        cerr << "Unknown dataset: " << dataset << endl;
        return 1;
    }
    if (!index_dir.empty())
        index_prefix = index_dir + (index_dir.back() == '/' ? "" : "/");
    cout << "Index directory: " << index_prefix << endl;
    online_vamana_alpha = 1.2f;

    // ---- load full base vectors ----
    cout << "Loading base vectors..." << endl;
    auto base_storage = create_storage("float");
    base_storage->load_from_file(base_bin_file, "");
    online_nb = base_storage->get_num_points();
    online_dim = base_storage->get_dim();
    int nb = online_nb, dim = online_dim;

    // Full benchmark runs 100 batches over 20% of nb; n_batches selects how many
    // of those 100 batches to actually run, so batch_size stays fixed regardless.
    constexpr int kFullBatches = 100;
    int batch_size = (int)(nb * 0.2) / kFullBatches;
    cout << "Auto batch_size = " << batch_size << " (nb=" << nb
         << ", " << kFullBatches << " full batches = 20%, running n_batches="
         << n_batches << "/" << kFullBatches << ")" << endl;

    cout << "Loading query vectors..." << endl;
    auto query_storage = create_storage("float");
    query_storage->load_from_file(query_bin_file, "");
    int nq = query_storage->get_num_points();
    cout << "  nb=" << nb << " nq=" << nq << " dim=" << dim << endl;

    online_distance_handler = get_distance_handler("float", "L2");

    // ---- load full attributes ----
    cout << "Loading attributes..." << endl;
    vector<float> full_attr = read_binary_float32(attr_bin_file);
    int na = full_attr.size() / nb;
    cout << "  na=" << na << endl;

    // ---- load initial ID mapping for PathFinder trees / HNSW root ----
    string mapping_path = pick_existing_path({
        index_prefix + "initial_ids_hnsw.bin",
        index_prefix + "initial_ids.bin",
    });
    cout << "Loading PathFinder ID mapping from " << mapping_path << "..." << endl;
    vector<IdxType> initial_ids = load_id_mapping(mapping_path);
    int n_initial = initial_ids.size();
    cout << "  " << n_initial << " nodes in initial graph" << endl;

    // ---- determine per-attribute schema from TSV header (use first TSV) ----
    string schema_tsv_file = query_tsv_files[0].second;
    cout << "Loading schema from " << schema_tsv_file << "..." << endl;
    auto [query_types, query_header] = read_queries_from_tsv(schema_tsv_file);
    query_header.describe();
    if ((int)query_header.types.size() != na) {
        cerr << "TSV has " << query_header.types.size()
             << " columns but attribute file has " << na << " attrs" << endl;
        return 1;
    }
    vector<bool> is_numerical(na);
    for (int i = 0; i < na; i++) {
        is_numerical[i] = (query_header.types[i] == QueryType::PredType::RANGE);
        cout << "  attr " << i << ": " << (is_numerical[i] ? "NUMERICAL" : "CATEGORICAL") << endl;
    }

    // ---- load and translate trees/hashes ----
    vector<shared_ptr<irg_attr>> trees(na);

    // Helper: collect all nodes from an index (regardless of schema)
    auto get_all_nodes = [](shared_ptr<irg_attr> &idx) -> vector<AttrTreeNode*> {
        if (idx->schema == irg_attr::NUMERICAL)
            return idx->tree->treenodes;
        else
            return idx->hash->hashednodes;
    };

    for (int a = 0; a < na; a++) {
        trees[a] = make_shared<irg_attr>();
        trees[a]->_attr_id = a;
        trees[a]->_attr = &full_attr;
        trees[a]->_na = na;

        if (is_numerical[a]) {
            string tree_path = index_prefix + "attr" + to_string(a) + "_tree.bin";
            cout << "\nLoading NUMERICAL tree for attribute " << a
                 << " from " << tree_path << "..." << endl;
            trees[a]->schema = irg_attr::NUMERICAL;
            trees[a]->load_tree(tree_path, base_storage, online_distance_handler,
                                online_M, ef_con);
            cout << "  Tree nodes: " << trees[a]->tree->treenodes.size() << endl;
        } else {
            string hash_path = index_prefix + "attr" + to_string(a) + "_hash.bin";
            cout << "\nLoading CATEGORICAL hash for attribute " << a
                 << " from " << hash_path << "..." << endl;
            trees[a]->schema = irg_attr::CATEGORICAL;
            trees[a]->load_hash(hash_path, base_storage, online_distance_handler,
                                online_M, ef_con);
            cout << "  Hash buckets: " << trees[a]->hash->hash_table.size() << endl;
        }

        auto all_nodes = get_all_nodes(trees[a]);

        // Remap vertex_table keys AND graph edges: packed_storage_id → global_id.
        // The saved graph already has edges translated to packed storage IDs
        // (translate_edge_destination was called during build before save),
        // so we must NOT call translate_edge_destination again — just remap both
        // vertex_table keys and edge destinations directly.
        for (auto *node : all_nodes) {
            // Remap vertex_table keys
            unordered_map<IdxType, IdxType> new_vtable;
            new_vtable.reserve(node->vertex_table.size());
            for (auto &[packed_id, within_node_id] : node->vertex_table) {
                IdxType global_id = initial_ids.at(packed_id);
                new_vtable[global_id] = within_node_id;
            }
            node->vertex_table = std::move(new_vtable);

            // Remap graph edge destinations from packed to global
            if (node->g) {
                for (int u = 0; u < node->size; u++) {
                    for (IdxType &nbr : node->g->neighbors[u]) {
                        nbr = initial_ids.at(nbr);
                    }
                }
            }
        }
        // Map to edgemaps (edges are now in global ID space)
        // Reserve 2x initial size to avoid rehashing during insertions
        for (auto *node : all_nodes) {
            node->map_global_id_directly_to_edge_table();
            node->edgelist_hashmap.reserve(node->edgelist_hashmap.size() * 2);
            node->vertex_table.clear();
        }

        // Report
        size_t total_nodes_in_edgemaps = 0;
        for (auto *node : all_nodes)
            total_nodes_in_edgemaps += node->edgelist_hashmap.size();
        cout << "  Total entries across edgemaps: " << total_nodes_in_edgemaps << endl;
    }

    // ---- load and wire root graph ----
    AttrTreeNode *rootnode = is_numerical[0]
        ? trees[0]->tree->root
        : trees[0]->hash->root_collection;
    if (use_vamana_root) {
        // ---- Vamana flat graph (no upper layer) ----
        string vamana_path = index_prefix + "root_vamana.txt";
        string vamana_mapping_path = pick_existing_path({
            index_prefix + "initial_ids_vamana.bin",
            index_prefix + "initial_ids.bin",
        });
        cout << "\nLoading Vamana root graph from " << vamana_path << "..." << endl;
        cout << "Loading Vamana ID mapping from " << vamana_mapping_path << "..." << endl;

        vector<IdxType> vamana_initial_ids = load_id_mapping(vamana_mapping_path);
        if (vamana_initial_ids != initial_ids) {
            cerr << "Vamana root ID mapping does not match the PathFinder/tree mapping." << endl;
            cerr << "These artifacts must be built on the same initial subset." << endl;
            return 1;
        }

        auto root_g = make_shared<Graph>(n_initial);
        root_g->load(vamana_path);

        // Remap edge destinations from packed to global
        for (IdxType i = 0; i < (IdxType)n_initial; i++)
            for (IdxType &nbr : root_g->neighbors[i])
                nbr = initial_ids.at(nbr);

        // Place graph into root node and build edgemap
        rootnode->g = root_g;
        rootnode->size = n_initial;
        rootnode->vertex_table.clear();
        rootnode->vertex_table.reserve(n_initial);
        for (int i = 0; i < n_initial; i++)
            rootnode->vertex_table[initial_ids[i]] = i;
        rootnode->map_global_id_directly_to_edge_table();
        rootnode->edgelist_hashmap.reserve(rootnode->edgelist_hashmap.size() * 2);
        rootnode->vertex_table.clear();

        cout << "  Root edgemap entries: " << rootnode->edgelist_hashmap.size() << endl;

        // Wire delegate_root for all other attributes
        for (int a = 1; a < na; a++) {
            AttrTreeNode *other_root = is_numerical[a]
                ? trees[a]->tree->root
                : trees[a]->hash->root_collection;
            other_root->delegate_root = rootnode;
        }
        cout << "  Wired delegate_root for " << (na - 1) << " other attribute(s)" << endl;
    } else {
        // ---- HNSW root graph (with upper layer) ----
        string root_path = index_prefix + "root.txt";
        string root_upper_path = index_prefix + "root_upper.txt";
        cout << "\nLoading HNSW root graph from " << root_path << "..." << endl;

        auto root_g = make_shared<Graph>(n_initial);
        root_g->load(root_path);

        // Remap edge destinations from packed to global
        for (IdxType i = 0; i < (IdxType)n_initial; i++)
            for (IdxType &nbr : root_g->neighbors[i])
                nbr = initial_ids.at(nbr);

        // Place graph into root node and build edgemap
        rootnode->g = root_g;
        rootnode->size = n_initial;
        rootnode->vertex_table.clear();
        rootnode->vertex_table.reserve(n_initial);
        for (int i = 0; i < n_initial; i++)
            rootnode->vertex_table[initial_ids[i]] = i;
        rootnode->map_global_id_directly_to_edge_table();
        rootnode->edgelist_hashmap.reserve(rootnode->edgelist_hashmap.size() * 2);
        rootnode->vertex_table.clear();

        cout << "  Root edgemap entries: " << rootnode->edgelist_hashmap.size() << endl;

        // Load HNSW upper layer and remap IDs from packed to global
        rootnode->optional_HNSW_upper = make_shared<HNSWUpperLayer>();
        rootnode->optional_HNSW_upper->load(root_upper_path);

        auto &upper = *rootnode->optional_HNSW_upper;
        upper.enterpoint_node_ = initial_ids.at(upper.enterpoint_node_);
        // Expand element_levels_ from packed (0..n_initial-1) to global (0..nb-1)
        vector<int> old_levels = std::move(upper.element_levels_);
        upper.element_levels_.assign(nb, 0);
        for (int i = 0; i < n_initial; i++)
            upper.element_levels_[initial_ids[i]] = old_levels[i];
        // Remap upper-layer edgelist keys and neighbor values
        unordered_map<IdxType, vector<IdxType>> new_edgelist;
        for (auto &[node, neighbors] : upper.edgelist) {
            auto &dst = new_edgelist[initial_ids.at(node)];
            dst.resize(neighbors.size());
            for (size_t j = 0; j < neighbors.size(); j++)
                dst[j] = initial_ids.at(neighbors[j]);
        }
        upper.edgelist = std::move(new_edgelist);

        cout << "  Upper layer: entry=" << upper.enterpoint_node_
             << " maxlevel=" << upper.maxlevel_ << endl;

        // Wire delegate_root for all other attributes
        for (int a = 1; a < na; a++) {
            AttrTreeNode *other_root = is_numerical[a]
                ? trees[a]->tree->root
                : trees[a]->hash->root_collection;
            other_root->delegate_root = rootnode;
        }
        cout << "  Wired delegate_root for " << (na - 1) << " other attribute(s)" << endl;
    }

    // ---- cache entry points per node (before parallel updates) ----
    // Map: attr_id → node_id → entry_points.  100 per non-empty node.
    unordered_map<int, unordered_map<int, vector<IdxType>>> node_entries;
    for (int a = 0; a < na; a++) {
        auto all_nodes = get_all_nodes(trees[a]);
        for (auto *node : all_nodes) {
            if (node->edgelist_hashmap.size() == 0) continue;
            node_entries[a][node->node_id] = sample_entry_points(node->edgelist_hashmap, 100);
        }
    }
    // Entry points for the root graph (used for root-level updates)
    vector<IdxType> root_entry_points = sample_entry_points(rootnode->edgelist_hashmap, 100);

    TableStorage table_storage;
    init_table_storage_schema(base_storage, na, table_storage);
    for (IdxType gid : initial_ids) {
        const float *attr_ptr = full_attr.data() + (size_t)gid * na;
        const float *vec_ptr = reinterpret_cast<const float *>(base_storage->get_vector(gid));
        if (!table_storage.insert(gid, vec_ptr, attr_ptr)) {
            throw runtime_error("Failed to seed TableStorage with PathFinder initial IDs");
        }
    }

    // ---- prepare insert/delete lists ----
    vector<bool> in_initial(nb, false);
    for (IdxType id : initial_ids) in_initial[id] = true;

    vector<IdxType> insert_ids;
    insert_ids.reserve(nb - n_initial);
    for (int i = 0; i < nb; i++)
        if (!in_initial[i]) insert_ids.push_back(i);

    mt19937 rng(42);
    shuffle(insert_ids.begin(), insert_ids.end(), rng);

    int total_updates = mixed_ops ? int64_t(mixed_ops)*update_percent/200 : n_batches * batch_size;
    if ((int)insert_ids.size() < total_updates) {
        cerr << "Not enough insert candidates: have " << insert_ids.size()
             << ", need " << total_updates << endl;
        return 1;
    }
    insert_ids.resize(total_updates);

    // Delete targets: random subset of initial IDs
    vector<IdxType> delete_ids = initial_ids;
    if (mixed_ops) rng.seed(42);
    shuffle(delete_ids.begin(), delete_ids.end(), rng);
    if ((int)delete_ids.size() < total_updates) {
        cerr << "Not enough delete candidates: have " << delete_ids.size()
             << ", need " << total_updates << endl;
        return 1;
    }
    delete_ids.resize(total_updates);

    cout << "\n=== Update plan ===" << endl;
    cout << "  Batches: " << n_batches << " x (" << batch_size << " inserts + "
         << batch_size << " deletes)" << endl;
    cout << "  Total inserts: " << total_updates << endl;
    cout << "  Total deletes: " << total_updates << endl;

    // Count subgraphs touched per point (for reporting)
    {
        int sample_id = insert_ids[0];
        int touched = 0;
        for (int a = 0; a < na; a++) {
            float val = full_attr[(size_t)sample_id * na + a];
            auto nodes = find_containing_nodes(trees[a], val);
            touched += nodes.size();
        }
        int total_subgraphs = 0;
        for (int a = 0; a < na; a++) {
            auto all_nodes = get_all_nodes(trees[a]);
            for (auto *node : all_nodes)
                if (node->depth > 0) total_subgraphs++;
        }
        cout << "  Subgraphs per point: ~" << touched << " of " << total_subgraphs << " total" << endl;
    }

    // ---- search caches ----
    SearchCacheList insert_cache_list(num_threads + 5, nb, ef_con);
    // search_cache_list will be created per ef_search value during eval sweep
    int ef_search = mixed_ops ? mixed_ef : 200;
    SearchCacheList search_cache_list(num_threads + 5, nb, ef_search);

    // ---- query planner ----
    PathFinderPlanner planner(trees);

    // ---- build context ----
    TestContext ctx;
    ctx.base_storage = base_storage;
    ctx.query_storage = query_storage;
    ctx.distance_handler = online_distance_handler;
    ctx.table_storage = &table_storage;
    ctx.nb = nb; ctx.nq = nq; ctx.dim = dim; ctx.na = na;
    ctx.K = K; ctx.ef_search = ef_search; ctx.ef_con = ef_con;
    ctx.num_threads = num_threads;
    ctx.n_batches = n_batches; ctx.batch_size = batch_size;
    ctx.total_updates = total_updates;
    ctx.full_attr = &full_attr;
    ctx.query_types = &query_types;
    ctx.is_numerical = &is_numerical;
    ctx.trees = &trees;
    ctx.rootnode = rootnode;
    ctx.initial_ids = &initial_ids;
    ctx.n_initial = n_initial;
    ctx.node_entries = &node_entries;
    ctx.root_entry_points = &root_entry_points;
    ctx.insert_ids = &insert_ids;
    ctx.delete_ids = &delete_ids;
    ctx.insert_cache_list = &insert_cache_list;
    ctx.search_cache_list = &search_cache_list;
    ctx.planner = &planner;
    ctx.get_all_nodes = get_all_nodes;

    // =====================================================================
    //  Run phases
    // =====================================================================
    if (mixed_ops) {
        vector<vector<QueryType>> sets;
        int per_set=1000;
        for(const auto &item:query_tsv_files){
            auto parsed=read_queries_from_tsv(item.second);
            sets.push_back(std::move(parsed.first));
            per_set=min(per_set,min(nq,int(sets.back().size())));
        }
        unordered_map<AttrTreeNode*,vector<IdxType>> seeds;
        seeds[rootnode]=root_entry_points;
        for(int a=0;a<na;++a)for(auto *node:get_all_nodes(trees[a])) {
            if(node->delegate_root)continue;
            seeds[node]=sample_entry_points(node->edgelist_hashmap,100);
        }
        auto query=[&](int qid){
            const auto &qt=sets[qid%sets.size()][qid/sets.size()];
            auto pred=[&](IdxType id){const float *a=get_live_attr_or_null(table_storage,id);return a && qt.check(a);};
            vector<pair<IdxType,float>> results;
            pathfinder_filtered_search(planner,query_storage->get_vector(qid/sets.size()),table_storage,
                online_distance_handler,dim,pred,qt,search_cache_list,K,mixed_ef,results,&seeds);
        };
        auto update=[&](int i,bool inserting){
            auto id=inserting?insert_ids[i]:delete_ids[i];
            const char *vec=base_storage->get_vector(id);
            const float *attr=full_attr.data()+size_t(id)*na;
            if(inserting && !table_storage.insert(id,reinterpret_cast<const float*>(vec),attr))
                throw runtime_error("Mixed PathFinder insertion failed");
            auto apply=[&](AttrTreeNode *node,const vector<IdxType> &entries){
                if(inserting)insert_point(node->edgelist_hashmap,id,vec,table_storage,insert_cache_list,entries);
                else delete_point(node->edgelist_hashmap,id,vec,table_storage,insert_cache_list,entries);
            };
            apply(rootnode,root_entry_points);
            for(int a=0;a<na;++a)for(auto *node:find_containing_nodes(trees[a],attr[a])) {
                const auto &entries=node_entries.at(a);
                auto it=entries.find(node->node_id);
                if(it!=entries.end())apply(node,it->second);
            }
            if(!inserting && !table_storage.remove(id))throw runtime_error("Mixed PathFinder deletion failed");
        };
        return run_mixed_workload("PathFinder",mixed_ops,update_percent,num_threads,mixed_ef,
            per_set*sets.size(),query,[&](int i){update(i,true);},[&](int i){update(i,false);});
    }
    if (calibrate_only) {
        auto search = [&](const char *query, const QueryType &qt, int ef, SearchCacheList &cache,
                          vector<pair<IdxType, float>> &results) {
            auto pred = [&](IdxType id) {
                const float *a = get_live_attr_or_null(table_storage, id);
                return a && qt.check(a);
            };
            pathfinder_filtered_search(planner, query, table_storage, online_distance_handler, dim,
                                       pred, qt, cache, K, ef, results);
        };
        return calibrate_mixed_queries("PathFinder", dataset, query_tsv_files, base_storage,
            query_storage, online_distance_handler, full_attr, na, initial_ids, K,
            num_threads, calibration_max_ef, search, calibration_target);
    }
    EvalStats pre_stats = evaluate_recall(ctx, "Phase 0: Pre-update filtered evaluation", initial_ids);
    UpdateStats update_stats = phase1_batched_updates(ctx);

    // ---- update RESULT lines (inserts / deletes) ----
    cout << "\n=== Update Summary ===" << endl;
    cout << "  Dataset: " << dataset << endl;
    cout << "  Initial: " << n_initial << " / " << nb << " ("
         << fixed << setprecision(1) << 100.0 * n_initial / nb << "%)" << endl;
    cout << "  Root graph: " << (use_vamana_root ? "Vamana" : "HNSW") << endl;
    cout << "  ef_con=" << ef_con << ", K=" << K << endl;
    cout << "  Batches: " << n_batches << " x " << batch_size << " ins + " << batch_size << " del" << endl;
    cout << "RESULT_INSERT"
         << " method=PathFinder"
         << " dataset=" << dataset
         << " n_batches=" << n_batches
         << " batch_size=" << batch_size
         << " ins_qps=" << setprecision(0) << update_stats.avg_ins_qps
         << " ins_dc=" << setprecision(1) << update_stats.avg_ins_dc
         << endl;
    cout << "RESULT_DELETE"
         << " method=PathFinder"
         << " dataset=" << dataset
         << " n_batches=" << n_batches
         << " batch_size=" << batch_size
         << " del_qps=" << setprecision(0) << update_stats.avg_del_qps
         << " del_dc=" << setprecision(1) << update_stats.avg_del_dc
         << endl;

    // =====================================================================
    //  Phase 2: Post-update query sweep (3 selectivities x multiple ef_search)
    // =====================================================================
    vector<int> ef_search_values = {10, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000};

    vector<IdxType> live_ids = live_ids_from_table(*ctx.table_storage);
    cout << "\n=== Phase 2: Post-update query sweep ===" << endl;
    cout << "  Live points: " << live_ids.size() << " / " << nb << endl;

    for (const auto &[sel_label, tsv_path] : query_tsv_files) {
        cout << "\n--- Selectivity: " << sel_label << " (" << tsv_path << ") ---" << endl;

        auto [sel_query_types, sel_header] = read_queries_from_tsv(tsv_path);
        int nq_tsv = sel_query_types.size();
        int nq_eval = min(nq, nq_tsv);

        auto predicate_2arg = [&](IdxType pid, int q) -> bool {
            const float *attr = get_live_attr_or_null(*ctx.table_storage, pid);
            return attr != nullptr && sel_query_types[q].check(attr);
        };

        cout << "  Computing filtered ground truth for " << nq_eval << " queries..." << endl;
        vector<vector<pair<IdxType, float>>> gt;
        {
            int gt_threads = omp_get_num_procs();
            omp_set_num_threads(gt_threads);
            compute_filtered_ground_truth(
                *ctx.table_storage, ctx.query_storage, ctx.distance_handler, ctx.dim,
                live_ids, predicate_2arg, nq_eval, K, gt);
            omp_set_num_threads(num_threads);
        }

        for (int ef_s : ef_search_values) {
            SearchCacheList sweep_cache_list(num_threads + 5, nb, ef_s);

            int total_found = 0;
            long total_dist_comps = 0;
            auto t0 = chrono::high_resolution_clock::now();

            #pragma omp parallel for schedule(dynamic, 1) reduction(+: total_found, total_dist_comps)
            for (int q = 0; q < nq_eval; q++) {
                auto query_vec = ctx.query_storage->get_vector(q);
                auto pred = [&, q](IdxType pid) -> bool {
                    const float *attr = get_live_attr_or_null(*ctx.table_storage, pid);
                    return attr != nullptr && sel_query_types[q].check(attr);
                };
                vector<pair<IdxType, float>> results;
                long before = online_dist_comps;
                pathfinder_filtered_search(
                    *ctx.planner, query_vec, *ctx.table_storage,
                    ctx.distance_handler, ctx.dim,
                    pred, sel_query_types[q],
                    sweep_cache_list, K, ef_s, results);

                unordered_set<IdxType> gt_set;
                for (auto &[id, d] : gt[q]) gt_set.insert(id);
                int found = 0;
                for (auto &[id, d] : results)
                    if (gt_set.count(id)) found++;
                total_found += found;
                total_dist_comps += online_dist_comps - before;
            }

            auto t1 = chrono::high_resolution_clock::now();
            double search_ms = chrono::duration_cast<chrono::microseconds>(t1 - t0).count() / 1e3;
            double recall = 100.0 * total_found / (nq_eval * K);
            double qps = nq_eval / (search_ms / 1e3);
            double avg_dc = (double)total_dist_comps / nq_eval;

            cout << "  ef_search=" << ef_s
                 << " recall=" << fixed << setprecision(2) << recall
                 << " qps=" << setprecision(0) << qps
                 << " avg_dc=" << setprecision(1) << avg_dc << endl;

            cout << "RESULT_QUERY"
                 << " method=PathFinder"
                 << " dataset=" << dataset
                 << " selectivity=" << sel_label
                 << " K=" << K
                 << " ef_search=" << ef_s
                 << " recall=" << fixed << setprecision(2) << recall
                 << " qps=" << setprecision(0) << qps
                 << " avg_dc=" << setprecision(1) << avg_dc
                 << endl;
        }
    }

    cout << "\nDone." << endl;
    return 0;
}
