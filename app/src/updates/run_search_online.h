#ifndef RUN_SEARCH_ONLINE_H
#define RUN_SEARCH_ONLINE_H

#include <vector>
#include <functional>
#include <memory>

#include "storage.h"
#include "distance.h"
#include "search_cache.h"
#include "concurrent_edge_map.h"
#include "tablestorage/tablestorage.h"

namespace ANNS {

// ---- Shared globals (set by the caller before use) ----
extern std::shared_ptr<DistanceHandler> online_distance_handler;
extern int online_nb, online_dim;
extern IdxType online_M;
extern float online_vamana_alpha;

/// Per-thread distance computation counter. Incremented at every
/// distance_handler->compute() call in beam_search_edgemap, vamana_prune,
/// insert_point, and delete_point. Callers snapshot before/after to
/// attribute costs to individual queries or update operations.
extern thread_local long online_dist_comps;

// ---- Core operations on ConcurrentEdgeMap ----

/// Beam search on ConcurrentEdgeMap (mirrors Hnsw::iterate_to_fixed_point).
/// expanded_out: all visited candidates (for use as prune input after insertion).
/// topk_out: top-K results sorted by distance.
/// entry_points: seed nodes for the search.
/// self_id: exclude this ID from expanded_out (set to the inserting point's ID).
void beam_search_edgemap(
        ConcurrentEdgeMap &edgemap,
        const char *query,
        TableStorage &table_storage,
        SearchCacheList &cache_list,
        std::vector<Candidate> &expanded_out,
        std::vector<std::pair<IdxType,float>> &topk_out,
        int K,
        const std::vector<IdxType> &entry_points,
        IdxType self_id = static_cast<IdxType>(-1));

/// Vamana prune (mirrors Hnsw::vamana_prune_neighbors).
/// Reads online_vamana_alpha, online_distance_handler, online_dim.
void vamana_prune(
        IdxType id,
        std::vector<Candidate> &candidates,
        std::vector<IdxType> &pruned,
        int maxdeg,
        std::function<char*(IdxType)> get_vec);

/// Insert one point: beam search → prune → forward edges → reverse edges.
/// Mirrors build_NSW's per-point insertion logic.
/// pool_dists_out (optional): distances of every candidate the insertion's beam
/// search visited (self excluded). FAVOR derives its per-insert delta_d term
/// from it, as upstream does from its ef_construction pool.
void insert_point(
        ConcurrentEdgeMap &edgemap,
        IdxType new_id,
        const char *vec,
        TableStorage &table_storage,
        SearchCacheList &cache_list,
        const std::vector<IdxType> &entry_points,
        std::vector<float> *pool_dists_out = nullptr);

/// Delete one point in-place (IP-DiskANN Algorithm 5):
///   beam search for the point → approximate in-neighbors from visited list
///   → repair in-neighbors (add c replacement edges) → repair out-neighbors
///   (add back-edges) → clear edges (soft delete).
/// The caller is responsible for tracking deleted IDs (e.g. for filtering
/// search results and entry points).
void delete_point(
        ConcurrentEdgeMap &edgemap,
        IdxType del_id,
        const char *vec,
        TableStorage &table_storage,
        SearchCacheList &cache_list,
        const std::vector<IdxType> &entry_points,
        int c_replace = 3);

// ---- Evaluation utilities ----

/// Brute-force ground truth (unfiltered).
void compute_ground_truth(
        std::shared_ptr<IStorage> base_storage,
        std::shared_ptr<IStorage> query_storage,
        int n_queries, int K,
        std::vector<std::vector<IdxType>> &gt_out);

/// Recall@K via beam_search_edgemap against precomputed ground truth.
double calc_recall(
        ConcurrentEdgeMap &edgemap,
        std::shared_ptr<IStorage> query_storage,
        TableStorage &table_storage,
        SearchCacheList &cache_list,
        int n_queries, int K,
        const std::vector<std::vector<IdxType>> &gt,
        const std::vector<IdxType> &entry_points);

// ---- I/O utilities ----

/// Load binary ID mapping written by build_partial_index.
/// Format: [n:uint32] [id0:uint32] [id1:uint32] ...
std::vector<IdxType> load_id_mapping(const std::string &path);

/// Save binary ID mapping.
void save_id_mapping(const std::string &path, const std::vector<IdxType> &ids);

} // namespace ANNS

#endif // RUN_SEARCH_ONLINE_H
