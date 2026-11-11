#include "run_search_online.h"

#include <algorithm>
#include <random>
#include <thread>
#include <atomic>
#include <queue>
#include <unordered_set>
#include <fstream>
#include <omp.h>

namespace ANNS {

using namespace std;

// ---- Globals ----
std::shared_ptr<DistanceHandler> online_distance_handler;
int online_nb = 0, online_dim = 0;
IdxType online_M = 0;
float online_vamana_alpha = 1.2f;
thread_local long online_dist_comps = 0;

namespace {

char *lookup_vec_or_null(TableStorage &table_storage, IdxType id) {
    void *row = table_storage.find(id);
    return row == nullptr ? nullptr : table_storage.get_vector(row);
}

} // namespace

// =====================================================================
//  Beam search on ConcurrentEdgeMap  (mirrors Hnsw::iterate_to_fixed_point)
// =====================================================================
void beam_search_edgemap(
        ConcurrentEdgeMap &edgemap,
        const char *query,
        TableStorage &table_storage,
        SearchCacheList &cache_list,
        vector<Candidate> &expanded_out,
        vector<pair<IdxType,float>> &topk_out,
        int K,
        const vector<IdxType> &entry_points,
        IdxType self_id)
{
    auto cache = cache_list.get_free_cache();
    auto &sq = cache->search_queue;
    auto &vs = cache->visited_set;
    expanded_out.clear();
    topk_out.clear();
    sq.clear();
    vs.clear();

    thread_local mt19937 rng(hash<thread::id>{}(this_thread::get_id()));
    uniform_int_distribution<int> dist_entry(0, (int)entry_points.size() - 1);
    int found = 0;
    int attempts = 0;
    const int max_attempts = (int)entry_points.size() * 3;
    ConcurrentEdgeMap::InPlaceReadGuard rguard;
    while (found < 5 && attempts < max_attempts) {
        ++attempts;
        IdxType entry = entry_points[dist_entry(rng)];
        if (vs.check(entry)) continue;
        if (!edgemap.try_read_lock(entry, rguard)) continue;
        rguard.release();
        char *entry_vec = lookup_vec_or_null(table_storage, entry);
        if (entry_vec == nullptr) continue;
        vs.set(entry);
        float d = online_distance_handler->compute(query, entry_vec, online_dim);
        ++online_dist_comps;
        sq.insert(entry, d);
        ++found;
    }

    while (sq.has_unexpanded_node()) {
        const Candidate cur = sq.get_closest_unexpanded();

        vector<IdxType> edges;
        if (edgemap.try_read_lock(cur.id, rguard)) {
            edges = *rguard.edges;
            rguard.release();
        } else {
            continue;
        }

        if (cur.id != self_id)
            expanded_out.push_back(cur);

        for (IdxType nbr : edges) {
            if (vs.check(nbr)) continue;
            char *nbr_vec = lookup_vec_or_null(table_storage, nbr);
            if (nbr_vec == nullptr) continue;
            vs.set(nbr);
            float d = online_distance_handler->compute(query, nbr_vec, online_dim);
            ++online_dist_comps;
            sq.insert(nbr, d);
        }
    }

    vector<pair<float, IdxType>> heap;
    for (auto &c : expanded_out) {
        heap.emplace_back(c.distance, c.id);
        push_heap(heap.begin(), heap.end());
        if ((int)heap.size() > K) {
            pop_heap(heap.begin(), heap.end());
            heap.pop_back();
        }
    }
    topk_out.reserve(heap.size());
    for (auto &[d, id] : heap)
        topk_out.emplace_back(id, d);
    sort(topk_out.begin(), topk_out.end(), [](auto &a, auto &b) { return a.second < b.second; });

    cache_list.release_cache(cache);
}

// =====================================================================
//  Vamana prune  (mirrors Hnsw::vamana_prune_neighbors)
// =====================================================================
void vamana_prune(
        IdxType id,
        vector<Candidate> &candidates,
        vector<IdxType> &pruned,
        int maxdeg,
        std::function<char*(IdxType)> get_vec)
{
    pruned.clear();
    pruned.reserve(maxdeg);

    sort(candidates.begin(), candidates.end(),
         [](const Candidate &a, const Candidate &b) { return a.distance < b.distance; });

    IdxType max_cand = min((IdxType)candidates.size(), (IdxType)750);

    vector<float> occlude_factor(max_cand, 0.0f);

    float cur_alpha = 1.0f;
    while (cur_alpha <= online_vamana_alpha && (int)pruned.size() < maxdeg) {
        for (IdxType i = 0; i < max_cand && (int)pruned.size() < maxdeg; i++) {
            if (occlude_factor[i] > cur_alpha)
                continue;
            char *vec_i = get_vec(candidates[i].id);
            if (vec_i == nullptr)
                continue;
            occlude_factor[i] = numeric_limits<float>::max();
            if (candidates[i].id != id)
                pruned.push_back(candidates[i].id);

            for (IdxType j = i + 1; j < max_cand; j++) {
                if (occlude_factor[j] > online_vamana_alpha)
                    continue;
                char *vec_j = get_vec(candidates[j].id);
                if (vec_j == nullptr)
                    continue;
                float dist_ij = online_distance_handler->compute(
                    vec_i, vec_j, online_dim);
                ++online_dist_comps;
                occlude_factor[j] = (dist_ij == 0)
                    ? numeric_limits<float>::max()
                    : max(occlude_factor[j], candidates[j].distance / dist_ij);
            }
        }
        cur_alpha *= 1.2f;
    }
}

void insert_point(
        ConcurrentEdgeMap &edgemap,
        IdxType new_id,
        const char *vec,
        TableStorage &table_storage,
        SearchCacheList &cache_list,
        const vector<IdxType> &entry_points,
        vector<float> *pool_dists_out)
{
    auto get_vec = [&](IdxType id) -> char * {
        return lookup_vec_or_null(table_storage, id);
    };
    vector<Candidate> expanded;
    vector<pair<IdxType,float>> topk_unused;
    beam_search_edgemap(edgemap, vec, table_storage, cache_list, expanded, topk_unused, 0, entry_points, new_id);
    if (pool_dists_out) {
        pool_dists_out->clear();
        pool_dists_out->reserve(expanded.size());
        for (const auto &c : expanded) pool_dists_out->push_back(c.distance);
    }

    vector<IdxType> new_neighbors;
    vamana_prune(new_id, expanded, new_neighbors, online_M, get_vec);

    edgemap.insert_empty(new_id);
    {
        auto guard = edgemap.write_lock(new_id);
        *guard.edges = std::move(new_neighbors);
    }

    vector<IdxType> fwd_edges;
    { auto guard = edgemap.read_lock(new_id); fwd_edges = *guard.edges; }

    ConcurrentEdgeMap::InPlaceWriteGuard wguard;
    for (IdxType dst : fwd_edges) {
        bool need_prune = false;
        vector<Candidate> candidates;

        if (edgemap.try_write_lock(dst, wguard)) {
            auto &dst_edges = *wguard.edges;
            if (find(dst_edges.begin(), dst_edges.end(), new_id) == dst_edges.end()) {
                if ((int)dst_edges.size() < online_M) {
                    dst_edges.push_back(new_id);
                } else {
                    candidates.reserve(dst_edges.size() + 1);
                    for (IdxType nbr : dst_edges)
                        candidates.emplace_back(nbr, 0);
                    candidates.emplace_back(new_id, 0);
                    need_prune = true;
                }
            }
            wguard.release();
        } else {
            continue;
        }

        if (need_prune) {
            char *dst_vec = lookup_vec_or_null(table_storage, dst);
            if (dst_vec == nullptr) continue;
            for (auto &c : candidates) {
                char *cand_vec = lookup_vec_or_null(table_storage, c.id);
                if (cand_vec == nullptr) continue;
                c.distance = online_distance_handler->compute(dst_vec, cand_vec, online_dim);
                ++online_dist_comps;
            }

            vector<IdxType> pruned;
            vamana_prune(dst, candidates, pruned, online_M, get_vec);

            if (edgemap.try_write_lock(dst, wguard)) {
                *wguard.edges = std::move(pruned);
                wguard.release();
            }
        }
    }
}

// =====================================================================
//  Delete one point in-place (IP-DiskANN Algorithm 5)
// =====================================================================
void delete_point(
        ConcurrentEdgeMap &edgemap,
        IdxType del_id,
        const char *vec,
        TableStorage &table_storage,
        SearchCacheList &cache_list,
        const vector<IdxType> &entry_points,
        int c_replace)
{
    constexpr int k_candidates = 50;
    auto get_vec = [&](IdxType id) -> char * {
        return lookup_vec_or_null(table_storage, id);
    };

    vector<Candidate> expanded;
    vector<pair<IdxType,float>> topk_unused;
    beam_search_edgemap(edgemap, vec, table_storage, cache_list, expanded, topk_unused, 0, entry_points, del_id);

    ConcurrentEdgeMap::InPlaceReadGuard rguard;
    vector<IdxType> approx_in;
    for (auto &c : expanded) {
        if (!edgemap.try_read_lock(c.id, rguard)) continue;
        if (find(rguard.edges->begin(), rguard.edges->end(), del_id) != rguard.edges->end())
            approx_in.push_back(c.id);
        rguard.release();
    }

    vector<IdxType> out_nbrs;
    if (edgemap.try_read_lock(del_id, rguard)) {
        out_nbrs = *rguard.edges;
        rguard.release();
    }

    sort(expanded.begin(), expanded.end(),
         [](const Candidate &a, const Candidate &b) { return a.distance < b.distance; });
    int n_cand = min((int)expanded.size(), k_candidates);
    vector<Candidate> candidates(expanded.begin(), expanded.begin() + n_cand);

    ConcurrentEdgeMap::InPlaceWriteGuard wguard;
    for (IdxType z : approx_in) {
        char *z_vec = lookup_vec_or_null(table_storage, z);
        if (z_vec == nullptr) continue;
        vector<pair<float, IdxType>> dists;
        for (auto &cand : candidates) {
            if (cand.id == del_id || cand.id == z) continue;
            char *cand_vec = lookup_vec_or_null(table_storage, cand.id);
            if (cand_vec == nullptr) continue;
            float d = online_distance_handler->compute(z_vec, cand_vec, online_dim);
            ++online_dist_comps;
            dists.emplace_back(d, cand.id);
        }
        sort(dists.begin(), dists.end());

        bool need_prune = false;
        vector<Candidate> prune_cands;

        if (edgemap.try_write_lock(z, wguard)) {
            auto &edges = *wguard.edges;
            edges.erase(remove(edges.begin(), edges.end(), del_id), edges.end());
            if ((int)edges.size() + c_replace <= online_M) {
                int added = 0;
                for (auto &[d, id] : dists) {
                    if (added >= c_replace) break;
                    if (find(edges.begin(), edges.end(), id) == edges.end()) {
                        edges.push_back(id);
                        added++;
                    }
                }
            } else {
                prune_cands.reserve(edges.size() + dists.size());
                for (IdxType nbr : edges)
                    prune_cands.emplace_back(nbr, 0);
                for (auto &[d, id] : dists)
                    if (find(edges.begin(), edges.end(), id) == edges.end())
                        prune_cands.emplace_back(id, 0);
                need_prune = true;
            }
            wguard.release();
        } else {
            continue;
        }

        if (need_prune) {
            for (auto &c : prune_cands) {
                char *cand_vec = lookup_vec_or_null(table_storage, c.id);
                if (cand_vec == nullptr) continue;
                c.distance = online_distance_handler->compute(z_vec, cand_vec, online_dim);
                ++online_dist_comps;
            }

            vector<IdxType> pruned;
            vamana_prune(z, prune_cands, pruned, online_M, get_vec);

            if (edgemap.try_write_lock(z, wguard)) {
                *wguard.edges = std::move(pruned);
                wguard.release();
            }
        }
    }

    struct OutNbrDists {
        IdxType w;
        vector<pair<float, IdxType>> dists;
    };
    vector<OutNbrDists> out_dists(out_nbrs.size());
    for (int i = 0; i < (int)out_nbrs.size(); i++) {
        out_dists[i].w = out_nbrs[i];
        char *w_vec = lookup_vec_or_null(table_storage, out_nbrs[i]);
        if (w_vec == nullptr) continue;
        for (auto &cand : candidates) {
            if (cand.id == del_id || cand.id == out_nbrs[i]) continue;
            char *cand_vec = lookup_vec_or_null(table_storage, cand.id);
            if (cand_vec == nullptr) continue;
            float d = online_distance_handler->compute(w_vec, cand_vec, online_dim);
            ++online_dist_comps;
            out_dists[i].dists.emplace_back(d, cand.id);
        }
        sort(out_dists[i].dists.begin(), out_dists[i].dists.end());
    }

    for (auto &od : out_dists) {
        int added = 0;
        for (auto &[d, y] : od.dists) {
            if (added >= c_replace) break;

            bool need_prune = false;
            vector<Candidate> prune_cands;

            if (edgemap.try_write_lock(y, wguard)) {
                auto &edges = *wguard.edges;
                if (find(edges.begin(), edges.end(), od.w) == edges.end()) {
                    if ((int)edges.size() < online_M) {
                        edges.push_back(od.w);
                        added++;
                    } else {
                        prune_cands.reserve(edges.size() + 1);
                        for (IdxType nbr : edges)
                            prune_cands.emplace_back(nbr, 0);
                        prune_cands.emplace_back(od.w, 0);
                        need_prune = true;
                    }
                }
                wguard.release();
            } else {
                continue;
            }

            if (need_prune) {
                char *y_vec = lookup_vec_or_null(table_storage, y);
                if (y_vec == nullptr) continue;
                for (auto &c : prune_cands) {
                    char *cand_vec = lookup_vec_or_null(table_storage, c.id);
                    if (cand_vec == nullptr) continue;
                    c.distance = online_distance_handler->compute(y_vec, cand_vec, online_dim);
                    ++online_dist_comps;
                }

                vector<IdxType> pruned;
                vamana_prune(y, prune_cands, pruned, online_M, get_vec);

                if (edgemap.try_write_lock(y, wguard)) {
                    *wguard.edges = std::move(pruned);
                    wguard.release();
                }
                added++;
            }
        }
    }
    edgemap.erase(del_id);
}

// =====================================================================
//  Brute-force ground truth (unfiltered, K=10)
// =====================================================================
void compute_ground_truth(
        shared_ptr<IStorage> base_storage,
        shared_ptr<IStorage> query_storage,
        int n_queries,
        int K,
        vector<vector<IdxType>> &gt_out)
{
    gt_out.resize(n_queries);
    #pragma omp parallel for schedule(dynamic, 4)
    for (int q = 0; q < n_queries; q++) {
        auto query = query_storage->get_vector(q);
        priority_queue<pair<float, IdxType>> heap;
        for (int i = 0; i < online_nb; i++) {
            float d = online_distance_handler->compute(query, base_storage->get_vector(i), online_dim);
            heap.emplace(d, i);
            if ((int)heap.size() > K) heap.pop();
        }
        gt_out[q].resize(heap.size());
        int idx = heap.size() - 1;
        while (!heap.empty()) {
            gt_out[q][idx--] = heap.top().second;
            heap.pop();
        }
    }
}

// =====================================================================
//  Recall calculation
// =====================================================================
double calc_recall(
        ConcurrentEdgeMap &edgemap,
        shared_ptr<IStorage> query_storage,
        TableStorage &table_storage,
        SearchCacheList &cache_list,
        int n_queries, int K,
        const vector<vector<IdxType>> &gt,
        const vector<IdxType> &entry_points)
{
    atomic<int> total_found(0);

    #pragma omp parallel for schedule(dynamic, 4)
    for (int q = 0; q < n_queries; q++) {
        auto query = query_storage->get_vector(q);

        vector<Candidate> expanded;
        vector<pair<IdxType,float>> topk;
        beam_search_edgemap(edgemap, query, table_storage, cache_list,
                            expanded, topk, K, entry_points);

        // Count overlap with ground truth
        unordered_set<IdxType> gt_set(gt[q].begin(), gt[q].end());
        int found = 0;
        for (auto &[id, d] : topk) {
            if (gt_set.count(id)) found++;
        }
        total_found.fetch_add(found, memory_order_relaxed);
    }

    return 100.0 * total_found.load() / (n_queries * K);
}

// =====================================================================
//  Load binary ID mapping written by build_partial_index
// =====================================================================
vector<IdxType> load_id_mapping(const string &path) {
    ifstream in(path, ios::binary);
    if (!in) throw runtime_error("Cannot open " + path);
    uint32_t n;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    vector<IdxType> ids(n);
    in.read(reinterpret_cast<char*>(ids.data()), n * sizeof(IdxType));
    return ids;
}

// =====================================================================
//  Save binary ID mapping
// =====================================================================
void save_id_mapping(const string &path, const vector<IdxType> &ids) {
    ofstream out(path, ios::binary);
    if (!out) throw runtime_error("Cannot open " + path + " for writing");
    uint32_t n = ids.size();
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(ids.data()), n * sizeof(IdxType));
}

} // namespace ANNS
