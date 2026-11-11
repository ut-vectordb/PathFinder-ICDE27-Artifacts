#ifndef ONLINE_UPDATE_BASELINE_COMMON_H
#define ONLINE_UPDATE_BASELINE_COMMON_H

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "hnsw/hnsw.h"
#include "querytype/querytype.h"
#include "run_search_online.h"
#include "storage.h"
#include "tablestorage/tablestorage.h"

namespace ANNS {

enum class BaselineSearchMode {
    FlatOOR,
    HNSWOOR,
    NaviX,
    FAVOR,   // FAVOR search on the updated stock-HNSW graph (baselines/search.h)
};

struct BaselineDatasetConfig {
    std::string dataset;
    std::string base_bin_file;
    std::string query_bin_file;
    std::string attr_bin_file;
    std::string query_tsv_file;  // used for schema detection only
    std::vector<std::pair<std::string, std::string>> query_tsv_files;  // {label, path}
    std::string index_prefix_pathfinder;
    IdxType M;
};

struct BaselineArtifacts {
    std::string graph_path;
    std::string mapping_path;
    std::string upper_path;
};

struct BaselineIndex {
    std::shared_ptr<IStorage> base_storage;
    std::shared_ptr<IStorage> query_storage;
    std::shared_ptr<DistanceHandler> distance_handler;
    std::shared_ptr<HNSWUpperLayer> upper_layer;
    ConcurrentEdgeMap edgemap;
    TableStorage table_storage;
    std::vector<float> full_attr;
    std::vector<QueryType> query_types;
    std::vector<IdxType> initial_ids;
    int nb = 0;
    int nq = 0;
    int dim = 0;
    int na = 0;
    IdxType M = 0;
};

struct UpdateStats {
    double avg_ins_ms = 0.0;
    double avg_del_ms = 0.0;
    double avg_ins_qps = 0.0;
    double avg_del_qps = 0.0;
    double avg_update_qps = 0.0;
    double avg_ins_dc = 0.0;
    double avg_del_dc = 0.0;
};

struct EvalStats {
    double recall = 0.0;
    double qps = 0.0;
    double avg_dist_comps = 0.0;
    double wall_ms = 0.0;
    size_t live_points = 0;
};

struct BaselineRunConfig {
    std::string method_name;
    std::string artifact_method;
    BaselineSearchMode search_mode = BaselineSearchMode::FlatOOR;
    std::string dataset = "sift3";
    uint32_t num_threads = 16;
    int ef_con = 200;
    int K = 10;
    int n_batches = 100;
    int navix_n_entry = 15;
    uint32_t update_threads = 0;    // 0: same as num_threads. >0: threads for the update batches only
                                    // (queries/sweep stay on num_threads)
    int mixed_ops = 0;
    int update_percent = 1;
    int mixed_ef = 50;
    int stress_seconds = 0; // FAVOR concurrent q/i/d smoke test; no recall evaluation
    int stress_ef = 50;
    double stress_delta = -1; // reuse initial delta_d from the existing calibration
    bool calibrate_only = false;
    int calibration_max_ef = 3000;
    int calibration_target = 0;
    bool ef_descending = false;     // post-update sweep order (existing methods: ascending)
    // FAVOR delta_d under updates (upstream: delta_d += term/max_elements per insert,
    // never decremented on delete). capacity = nb (all ids ever inserted; no slot reuse).
    // "capacity": initial = calibrated mean * n_initial/nb (what upstream holds after
    // building n_initial points with max_elements = nb); "full": initial = calibrated mean.
    std::string favor_initial = "capacity";
};

// FAVOR's delta_d bookkeeping during updates (guarded like upstream's delta_mutex).
struct FavorUpdateState {
    double delta_d = 0.0;
    double capacity = 0.0;
    int ef_con = 200;
    long terms_added = 0;
    std::mutex mu;
};

BaselineDatasetConfig get_baseline_dataset_config(const std::string &dataset);
BaselineArtifacts get_baseline_artifacts(const BaselineDatasetConfig &cfg,
                                         const std::string &method);

std::shared_ptr<BaselineIndex> load_baseline_index(const BaselineDatasetConfig &cfg,
                                                   const std::string &method);

std::vector<IdxType> sample_entry_points(ConcurrentEdgeMap &edgemap, int n);

void compute_filtered_ground_truth(
        std::shared_ptr<IStorage> base_storage,
        std::shared_ptr<IStorage> query_storage,
        std::shared_ptr<DistanceHandler> distance_handler,
        int dim,
        const std::vector<IdxType> &live_ids,
        std::function<bool(IdxType, int)> predicate,
        int nq, int K,
        std::vector<std::vector<std::pair<IdxType, float>>> &gt_out);

UpdateStats run_batched_updates(BaselineIndex &index,
                                int n_batches,
                                int batch_size,
                                int ef_con,
                                uint32_t num_threads,
                                std::vector<IdxType> &insert_ids,
                                std::vector<IdxType> &delete_ids,
                                std::vector<IdxType> &entry_points,
                                FavorUpdateState *favor = nullptr);

EvalStats evaluate_filtered_queries(BaselineIndex &index,
                                    BaselineSearchMode mode,
                                    const std::string &title,
                                    const std::vector<IdxType> &live_ids,
                                    int K,
                                    int ef_search,
                                    uint32_t num_threads,
                                    int navix_n_entry);

std::vector<IdxType> make_insert_ids(const BaselineIndex &index,
                                     int total_updates,
                                     int seed);

std::vector<IdxType> make_delete_ids(const BaselineIndex &index,
                                     int total_updates,
                                     int seed);

int run_baseline_experiment(const BaselineRunConfig &cfg);

} // namespace ANNS

#endif
