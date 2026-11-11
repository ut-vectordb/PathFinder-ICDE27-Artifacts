#pragma once

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <queue>
#include <map>
#include "calibration_binary_search.h"
#include <stdexcept>
#include <omp.h>
#include "querytype/querytype.h"
#include "search_cache.h"
#include "storage.h"

namespace ANNS {

// One fixed, equally weighted mixture: every query from each selectivity set.
// Only ground truth uses dense attributes/vectors; timed searches retain B-tree access.
template<class Search>
int calibrate_mixed_queries(
        const std::string &method, const std::string &dataset,
        const std::vector<std::pair<std::string, std::string>> &tsvs,
        std::shared_ptr<IStorage> base, std::shared_ptr<IStorage> queries,
        std::shared_ptr<DistanceHandler> distance, const std::vector<float> &attrs,
        int na, const std::vector<IdxType> &initial_ids, int K,
        int threads, int max_ef, Search search, int target_recall = 0) {
    using namespace std;
    if (threads < 1 || max_ef < 10 || max_ef % 10 != 0)
        throw runtime_error("calibration requires threads > 0 and max_ef a positive multiple of 10");
    if (target_recall < 0 || target_recall >= 100)
        throw runtime_error("target recall must be in (0,100), or 0 for the 90/95 pair");
    omp_set_num_threads(threads);
    vector<vector<QueryType>> types;
    for (const auto &[label, path] : tsvs) {
        auto parsed = read_queries_from_tsv(path);
        types.push_back(std::move(parsed.first));
        cout << "CALIBRATION_INPUT selectivity=" << label << " path=" << path
             << " queries=" << types.back().size() << endl;
    }
    if (types.size() != 3 || types[0].empty() || types[0].size() != types[1].size()
            || types[0].size() != types[2].size() || types[0].size() > size_t(queries->get_num_points()))
        throw runtime_error("calibration requires three equally sized, nonempty query sets");
    const int per_set = types[0].size(), count = per_set * 3;
    vector<vector<IdxType>> gt(count);
    cout << "CALIBRATION_GT method=" << method << " queries=" << count
         << " live_points=" << initial_ids.size() << " threads=" << threads << endl;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < count; ++i) {
        const int q = i / 3, group = i % 3;
        auto query = queries->get_vector(q);
        priority_queue<pair<float, IdxType>> heap;
        for (auto id : initial_ids) {
            if (!types[group][q].check(attrs.data() + size_t(id) * na)) continue;
            float d = distance->compute(query, base->get_vector(id), base->get_dim());
            if (heap.size() < size_t(K) || d < heap.top().first) {
                heap.emplace(d, id);
                if (heap.size() > size_t(K)) heap.pop();
            }
        }
        while (!heap.empty()) { gt[i].push_back(heap.top().second); heap.pop(); }
    }
    for (const auto &row : gt)
        if (row.size() != size_t(K)) throw runtime_error("fewer than K qualifying initial points in a query");
    cout << "CALIBRATION_GT_DONE method=" << method << endl;
    map<int, long> measured_hits;
    bool observed_nonmonotonic = false;
    auto measure = [&](int ef) -> long {
        auto cached = measured_hits.find(ef);
        if (cached != measured_hits.end()) return cached->second;
        SearchCacheList caches(threads + 5, base->get_num_points(), ef);
        long hits[3] = {0, 0, 0};
        auto start = chrono::steady_clock::now();
        #pragma omp parallel for schedule(dynamic, 1) reduction(+:hits[:3])
        for (int i = 0; i < count; ++i) {
            int q = i / 3, group = i % 3;
            vector<pair<IdxType, float>> results;
            search(queries->get_vector(q), types[group][q], ef, caches, results);
            // Count each true neighbor once, even if a search returns duplicates.
            for (auto id : gt[i])
                if (any_of(results.begin(), results.end(), [id](const auto &r) { return r.first == id; }))
                    ++hits[group];
        }
        double seconds = chrono::duration<double>(chrono::steady_clock::now() - start).count();
        long total = hits[0] + hits[1] + hits[2];
        double recall = 100.0 * total / (count * K);
        cout << fixed << setprecision(6) << "RESULT_CALIBRATION method=" << method
             << " dataset=" << dataset << " ef_search=" << ef << " recall=" << recall
             << " recall_low=" << 100.0 * hits[0] / (per_set * K)
             << " recall_mid=" << 100.0 * hits[1] / (per_set * K)
             << " recall_high=" << 100.0 * hits[2] / (per_set * K)
             << " qps=" << count / seconds << " seconds=" << seconds
             << " queries=" << count << " K=" << K << " threads=" << threads << endl;
        measured_hits[ef] = total;
        for (auto it = next(measured_hits.begin()); it != measured_hits.end(); ++it) {
            auto previous = prev(it);
            if (previous->second > it->second) {
                observed_nonmonotonic = true;
                cout << "CALIBRATION_NONMONOTONIC method=" << method
                     << " lower_ef=" << previous->first << " higher_ef=" << it->first << endl;
            }
        }
        return total;
    };
    // Cache each measurement across the two strict thresholds. Binary search
    // assumes monotonic threshold crossing; FAVOR need not satisfy that exactly.
    cout << "CALIBRATION_SEARCH method=" << method << " strategy=binary step=10 max_ef="
         << max_ef << " monotonicity_assumed=1" << endl;
    auto threshold = [&](int target) {
        int result = calibration_binary_threshold(max_ef / 10, [&](int grid_index) {
            return measure(grid_index * 10) * 100L > long(target) * count * K;
        }) * 10;
        cout << "CALIBRATION_THRESHOLD method=" << method << " target=" << target
             << " ef_search=" << result << " search=binary" << endl;
        return result;
    };
    if (target_recall) {
        int ef = threshold(target_recall);
        cout << "CALIBRATION_SUMMARY method=" << method << " target=" << target_recall
             << " ef_search=" << ef << " max_ef=" << max_ef
             << " search=binary evaluations=" << measured_hits.size()
             << " observed_nonmonotonic=" << observed_nonmonotonic << endl;
        return 0;
    }
    int first95 = threshold(95);
    int first90 = threshold(90);
    cout << "CALIBRATION_SUMMARY method=" << method << " first_ef_gt90=" << first90
         << " first_ef_gt95=" << first95 << " max_ef=" << max_ef
         << " search=binary evaluations=" << measured_hits.size()
         << " observed_nonmonotonic=" << observed_nonmonotonic << endl;
    return 0;
}
} // namespace ANNS
