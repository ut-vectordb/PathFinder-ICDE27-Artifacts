#pragma once
#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <vector>
#include "tablestorage/tablestorage.h"
#include "utils.h"
#include <omp.h>

struct PrefilterResult {
    uint64_t comparisons = 0;
    double seconds = 0;
    std::vector<std::pair<ANNS::IdxType, float>> hits;
};

// Exact prefiltering: preserve the original full-sort ranking, but fetch both
// predicates and vectors through TableStorage's batched scan, as FAVOR does.
template<class Predicate>
PrefilterResult prefilter_scan(TableStorage& table, ANNS::IStorage& queries,
        const std::vector<Predicate>& predicates, const std::vector<int>& query_ids,
        ANNS::DistanceHandler& distance, int k) {
    PrefilterResult result;
    result.hits.resize(predicates.size() * k, {-1, std::numeric_limits<float>::infinity()});
    uint64_t comparisons = 0;
    auto start = std::chrono::steady_clock::now();
    #pragma omp parallel for schedule(dynamic, 4) reduction(+:comparisons)
    for (size_t q = 0; q < predicates.size(); ++q) {
        auto* query = queries.get_vector(query_ids.at(q));
        std::vector<std::pair<ANNS::IdxType, float>> ranking;
        auto visit = [&](uint32_t id, auto* record) {
            if (predicates[q].check(table.get_attr(record))) {
                ++comparisons;
                ranking.emplace_back(id, distance.compute(query, table.get_vector(record), table.n_dim));
            }
        };
#ifdef BYPASS_B_TREE
        for (uint32_t id = 0; id < table.nb; ++id) visit(id, table.lookup(id));
#else
        using Tree = decltype(table.tree);
        std::vector<typename Tree::KeyValuePair> rows;
        rows.reserve(4096);
        uint32_t low = 0;
        for (;;) {
            table.tree.scan(TableStorage::KeyWrapper(low), TableStorage::KeyWrapper(UINT32_MAX),
                            true, true, 4096, rows);
            if (rows.empty()) break;
            for (const auto& row : rows) visit(row.first.getValue(), row.second.getValue());
            auto last = rows.back().first.getValue();
            if (rows.size() < 4096 || last == UINT32_MAX) break;
            low = last + 1;
        }
#endif
        std::sort(ranking.begin(), ranking.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
        for (size_t j = 0; j < std::min(size_t(k), ranking.size()); ++j) result.hits[q*k+j] = ranking[j];
    }
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    result.comparisons = comparisons;
    return result;
}
