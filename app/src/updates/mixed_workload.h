#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
#include <omp.h>

namespace ANNS {
// Shuffle a fixed operation multiset: exact ratios, random worker assignments.
// Queries uniformly cycle the equally sized low/mid/high sets before shuffling.
template<class Query, class Insert, class Delete>
int run_mixed_workload(const std::string &method, int total, int percent, int threads,
                       int ef, int query_count, Query query, Insert insert, Delete remove) {
    if (total <= 0 || percent < 0 || percent > 100 || threads <= 0 || ef <= 0 ||
        query_count <= 0 || (int64_t(total)*percent)%200 != 0)
        throw std::invalid_argument("Mixed workload requires exact equal insert/delete counts");
    struct Op { int kind, id; };
    int updates = int64_t(total)*percent/200;
    std::vector<Op> ops;
    ops.reserve(total);
    for (int i=0;i<total-2*updates;++i) ops.push_back({0,i%query_count});
    for (int i=0;i<updates;++i) { ops.push_back({1,i}); ops.push_back({2,i}); }
    std::mt19937 rng(42);
    std::shuffle(ops.begin(),ops.end(),rng);
    std::exception_ptr error;
    std::mutex error_mu;
    std::atomic<bool> failed{false};
    auto guarded = [&](auto action) {
        try { action(); } catch (...) {
            std::lock_guard<std::mutex> lock(error_mu);
            if (!failed.exchange(true)) error=std::current_exception();
        }
    };
    using Clock=std::chrono::steady_clock;
    // Initial-state timing only: no ground truth, recall, or ef calibration.
    auto begin=Clock::now();
    #pragma omp parallel for num_threads(threads) schedule(dynamic,1)
    for (int i=0;i<query_count;++i) {
        if (!failed.load()) guarded([&]{query(i);});
    }
    if (error) std::rethrow_exception(error);
    double seconds=std::chrono::duration<double>(Clock::now()-begin).count();
    std::cout << std::setprecision(12) << "RESULT_QUERY_ONLY method=" << method
              << " threads=" << threads << " ef_search=" << ef << " queries=" << query_count
              << " seconds=" << seconds << " qps=" << query_count/seconds << std::endl;
    std::cout << "MIXED_START method=" << method << " operations=" << total
              << " update_percent=" << percent << " threads=" << threads << std::endl;
    int nq=0,ni=0,nd=0;
    begin=Clock::now();
    #pragma omp parallel for num_threads(threads) schedule(dynamic,1) reduction(+:nq,ni,nd)
    for (int i=0;i<total;++i) {
        if (failed.load()) continue;
        auto op=ops[i];
        guarded([&] {
            if (op.kind==0) {query(op.id); ++nq;}
            else if (op.kind==1) {insert(op.id); ++ni;}
            else {remove(op.id); ++nd;}
        });
    }
    seconds=std::chrono::duration<double>(Clock::now()-begin).count();
    if (error) std::rethrow_exception(error);
    if (nq+ni+nd!=total || ni!=updates || nd!=updates)
        throw std::runtime_error("Mixed operation count mismatch");
    std::cout << "RESULT_MIXED method=" << method << " threads=" << threads
              << " ef_search=" << ef << " update_percent=" << percent
              << " operations=" << total << " queries=" << nq << " inserts=" << ni
              << " deletes=" << nd << " seconds=" << seconds << " ops_per_second=" << total/seconds << std::endl;
    return 0;
}
}
