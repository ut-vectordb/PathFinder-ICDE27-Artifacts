#ifndef CONCURRENT_EDGE_MAP_H
#define CONCURRENT_EDGE_MAP_H

#include <vector>
#include <memory>
#include <random>
#include <thread>
#include <stdexcept>
#include <functional>
#include <tbb/concurrent_hash_map.h>
#include "config.h"

namespace ANNS {

/**
 * Thread-safe hash map from IdxType → vector<IdxType>, backed by
 * tbb::concurrent_hash_map.  Drop-in replacement for the old
 * open-addressing implementation (which suffered from tombstone
 * probe-chain blowup under deletions).
 *
 * Concurrency guarantees are provided by TBB's per-bucket reader-writer
 * locks via accessor / const_accessor handles.
 */
class ConcurrentEdgeMap {
public:
    static constexpr size_t DEFAULT_BUCKETS_PER_STRIPE = 4;  // kept for API compat

private:
    using MapType = tbb::concurrent_hash_map<IdxType, std::vector<IdxType>>;
    MapType map_;

public:
    ConcurrentEdgeMap() = default;

    // Non-copyable (contains TBB map)
    ConcurrentEdgeMap(const ConcurrentEdgeMap&) = delete;
    ConcurrentEdgeMap& operator=(const ConcurrentEdgeMap&) = delete;

    // --------------- Initialization ---------------

    void reserve(size_t n, size_t /*buckets_per_stripe*/ = DEFAULT_BUCKETS_PER_STRIPE) {
        // TBB uses n as a bucket-count hint (not element count).
        // Match old behaviour: 2*n buckets ≈ 50% load factor.
        map_.rehash(n * 2);
    }

    // --------------- Insert ---------------

    void emplace(IdxType key, std::vector<IdxType>&& edges) {
        MapType::accessor acc;
        map_.insert(acc, key);
        acc->second = std::move(edges);
    }

    bool insert_empty(IdxType key) {
        return map_.insert(std::make_pair(key, std::vector<IdxType>{}));
    }

    // --------------- Delete ---------------

    bool erase(IdxType key) {
        return map_.erase(key);
    }

    // --------------- Lookup ---------------

    size_t size() const { return map_.size(); }

    // --------------- Locked access to edge lists ---------------
    //
    // TBB accessors are non-movable, so we heap-allocate them inside
    // the guard via unique_ptr.  The guard itself is movable.

    struct ReadGuard {
        std::unique_ptr<MapType::const_accessor> acc_;
        const std::vector<IdxType>* edges;
    };

    ReadGuard read_lock(IdxType key) const {
        auto acc = std::make_unique<MapType::const_accessor>();
        if (!map_.find(*acc, key))
            throw std::out_of_range("ConcurrentEdgeMap::read_lock: key not found");
        const std::vector<IdxType>* e = &(*acc)->second;
        return ReadGuard{std::move(acc), e};
    }

    struct WriteGuard {
        std::unique_ptr<MapType::accessor> acc_;
        std::vector<IdxType>* edges;
    };

    WriteGuard write_lock(IdxType key) {
        auto acc = std::make_unique<MapType::accessor>();
        if (!map_.find(*acc, key))
            throw std::out_of_range("ConcurrentEdgeMap::write_lock: key not found");
        std::vector<IdxType>* e = &(*acc)->second;
        return WriteGuard{std::move(acc), e};
    }

    // --------------- Non-throwing locked access ---------------

    struct OptionalReadGuard {
        std::unique_ptr<MapType::const_accessor> acc_;
        const std::vector<IdxType>* edges = nullptr;
        explicit operator bool() const { return edges != nullptr; }
    };

    OptionalReadGuard try_read_lock(IdxType key) const {
        auto acc = std::make_unique<MapType::const_accessor>();
        if (map_.find(*acc, key)) {
            const std::vector<IdxType>* e = &(*acc)->second;
            return OptionalReadGuard{std::move(acc), e};
        }
        return OptionalReadGuard{};
    }

    struct OptionalWriteGuard {
        std::unique_ptr<MapType::accessor> acc_;
        std::vector<IdxType>* edges = nullptr;
        explicit operator bool() const { return edges != nullptr; }
    };

    OptionalWriteGuard try_write_lock(IdxType key) {
        auto acc = std::make_unique<MapType::accessor>();
        if (map_.find(*acc, key)) {
            std::vector<IdxType>* e = &(*acc)->second;
            return OptionalWriteGuard{std::move(acc), e};
        }
        return OptionalWriteGuard{};
    }

    // --------------- Stack-allocated locked access (no heap alloc) ----
    //
    // Caller owns the accessor on their stack frame.  Usage:
    //   InPlaceReadGuard guard;
    //   if (edgemap.try_read_lock(key, guard)) { use guard.edges; }
    //   // guard dtor releases lock

    struct InPlaceReadGuard {
        MapType::const_accessor acc;
        const std::vector<IdxType>* edges = nullptr;
        explicit operator bool() const { return edges != nullptr; }
        void release() { acc.release(); edges = nullptr; }
    };

    bool try_read_lock(IdxType key, InPlaceReadGuard &guard) const {
        guard.edges = nullptr;
        if (map_.find(guard.acc, key)) {
            guard.edges = &guard.acc->second;
            return true;
        }
        return false;
    }

    struct InPlaceWriteGuard {
        MapType::accessor acc;
        std::vector<IdxType>* edges = nullptr;
        explicit operator bool() const { return edges != nullptr; }
        void release() { acc.release(); edges = nullptr; }
    };

    bool try_write_lock(IdxType key, InPlaceWriteGuard &guard) {
        guard.edges = nullptr;
        if (map_.find(guard.acc, key)) {
            guard.edges = &guard.acc->second;
            return true;
        }
        return false;
    }

    // --------------- Sampling (thread-safe) ---------------

    /// Return up to `k` randomly-chosen live keys.
    /// Uses a random starting offset into the TBB hash map's internal
    /// iteration order, which is already pseudo-random (hash-bucketed).
    std::vector<IdxType> sample_keys(int k) const {
        size_t n = map_.size();
        if (n == 0 || k <= 0) return {};
        k = std::min(k, (int)n);

        // Pick a random offset so concurrent callers don't all grab the same keys
        thread_local std::mt19937 rng(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        size_t offset = dist(rng);

        std::vector<IdxType> result;
        result.reserve(k);

        // Advance iterator to offset, then collect k keys (wrap around)
        auto it = map_.begin();
        for (size_t i = 0; i < offset && it != map_.end(); ++i, ++it) {}
        for (int i = 0; i < k; ++i) {
            if (it == map_.end()) it = map_.begin();
            if (it == map_.end()) break;  // map became empty
            result.push_back(it->first);
            ++it;
        }
        return result;
    }

    // --------------- Unlocked access (single-threaded only) ---------------

    std::vector<IdxType>& at_unsafe(IdxType key) {
        MapType::accessor acc;
        if (!map_.find(acc, key))
            throw std::out_of_range("ConcurrentEdgeMap::at_unsafe: key not found");
        return acc->second;
    }

    const std::vector<IdxType>& at_unsafe(IdxType key) const {
        MapType::const_accessor acc;
        if (!map_.find(acc, key))
            throw std::out_of_range("ConcurrentEdgeMap::at_unsafe: key not found");
        return acc->second;
    }

    // --------------- Iteration (single-threaded only) ---------------

    class iterator {
    public:
        iterator(MapType::iterator it) : it_(it) {}
        std::pair<IdxType, std::vector<IdxType>&> operator*() {
            return {it_->first, it_->second};
        }
        iterator& operator++() { ++it_; return *this; }
        bool operator!=(const iterator& o) const { return it_ != o.it_; }
    private:
        MapType::iterator it_;
    };

    class const_iterator {
    public:
        const_iterator(MapType::const_iterator it) : it_(it) {}
        std::pair<IdxType, const std::vector<IdxType>&> operator*() const {
            return {it_->first, it_->second};
        }
        const_iterator& operator++() { ++it_; return *this; }
        bool operator!=(const const_iterator& o) const { return it_ != o.it_; }
    private:
        MapType::const_iterator it_;
    };

    iterator begin() { return iterator(map_.begin()); }
    iterator end()   { return iterator(map_.end()); }
    const_iterator begin() const { return const_iterator(map_.begin()); }
    const_iterator end()   const { return const_iterator(map_.end()); }

    void clear() { map_.clear(); }
};

} // namespace ANNS

#endif // CONCURRENT_EDGE_MAP_H
