#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <cassert>
#include "concurrent_edge_map.h"

using namespace ANNS;

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "FAIL: " << msg << "  (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        failures++; \
    } \
} while(0)

void test_reserve_then_insert() {
    std::cout << "test_reserve_then_insert..." << std::endl;

    ConcurrentEdgeMap m;
    m.reserve(1000);

    // Fill to capacity — should not throw
    for (IdxType i = 0; i < 1000; i++) {
        m.insert_empty(i);
    }
    CHECK(m.size() == 1000, "size should be 1000 after inserting 1000 keys");

    // Verify all keys are accessible
    for (IdxType i = 0; i < 1000; i++) {
        auto guard = m.try_read_lock(i);
        CHECK(guard, "key should exist after insert");
        CHECK(guard.edges->empty(), "edge list should be empty after insert_empty");
    }
    std::cout << "  PASSED" << std::endl;
}

void test_reserve_then_emplace_with_edges() {
    std::cout << "test_reserve_then_emplace_with_edges..." << std::endl;

    ConcurrentEdgeMap m;
    m.reserve(100);

    for (IdxType i = 0; i < 100; i++) {
        std::vector<IdxType> edges = {i + 1, i + 2, i + 3};
        m.emplace(i, std::move(edges));
    }
    CHECK(m.size() == 100, "size should be 100");

    for (IdxType i = 0; i < 100; i++) {
        auto& edges = m.at_unsafe(i);
        CHECK(edges.size() == 3, "each key should have 3 edges");
        CHECK(edges[0] == i + 1, "first edge value check");
    }
    std::cout << "  PASSED" << std::endl;
}

void test_overflow_grows() {
    std::cout << "test_overflow_grows..." << std::endl;

    // TBB dynamically resizes — inserting beyond initial capacity should work.
    ConcurrentEdgeMap m;
    m.reserve(10);
    for (IdxType i = 0; i < 100; i++) {
        m.insert_empty(i);
    }
    CHECK(m.size() == 100, "should hold 100 entries after dynamic growth");
    std::cout << "  PASSED" << std::endl;
}

void test_thread_safe_insert_empty() {
    std::cout << "test_thread_safe_insert_empty..." << std::endl;

    ConcurrentEdgeMap m;
    const int N = 10000;
    m.reserve(N);

    // Pre-populate half single-threaded
    for (IdxType i = 0; i < N / 2; i++) {
        m.insert_empty(i);
    }

    // Insert the other half from multiple threads
    const int n_threads = 8;
    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; t++) {
        threads.emplace_back([&m, t, N, n_threads]() {
            for (IdxType i = N / 2 + t; i < N; i += n_threads) {
                m.insert_empty(i);
            }
        });
    }
    for (auto& t : threads) t.join();

    CHECK(m.size() == N, "all keys should be present after concurrent insert");
    for (IdxType i = 0; i < N; i++) {
        auto guard = m.try_read_lock(i);
        CHECK(guard, "every key 0..N-1 should exist");
    }
    std::cout << "  PASSED" << std::endl;
}

void test_concurrent_read_write() {
    std::cout << "test_concurrent_read_write..." << std::endl;

    ConcurrentEdgeMap m;
    const int N = 1000;
    m.reserve(N);
    for (IdxType i = 0; i < N; i++) {
        m.emplace(i, std::vector<IdxType>{i});
    }

    std::atomic<bool> stop{false};
    std::atomic<int> reads_done{0};
    std::atomic<int> writes_done{0};

    // Reader threads: continuously read edge lists
    const int n_readers = 4;
    std::vector<std::thread> readers;
    for (int t = 0; t < n_readers; t++) {
        readers.emplace_back([&]() {
            while (!stop.load(std::memory_order_relaxed)) {
                for (IdxType i = 0; i < N; i++) {
                    auto guard = m.read_lock(i);
                    // Just read — ensure no crash
                    volatile size_t sz = guard.edges->size();
                    (void)sz;
                    reads_done.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    // Writer threads: continuously append to edge lists
    const int n_writers = 2;
    std::vector<std::thread> writers;
    for (int t = 0; t < n_writers; t++) {
        writers.emplace_back([&, t]() {
            for (int round = 0; round < 100; round++) {
                for (IdxType i = t; i < N; i += n_writers) {
                    auto guard = m.write_lock(i);
                    guard.edges->push_back(round);
                    writes_done.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    // Let writers finish, then stop readers
    for (auto& t : writers) t.join();
    stop.store(true, std::memory_order_relaxed);
    for (auto& t : readers) t.join();

    CHECK(reads_done > 0, "readers should have completed some reads");
    CHECK(writes_done > 0, "writers should have completed some writes");

    // Verify final state: each key should have 1 (initial) + 100 (per writer) edges
    for (IdxType i = 0; i < N; i++) {
        auto& edges = m.at_unsafe(i);
        size_t expected = 1 + 100;  // initial + 100 rounds from one writer
        CHECK(edges.size() == expected,
              "key " + std::to_string(i) + " has " + std::to_string(edges.size()) +
              " edges, expected " + std::to_string(expected));
    }
    std::cout << "  PASSED" << std::endl;
}

void test_read_lock_missing_key() {
    std::cout << "test_read_lock_missing_key..." << std::endl;

    ConcurrentEdgeMap m;
    m.reserve(10);
    m.insert_empty(0);

    bool threw = false;
    try {
        auto guard = m.read_lock(999);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw, "read_lock on missing key should throw out_of_range");
    std::cout << "  PASSED" << std::endl;
}

void test_erase_basic() {
    std::cout << "test_erase_basic..." << std::endl;

    ConcurrentEdgeMap m;
    m.reserve(100);
    for (IdxType i = 0; i < 50; i++) {
        m.emplace(i, std::vector<IdxType>{i + 1, i + 2});
    }
    CHECK(m.size() == 50, "initial size should be 50");

    // Erase every other key
    for (IdxType i = 0; i < 50; i += 2) {
        bool ok = m.erase(i);
        CHECK(ok, "erase of existing key should return true");
    }
    CHECK(m.size() == 25, "size should be 25 after erasing 25 keys");

    // Erased keys should not be found
    for (IdxType i = 0; i < 50; i += 2) {
        auto guard = m.try_read_lock(i);
        CHECK(!guard, "erased key should not be found");
    }
    // Surviving keys should still be accessible
    for (IdxType i = 1; i < 50; i += 2) {
        auto guard = m.try_read_lock(i);
        CHECK(guard, "surviving key should still exist");
        CHECK(guard.edges->size() == 2, "surviving key should retain its edges");
    }

    // Erase of non-existent key
    CHECK(!m.erase(999), "erase of missing key should return false");
    // Double erase
    CHECK(!m.erase(0), "double erase should return false");

    std::cout << "  PASSED" << std::endl;
}

void test_erase_then_reinsert() {
    std::cout << "test_erase_then_reinsert..." << std::endl;

    ConcurrentEdgeMap m;
    m.reserve(100);
    for (IdxType i = 0; i < 50; i++) {
        m.emplace(i, std::vector<IdxType>{i * 10});
    }

    // Erase all keys
    for (IdxType i = 0; i < 50; i++) {
        m.erase(i);
    }
    CHECK(m.size() == 0, "size should be 0 after erasing all");

    // Reinsert the same keys
    for (IdxType i = 0; i < 50; i++) {
        bool ok = m.insert_empty(i);
        CHECK(ok, "reinsert into erased slot should succeed");
    }
    CHECK(m.size() == 50, "size should be 50 after reinserting");

    // All keys should exist with empty edge lists
    for (IdxType i = 0; i < 50; i++) {
        auto guard = m.try_read_lock(i);
        CHECK(guard, "reinserted key should exist");
        CHECK(guard.edges->empty(), "reinserted key should have empty edges");
    }

    std::cout << "  PASSED" << std::endl;
}

void test_erase_preserves_lookups() {
    std::cout << "test_erase_preserves_lookups..." << std::endl;

    // Use a small table so collisions are guaranteed.
    ConcurrentEdgeMap m;
    m.reserve(4);

    // Insert 6 keys (high load factor)
    for (IdxType i = 0; i < 6; i++) {
        m.insert_empty(i);
    }
    CHECK(m.size() == 6, "should have 6 keys");

    // Erase keys 0, 2, 4
    m.erase(0);
    m.erase(2);
    m.erase(4);

    // Keys 1, 3, 5 should still be found
    CHECK(m.try_read_lock(1), "key 1 should survive");
    CHECK(m.try_read_lock(3), "key 3 should survive");
    CHECK(m.try_read_lock(5), "key 5 should survive");
    CHECK(!m.try_read_lock(0), "erased key 0 should not be found");

    std::cout << "  PASSED" << std::endl;
}

void test_concurrent_erase_insert_read() {
    std::cout << "test_concurrent_erase_insert_read (stress)..." << std::endl;

    ConcurrentEdgeMap m;
    const int N = 2000;
    m.reserve(N);

    // Pre-populate all keys
    for (IdxType i = 0; i < N; i++) {
        m.emplace(i, std::vector<IdxType>{i});
    }

    const int n_rounds = 50;
    std::atomic<bool> stop{false};
    std::atomic<int> erases{0}, inserts{0}, reads{0}, writes{0};
    std::atomic<int> read_misses{0}, write_misses{0};

    std::vector<std::thread> threads;

    // Eraser threads: repeatedly erase random keys
    for (int t = 0; t < 2; t++) {
        threads.emplace_back([&, t]() {
            for (int round = 0; round < n_rounds; round++) {
                for (IdxType i = t; i < N; i += 2) {
                    if (m.erase(i))
                        erases.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    // Inserter threads: repeatedly reinsert keys (racing with erasers)
    for (int t = 0; t < 2; t++) {
        threads.emplace_back([&, t]() {
            for (int round = 0; round < n_rounds; round++) {
                for (IdxType i = t; i < N; i += 2) {
                    if (m.insert_empty(i))
                        inserts.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    // Reader threads: try_read_lock on arbitrary keys — WILL hit erased keys
    for (int t = 0; t < 4; t++) {
        threads.emplace_back([&]() {
            while (!stop.load(std::memory_order_relaxed)) {
                for (IdxType i = 0; i < N; i++) {
                    auto guard = m.try_read_lock(i);
                    if (guard) {
                        volatile size_t sz = guard.edges->size();
                        (void)sz;
                        reads.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        read_misses.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        });
    }

    // Writer threads: try_write_lock on arbitrary keys — WILL hit erased keys
    for (int t = 0; t < 2; t++) {
        threads.emplace_back([&]() {
            while (!stop.load(std::memory_order_relaxed)) {
                for (IdxType i = 0; i < N; i++) {
                    auto guard = m.try_write_lock(i);
                    if (guard) {
                        guard.edges->push_back(42);
                        writes.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        write_misses.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        });
    }

    // Let erasers and inserters finish, then stop readers/writers
    for (int i = 0; i < 4; i++) threads[i].join();
    stop.store(true, std::memory_order_relaxed);
    for (size_t i = 4; i < threads.size(); i++) threads[i].join();

    std::cout << "  erases=" << erases << " inserts=" << inserts
              << " reads=" << reads << " writes=" << writes
              << " read_misses=" << read_misses
              << " write_misses=" << write_misses << std::endl;

    CHECK(erases > 0, "should have erased some keys");
    CHECK(inserts > 0, "should have inserted some keys");
    CHECK(reads > 0, "should have completed some reads");
    CHECK(read_misses > 0, "readers SHOULD hit erased keys (proves the race exists)");

    // Reinsert everything and verify the map is still consistent
    for (IdxType i = 0; i < N; i++) {
        m.insert_empty(i);  // no-op if already present
    }
    CHECK(m.size() >= (size_t)N, "all keys should be present after recovery");

    // Every present key should be lockable without crash
    for (IdxType i = 0; i < N; i++) {
        auto guard = m.try_read_lock(i);
        CHECK(guard, "key should exist after full reinsert");
    }

    std::cout << "  PASSED" << std::endl;
}

int main() {
    test_reserve_then_insert();
    test_reserve_then_emplace_with_edges();
    test_overflow_grows();
    test_thread_safe_insert_empty();
    test_concurrent_read_write();
    test_read_lock_missing_key();
    test_erase_basic();
    test_erase_then_reinsert();
    test_erase_preserves_lookups();
    test_concurrent_erase_insert_read();

    std::cout << std::endl;
    if (failures == 0) {
        std::cout << "All tests passed." << std::endl;
    } else {
        std::cout << failures << " test(s) FAILED." << std::endl;
    }
    return failures;
}
