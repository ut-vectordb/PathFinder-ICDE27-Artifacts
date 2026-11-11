#include <iostream>
#include <thread>
#include <utility>
#include <vector>
#include <chrono>
#include <thread>
#include <type_traits>  // std::is_trivial

#include "btreeolc/btreeolc.hpp"
#include "tablestorage/tablestorage.h"

class Value {
public:
    Value() : value(-1) {}
    Value(int v) : value(v) {}

    Value(const Value &v) : value(v.value) {}

    Value &operator=(const Value &v) {
        value = v.value;
        return *this;
    }
    void setValue(int v) { value = v;}
    int getValue() const { return value; }
private:
    int value;
};

template<typename T>
class TrivialKey {
public:
    TrivialKey() = default;
    TrivialKey(T v) : value(v) {}

    T getValue() const { return value; }

    char *transfer() const { return nullptr; }  // just for convenience

private:
    T value;
};
static_assert(std::is_trivial<TrivialKey<int>>::value == true, "Test for trivial");

template<typename T>
struct Comparator {
    int operator()(const T &a, const T &b) const {
        if (a.getValue() > b.getValue())
            return 1;
        else if (a.getValue() == b.getValue())
            return 0;
        else
            return -1;
    }
};

template<typename KeyType, uint64_t leaf_size, uint64_t inner_size>
using btree_t = btreeolc::BPlusTree<KeyType, Value, Comparator<KeyType>, Comparator<Value>, 10, leaf_size, inner_size>;

#define ASSERT_EQ(x, y) do { \
    if ((x) != (y)) { \
        fprintf(stderr, "Assertion failed: %s == %s\n  Actual: %d != %d\n  In file %s, line %d\n", \
                #x, #y, (int)(x), (int)(y), __FILE__, __LINE__); \
        std::abort(); \
    } \
} while (0)

#define ASSERT_FALSE(x) ASSERT_EQ((x), 0)

template<typename KeyType, uint64_t leaf_size, uint64_t inner_size>
void singleInsertTest(btree_t<KeyType, leaf_size, inner_size> &btree) {
    std::vector<int> keys;
    constexpr int num_keys = 50;
    for (int i = 0; i < num_keys; ++i) {
        keys.push_back(i);
    }
    std::random_shuffle(keys.begin(), keys.end());
    for (int i = 0; i < num_keys; i++) {
        KeyType k{keys[i]};
        bool res = btree.insert(k, keys[i]);
        ASSERT_EQ(res, true);
    }
    for (int i = 0; i < num_keys; i++) {
        KeyType k{keys[i]};
        bool res = btree.insert(k, 0);
        ASSERT_EQ(res, false); // except insert failed
        delete[] k.transfer();
    }
    auto kv_size = sizeof(KeyType) + sizeof(Value);
    auto inner_nodes = btree.getNumInnerNodes();
    auto leaf_nodes = btree.getNumLeafNodes();
    auto tree_bytes = leaf_size * leaf_nodes + inner_nodes * inner_size;
    std::cout << "inner nodes " << inner_nodes << ", leaf nodes " <<  leaf_nodes << ", kv_size " << kv_size << ", tree_bytes " << tree_bytes << ", # keys " << 100000 << ", # keys per leave " << 100000.0 / leaf_nodes << std::endl;;
}

void test_btree() {
    using KeyType = TrivialKey<int>;
    // const size_t page_size = 64;
    // const size_t page_size = 4096;
    const size_t page_size = 256;
    btree_t<KeyType, page_size, page_size> btree;

    const size_t leaf_size = page_size;
    const size_t inner_size = page_size;
    // singleInsertTest<Type, leaf_size, inner_size>(btree);

    for (int i = 0; i < 10000; i++) {
        KeyType k{i};
        Value res;
        btree.insert(k, i);
        btree.lookup(k, res);
        ASSERT_EQ(res.getValue(), i);
    }
}

void test_table_read_performance(size_t n_dim=512, size_t nb=1000000, size_t na=5) {
    TableStorage tb;
    tb.n_dim = n_dim;
    tb.nb = nb;
    tb.na = na;

    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    
    // Insert `nb` entries
    for (int i = 0; i < nb; i++) {
        std::vector<float> vec(n_dim), attr(na);
        for (int k = 0; k < n_dim; k++) vec[k] = dist(rng);
        for (int k = 0; k < na; k++) attr[k] = dist(rng);
        attr[0] = i; // tag
        if (!tb.insert(i, vec.data(), attr.data())) {
            std::cerr << "Failed to insert key " << i << std::endl;
        }
    }

    // Prepare random lookup keys
    std::uniform_int_distribution<int> key_dist(0, nb - 1);
    const size_t num_queries = 10000 * 50;
    const size_t num_queries_parallel = 10000000 * 50;
    using namespace std::chrono;

    // Measure lookup latency
    {
        std::vector<int> keys(num_queries);
        for (int i = 0; i < num_queries; i++) { keys[i] = key_dist(rng); }

        auto start = high_resolution_clock::now();
        size_t found = 0;
        for (int i = 0; i < num_queries; i++) {
            void *row = tb.lookup(keys[i]);
            if (row) found++;
        }
        auto end = high_resolution_clock::now();
        auto total_time_ns = duration_cast<nanoseconds>(end - start).count();
        double avg_latency_ns = static_cast<double>(total_time_ns) / num_queries;

        std::cout << "Single-threaded lookup performance:" << std::endl;
        std::cout << " - Performed " << num_queries << " lookups. Found: " << found << std::endl;
        std::cout << " - Average lookup latency: " << avg_latency_ns << " ns" << std::endl;
    }
    std::cout << "\n";
    {
        std::vector<int> keys(num_queries);
        for (int i = 0; i < num_queries; i++) { keys[i] = key_dist(rng); }

        auto start = high_resolution_clock::now();
        size_t found = 0;
        for (int i = 0; i < num_queries; i++) {
            void *row = tb.lookup(keys[i]);
            if (row) found++;
        }
        auto end = high_resolution_clock::now();
        auto total_time_ns = duration_cast<nanoseconds>(end - start).count();
        double avg_latency_ns = static_cast<double>(total_time_ns) / num_queries;

        std::cout << "Single-threaded lookup performance:" << std::endl;
        std::cout << " - Performed " << num_queries << " lookups. Found: " << found << std::endl;
        std::cout << " - Average lookup latency: " << avg_latency_ns << " ns" << std::endl;
    }
    std::cout << "\n";
    {
        std::vector<int> keys(num_queries_parallel);
        for (int i = 0; i < num_queries_parallel; i++) { keys[i] = key_dist(rng); }

        int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 1;
        std::cout << "Using " << num_threads << " threads for parallel lookups." << std::endl;
        
        auto start = high_resolution_clock::now();
        size_t found = 0;
        #pragma omp parallel for num_threads(num_threads) reduction(+:found)
        for (size_t i = 0; i < num_queries_parallel; i++) {
            void *row = tb.lookup(keys[i]);
            if (row) found++;
        }
        auto end = high_resolution_clock::now();
        auto total_time_ns = duration_cast<nanoseconds>(end - start).count();

        // NOTE: the "avg_latency_ns", strictly speaking, is not an actual latency value;
        // it is only the total time divided by the number of queries.
        double avg_latency_ns = static_cast<double>(total_time_ns) / num_queries_parallel;
        
        std::cout << "Multi-threaded lookup performance:" << std::endl;
        std::cout << " - Performed " << num_queries_parallel << " lookups. Found: " << found << std::endl;
        std::cout << " - Amortized time per lookup (throughput-based): " << avg_latency_ns << " ns" << std::endl;

    }
}

void test_table_storage() {
    TableStorage tb;
    size_t n_dim = 64,
           nb = 100,
           na = 5;
    tb.n_dim = n_dim;
    tb.nb = nb;
    tb.na = na;
    tb.dim_byte_count = 4;
    tb.vector_bytes = (size_t)tb.dim_byte_count * n_dim;
    tb.attr_offset = TableStorage::roundUp(tb.vector_bytes, 4);
    tb.aligned_record_size = TableStorage::roundUp(tb.attr_offset + na * sizeof(float), 32);

    for (int i = 0; i < nb; i++) {
        std::vector<float> vec, attr;
        std::mt19937 rd;
        std::default_random_engine g(rd());
        std::uniform_real_distribution dist(-3.0f, 3.0f);
        for (int k = 0; k < n_dim; k++) {
            vec.push_back(dist(g));
        }
        for (int k = 0; k < na; k++) {
            attr.push_back(dist(g));
        }
        attr[0] = i;
        bool success = tb.insert(i, vec.data(), attr.data());
        if (!success) {
            std::cerr << "Failed to insert key " << i << std::endl;
        }
    }

    void *row = tb.lookup(10);
    bool success = (row != nullptr);
    std::cout << "Lookup 10: " << (success ? "Success" : "Failed") << std::endl;
    std::cout << "Attributes and vectors for key 10: ";
    float *attr = (float *)tb.get_attr(row);
    float *vec = (float *)tb.get_vector(row);
    std::cout << "\nAttributes: ";
    for (int i = 0; i < na; i++) {
        std::cout << attr[i] << " ";
    }
    std::cout << "\nVectors: ";
    for (int i = 0; i < n_dim; i++) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    // test_btree();
    // test_table_storage();
    test_table_read_performance();
    return 0;
}