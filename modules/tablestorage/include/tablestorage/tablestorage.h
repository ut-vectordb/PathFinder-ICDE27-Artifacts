#ifndef TABLESTORAGE_H

#include <vector>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <mutex>
#include <unordered_set>

#include "storage.h"
#include "btreeolc/btreeolc.hpp"
#include "tablestorage_utils.h"

// #define BYPASS_B_TREE

/**
 * This serves as the "Primary Table". It maps a unique identifier to a tuple of attributes and vector data.
 * The interfaces won't allow direct access to data; instead, we expose the interface to a B-Tree, which is used to look up the pointers to the actual data.
 * 
 * In PostgreSQL terms, this is a "heap storage, with B+ Tree indexing"
 */
class TableStorage {
public:
    using KeyType = uint32_t;

    struct ValueType {
        float *p;
        ValueType() : p(nullptr) {}
        ValueType(float *pp) : p(pp) {}
        ValueType(char *pp) : p((float *)pp) {}
        ValueType(const ValueType &v) : p(v.p) {}
        ValueType &operator=(const ValueType &v) { p = v.p; return *this; }
        void setValue(float *pp) { p = pp; }
        float *getValue() const { return p; }
    };
    struct KeyWrapper {
        KeyType key;
        KeyWrapper() = default;
        KeyWrapper(KeyType k) : key(k) {}
        // KeyWrapper(const KeyWrapper &v) : key(v.key) {}
        KeyType getValue() const { return key; }
        char *transfer() const { return nullptr; }
    };

    size_t nb;
    int na;
    int n_dim;

    int dim_byte_count = 0; // TableStorage was designed assuming float; This is an attempt to support uint8
    size_t vector_bytes = 0;
    size_t attr_offset = 0;
    size_t aligned_record_size = 0;

    void describe() const;

    static void load_table_storage_helper(const std::shared_ptr<ANNS::IStorage> base_storage, const std::vector<float> &attr_data, TableStorage &tbstorage);

#ifndef BYPASS_B_TREE
    const float *get_attr(const void *p) const  { return (float *)((char *)p + attr_offset); }
    const char *get_vector(const void *p) const { return (char *)p; }
    float *get_attr(void *p) const              { return (float *)((char *)p + attr_offset); }
    char *get_vector(void *p) const             { return (char *)p; }
    char *alloc_new_row() const;
#else
    // Encode id+1 so the valid row with id=0 is not confused with nullptr.
    const float *get_attr(const void *p) const { return attr_data.data() + (size_t(p) - 1) * na; }
    const char *get_vector(const void *p) const { return (const char *)base_storage->get_vector(size_t(p) - 1); }
    float *get_attr(void *p) { return attr_data.data() + (size_t(p) - 1) * na; }
    char *get_vector(void *p) { return (char *)base_storage->get_vector(size_t(p) - 1); }
#endif

    bool insert(KeyType key, const float *vecdata, const float *attrdata);
    void *lookup(KeyType key);
    void *find(KeyType key);
    std::vector<KeyType> keys();
    bool remove(KeyType key);

    static constexpr int roundUp(int numToRound, int multiple) {
        if (multiple == 0)
            return numToRound;

        int remainder = numToRound % multiple;
        if (remainder == 0)
            return numToRound;

        return numToRound + multiple - remainder;
    }

// private:
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

    btreeolc::BPlusTree<KeyWrapper, ValueType, Comparator<KeyWrapper>, Comparator<ValueType>, 10, 4096, 4096> tree;
    mutable std::mutex live_key_mutex;
    std::unordered_set<KeyType> live_key_set;

    typedef enum { NUMERICAL=0, CATEGORICAL=1 } ColumnType;
    std::vector<ColumnType> table_schema;   // TODO: When reading the queries, infer the type from there

#ifdef BYPASS_B_TREE
    std::shared_ptr<ANNS::IStorage> base_storage;
    std::vector<float> attr_data;
    std::vector<KeyType> live_keys;
#endif
};

#define TABLESTORAGE_H
#endif
