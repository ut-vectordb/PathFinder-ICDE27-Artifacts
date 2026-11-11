#ifndef ATTRIBUTE_TREE_INCLUDED

#include "irangegraph.h"
#include "tablestorage/tablestorage.h"
#include "concurrent_edge_map.h"
#include <functional>
#include <unordered_map>

#define REMOVE_INDIRECTION

namespace ANNS {

class AttrTreeNode {
public:
    int node_id;        // this is guaranteed to be the same as the index number in "treenodes"
    float lbound, rbound_open;
    int depth;
    int size;   // how many tuples does this node contain?
    size_t size_e; // number of edges in total;;;

    std::vector<AttrTreeNode *> childs;
    AttrTreeNode *parent;
    AttrTreeNode *delegate_root;    // if a root node is omitted, there will be a delegate root to use

    // ID mapping: global_id → packed_internal_id.
    //
    // Build-time bridge only.  Graph requires contiguous IDs 0..N-1 ("packed"
    // or "internal" IDs), but the rest of the system uses global IDs (the
    // original position in the base dataset).  vertex_table records that
    // mapping so we can:
    //   1. Build a Graph on packed IDs (via reorder_data or sorted arrays).
    //   2. translate_edge_destination(): rewrite every edge destination in
    //      Graph::neighbors from packed → global, using get_new_to_old().
    //   3. map_global_id_directly_to_edge_table(): move each node's edge
    //      list into ConcurrentEdgeMap keyed by global_id, then free Graph.
    //
    // After step 3, vertex_table is no longer needed and is cleared to free
    // memory.  At query time, ConcurrentEdgeMap operates entirely in global
    // ID space — no reverse lookup is required.
    //
    // The same pattern applies to build_partial_index + online_update_test:
    // the partial builder saves initial_ids.bin (packed → global mapping),
    // the tester reconstructs vertex_table, translates, and maps into the
    // ConcurrentEdgeMap, then discards vertex_table.
    std::unordered_map<IdxType, IdxType> vertex_table;
    std::vector<std::pair<float, float>> value_bounds;

    std::shared_ptr<HNSWUpperLayer> optional_HNSW_upper;
    ConcurrentEdgeMap edgelist_hashmap;

    // also store a graph...
    std::shared_ptr<Graph> g;

    AttrTreeNode(float l, float r_open, int d, int size) : lbound(l), rbound_open(r_open), depth(d), size(size), delegate_root(nullptr) {}
    void translate_edge_destination();
    void translate_HNSW_upper_layer();
    void map_global_id_directly_to_edge_table();

    float report_size_MB();

    void export_txt(std::string filename);
    void export_edge_table(std::string filename, std::vector<float> &attr, int na);

    std::vector<IdxType> get_new_to_old();
    void populate_value_bounds(const std::vector<float> &global_attr, int na);
};

void PrintTree(const AttrTreeNode *node, const std::string& prefix = "", bool isLast = true);

// Attribute-value based Segment Tree
class AttrSegmentTree {
public:
    AttrTreeNode *root{nullptr};
    int max_depth{-1};
    int size_threshold{64};
    std::vector<AttrTreeNode *> treenodes;

    void cutoff_too_narrow(float min_range_width);
    void cutoff_too_small(int min_node_count);
    void cutoff_too_deep(int new_max_depth);

    using pfi = std::pair<float, int>;  // value and ID

    AttrSegmentTree(float l, float r, int size, int size_threshold=64) {
        root = new AttrTreeNode(l, r, 0, size);
        this->size_threshold = size_threshold;
    }

    void BuildTree(AttrTreeNode *u, const pfi *sorted, int depth_max);
    void BuildTree_multiway(AttrTreeNode *u, const pfi *sorted, int depth_max, int fanout);
    std::vector<AttrTreeNode *> range_filter(AttrTreeNode *u, float ql, float qr);

};

// Attribute-value based hash index, mapping each category to one NSW graph

class AttrHashCollection {
public:
    int min_val, max_val;
    std::unordered_map<float, AttrTreeNode *> hash_table;

    std::vector<float> all_keys;
    std::vector<AttrTreeNode *> hashednodes;

    AttrTreeNode *root_collection;  // this thing should be treated in a special way

    using pfi = std::pair<float, int>;  // value and ID

    AttrHashCollection() {}

    void BuildTable(const pfi *sorted, int total_size);
};

/**
 * iRG variant that uses attribute ranges and handles node mappings using a vertex table
 * 
 * TODOs
 * [x] build tree structure
 * [x] build each graph
 *    [x] vertex tables for graphs here. familiar abstraction = good.
 * [x] save, load
 *    [x] save vertex table
 * [x] graphref: how to link here?
 */

class irg_attr {
public:
    enum UnitGraphBuildType { UNIT_GRAPH_NSW, UNIT_GRAPH_HNSW };

    void build(std::shared_ptr<IStorage> base_storage,
        std::vector<float> &attr,
        int attr_id,
        int depth_max,
        std::shared_ptr<DistanceHandler> distance_handler, 
        IdxType M, int ef_con, int n_threads,
        int fanout=2,
        UnitGraphBuildType unit_graph_build_type=UNIT_GRAPH_NSW);

    void build_print_stats(std::shared_ptr<IStorage> base_storage,
        std::vector<float> &attr,
        int attr_id,
        int depth_max,
        std::shared_ptr<DistanceHandler> distance_handler, 
        IdxType M, int ef_con, int n_threads);

    void build_node(AttrTreeNode *u, int n_threads, bool verbose=false);
    std::vector<IdxType> entrypoint_selection(float L_attr, float R_attr, int n_entry);
    std::vector<IdxType> entrypoint_selection(const std::vector<float> &vals, int n_entry);

#ifdef REMOVE_INDIRECTION
    void map_global_id_directly_to_edge_table();
#else
#endif

    void cutoff_too_narrow(float min_range_width);
    void cutoff_too_small(int min_node_count);
    void cutoff_too_deep(int new_max_depth);

    const std::vector<AttrTreeNode *> &get_all_unit_graphs() const {
        if (schema == NUMERICAL) {
            return tree->treenodes;
        } else {
            return hash->hashednodes;
        }
    }

    // select edges according to irangegraph skipping logic
    std::vector<IdxType> select_edge_infilter(
        IdxType pid,
        float myattr, float L_attr, float R_attr,
        VisitedSet &visited_set,
        std::function<bool(IdxType)> predicate,
        int max_edge_count,
        std::vector<int> *ne_sdnc=nullptr);
    // std::vector<IdxType> select_edge_infilter(  // mock iRangeGraph for Categorical data
    //     IdxType pid,
    //     float myattr, const std::vector<float> &vals,
    //     VisitedSet &visited_set,
    //     std::function<bool(IdxType)> predicate,
    //     int max_edge_count,
    //     std::vector<int> *ne_sdnc=nullptr);

    std::vector<IdxType> select_edge_infilter_prob(
        IdxType pid,
        float myattr, float L_attr, float R_attr,
        VisitedSet &visited_set,
        std::function<bool(IdxType)> predicate_one,
        std::function<bool(IdxType)> predicate,
        int current_step,
        int max_edge_count,
        unsigned int &prob_state,
        std::vector<int> *ne_sdnc=nullptr);
    // std::vector<IdxType> select_edge_infilter_prob(
    //     IdxType pid,
    //     float myattr, const std::vector<float> &vals,
    //     VisitedSet &visited_set,
    //     std::function<bool(IdxType)> predicate_one,
    //     std::function<bool(IdxType)> predicate,
    //     int current_step,
    //     int max_edge_count,
    //     unsigned int &prob_state,
    //     std::vector<int> *ne_sdnc=nullptr);

    static std::shared_ptr<irg_attr> build_or_load_graph_for_numerical_attr(
        std::vector<float> &attr, int attr_id, int depth_max, std::shared_ptr<ANNS::IStorage> base_storage, std::string idx_path,
        std::shared_ptr<ANNS::DistanceHandler> distance_handler, IdxType M, int ef_con, int num_threads, int fanout=2,
        UnitGraphBuildType unit_graph_build_type=UNIT_GRAPH_NSW);
    static std::shared_ptr<irg_attr> build_or_load_graph_for_hash_attr(
        std::vector<float> &attr, int attr_id, std::shared_ptr<ANNS::IStorage> base_storage, std::string idx_path,
        std::shared_ptr<ANNS::DistanceHandler> distance_handler, IdxType M, int ef_con, int num_threads,
        UnitGraphBuildType unit_graph_build_type=UNIT_GRAPH_NSW);

    void save_tree(const std::string& index_name);
    void load_tree(const std::string& index_name,
        std::shared_ptr<IStorage> base_storage,
        std::shared_ptr<DistanceHandler> distance_handler, 
        IdxType M, int ef_con);
    void save_hash(const std::string& index_name);
    void load_hash(const std::string& index_name,
        std::shared_ptr<IStorage> base_storage,
        std::shared_ptr<DistanceHandler> distance_handler, 
        IdxType M, int ef_con);

// private:
    enum { NUMERICAL, CATEGORICAL } schema;

    AttrSegmentTree *tree;
    AttrHashCollection *hash;

    int _attr_id;
    int _na;
    std::vector<float> *_attr;

    std::shared_ptr<IStorage> _base_storage;
    std::shared_ptr<DistanceHandler> _distance_handler;

    int _M, _ef_con;
    UnitGraphBuildType _unit_graph_build_type{UNIT_GRAPH_NSW};
};

}

#define ATTRIBUTE_TREE_INCLUDED
#endif
