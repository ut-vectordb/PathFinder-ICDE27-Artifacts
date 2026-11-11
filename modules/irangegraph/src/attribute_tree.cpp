#include "attribute_tree.h"
#include <iomanip>
#include <fstream>
#include <filesystem>

namespace ANNS {

using namespace std;
using pfi = AttrSegmentTree::pfi;

namespace {

string hnsw_upper_sidecar_path(const string &index_name, int node_id) {
    return index_name + "_hnsw_upper_node" + to_string(node_id) + ".txt";
}

void save_hnsw_upper_layers(const string &index_name, const vector<AttrTreeNode *> &nodes) {
    for (const auto *node : nodes) {
        if (node == nullptr || node->depth == 0 || !node->optional_HNSW_upper) {
            continue;
        }
        node->optional_HNSW_upper->save(hnsw_upper_sidecar_path(index_name, node->node_id));
    }
}

void load_hnsw_upper_layers(const string &index_name, const vector<AttrTreeNode *> &nodes) {
    for (auto *node : nodes) {
        if (node == nullptr || node->depth == 0) {
            continue;
        }
        string sidecar = hnsw_upper_sidecar_path(index_name, node->node_id);
        if (!std::filesystem::is_regular_file(sidecar)) {
            throw runtime_error("Missing HNSW upper-layer sidecar: " + sidecar);
        }
        node->optional_HNSW_upper = make_shared<HNSWUpperLayer>();
        node->optional_HNSW_upper->load(sidecar);
    }
}

}

void PrintTree(const AttrTreeNode* node, const std::string& prefix, bool isLast) {
    if (!node) return;
    // if (node->depth > 4) return;
    if (node->depth > 8) return;

    std::cout << prefix;
    std::cout << (isLast ? "└── " : "├── ");
    std::cout << "[" << node->lbound << ", " << node->rbound_open << ")"
            //   << " size=" << node->size << " nodeID=" << node->node_id << " depth=" << node->depth << "\n";
              << " size=" << node->size << " avgdeg=" << (double)node->size_e / node->size << " nodeID=" << node->node_id << " depth=" << node->depth << "\n";

    for (size_t i = 0; i < node->childs.size(); ++i) {
        PrintTree(node->childs[i], prefix + (isLast ? "    " : "│   "), i == node->childs.size() - 1);
    }
}

float AttrTreeNode::report_size_MB() {
    size_t total_bytes = 0;
    total_bytes += sizeof(this->childs[0]) * this->childs.size();
    // total_bytes += sizeof(this->edgelist_hashmap[0]) * this->edgelist_hashmap.size();
    // total_bytes += sizeof(this->vertex_table[0]) * this
    return total_bytes;
}

// Rewrite every edge destination in Graph::neighbors from packed internal IDs
// to global IDs, using the inverse of vertex_table (packed → global).
// Called once at build/load time, before map_global_id_directly_to_edge_table().
void AttrTreeNode::translate_edge_destination() {
    std::vector<IdxType> new_to_old = get_new_to_old();
    size_e = 0;
    for (int u = 0; u < size; u++) {
        size_e += g->neighbors[u].size();
        for (IdxType &k : g->neighbors[u]) {
            k = new_to_old.at(k);
        }
    }
    translate_HNSW_upper_layer();
}

void AttrTreeNode::translate_HNSW_upper_layer() {
    if (!optional_HNSW_upper) {
        return;
    }

    std::vector<IdxType> new_to_old = get_new_to_old();
    auto &upper = *optional_HNSW_upper;
    upper.enterpoint_node_ = new_to_old.at(upper.enterpoint_node_);

    std::unordered_map<IdxType, std::vector<IdxType>> translated;
    translated.reserve(upper.edgelist.size());
    for (auto &[local_node, data_all_layers] : upper.edgelist) {
        for (int level = upper.maxlevel_; level > 0; --level) {
            IdxType offset = upper.n_links_per_element_ * (level - 1);
            if (offset >= data_all_layers.size()) {
                continue;
            }
            IdxType count = data_all_layers[offset];
            for (IdxType j = 0; j < count; ++j) {
                data_all_layers[offset + 1 + j] = new_to_old.at(data_all_layers[offset + 1 + j]);
            }
        }
        translated.emplace(new_to_old.at(local_node), std::move(data_all_layers));
    }
    upper.edgelist = std::move(translated);
}

std::vector<IdxType> AttrTreeNode::get_new_to_old() {
    std::vector<IdxType> new_to_old(size);
    my_assert(vertex_table.size() == size);
    for (auto kv : vertex_table) {
        my_assert(kv.second < size && kv.second >= 0);
        new_to_old[kv.second] = kv.first;
    }
    return new_to_old;
}

// Move edge lists from Graph (packed ID space, edges already translated to
// global by translate_edge_destination) into ConcurrentEdgeMap keyed by
// global ID.  After this, Graph is freed — all subsequent access goes
// through edgelist_hashmap in global ID space.  vertex_table is not cleared
// here but is typically cleared by the caller once no longer needed.
void AttrTreeNode::map_global_id_directly_to_edge_table() {
    if (g == nullptr) {
        // this is the case of delegated root graph
        return;
    }

    my_assert(g != nullptr);

    this->size_e = 0;

    if (vertex_table.size() > 0) {
        // vertex_table present: use it to map packed internal_id → global_id
        edgelist_hashmap.reserve(g->_num_points);
        for (const auto& kv : vertex_table) {
            IdxType global_id = kv.first;
            IdxType internal_id = kv.second;
            std::vector<IdxType> &edgelist = g->neighbors[internal_id];
            size_e += edgelist.size();
            edgelist_hashmap.emplace(global_id, std::move(edgelist));
        }
    } else {
        edgelist_hashmap.reserve(g->_num_points);
        for (size_t i = 0; i < g->_num_points; i++) {
            std::vector<IdxType> &edgelist = g->neighbors[i];
            size_e += edgelist.size();
            edgelist_hashmap.emplace(static_cast<IdxType>(i), std::move(edgelist));
        }
    }

    // free g
    g.reset();
    g = nullptr;
}

void AttrTreeNode::populate_value_bounds(const vector<float> &global_attr, int na) {
    value_bounds.resize(na);
    float inf = numeric_limits<float>::infinity();
    for (int j = 0; j < na; j++) {
        value_bounds[j] = make_pair(inf, -inf);
    }
    for (auto i : vertex_table) {
        IdxType gid = i.first;
        for (int j = 0; j < na; j++) {
            float val = global_attr[gid * na + j];
            value_bounds[j].first = min(value_bounds[j].first, val);
            value_bounds[j].second = max(value_bounds[j].second, val);
        }
    }
}

void AttrSegmentTree::BuildTree(AttrTreeNode *u, const pfi *sorted, int depth_max) {
    using namespace std;

    if (u == nullptr)
        throw std::runtime_error("Tree node is a nullptr");

    // my_assert(sorted[0].first == u->lbound);
    my_assert(sorted[0].first >= u->lbound);
    my_assert(sorted[u->size - 1].first < u->rbound_open);

    treenodes.emplace_back(u);
    u->node_id = treenodes.size() - 1;

    max_depth = max(max_depth, u->depth);
    float L = u->lbound, R_open = u->rbound_open;
    if (L >= R_open)
        throw std::runtime_error("This shouldn't happen, size = " + to_string(u->size));

    // Build the vertex table: sorted[i].second is the global_id of the i-th
    // point in this node's sorted order; i becomes its packed internal_id.
    // This mapping is used later by translate_edge_destination() and
    // map_global_id_directly_to_edge_table(), then cleared after loading.
    int size = u->size;
    for (int i = 0; i < size; i++) {
        u->vertex_table[sorted[i].second] = i;
    }

    // conditions for not splitting any further
    if (sorted[0].first == sorted[size-1].first)
        return;
    if (size < size_threshold)
        return;

    if (u->depth >= depth_max)
        return;

    int mid = 0;
    float mid_value = 0;
    if (true) { // try to do equi-width
        int mid_wanted = size / 2;
        mid_value = sorted[mid_wanted].first;
        pfi pivot = std::make_pair(mid_value, -1);
        mid = std::lower_bound(sorted, sorted + size, pivot) - sorted;
        my_assert(sorted[mid].first == mid_value);

        if (mid == 0) {
            pivot.second = INT32_MAX;
            mid = std::upper_bound(sorted, sorted + size, pivot) - sorted;
            mid_value = sorted[mid].first;
        }
    } else {    // just divide by attribute
        mid_value = (L + R_open) / 2;
        pfi pivot = std::make_pair(mid_value, -1);
        mid = std::lower_bound(sorted, sorted + size, pivot) - sorted;
    }

    if (mid > 0) {
        u->childs.push_back(new AttrTreeNode(L, mid_value, u->depth + 1, mid));
        BuildTree(u->childs.back(), sorted, depth_max);
    }
    if (size - mid > 0) {
        u->childs.push_back(new AttrTreeNode(mid_value, R_open, u->depth + 1, size - mid));
        BuildTree(u->childs.back(), sorted + mid, depth_max);
    }
}

void AttrSegmentTree::BuildTree_multiway(AttrTreeNode *u, const pfi *sorted, int depth_max, int fanout) {
    using namespace std;

    if (u == nullptr)
        throw std::runtime_error("Tree node is a nullptr");

    // my_assert(sorted[0].first == u->lbound);
    my_assert(sorted[0].first >= u->lbound);
    my_assert(sorted[u->size - 1].first < u->rbound_open);

    treenodes.emplace_back(u);
    u->node_id = treenodes.size() - 1;

    max_depth = max(max_depth, u->depth);
    float L = u->lbound, R_open = u->rbound_open;
    if (L >= R_open)
        throw std::runtime_error("This shouldn't happen, size = " + to_string(u->size));

    // build the vertex table
    int size = u->size;
    for (int i = 0; i < size; i++) {
        u->vertex_table[sorted[i].second] = i;
    }

    // conditions for not splitting any further
    if (sorted[0].first == sorted[size-1].first)
        return;
    if (size < fanout)
        return;
    if (u->depth >= depth_max)
        return;

    int mid = 0;
    if (true) { // try to do equi-width
        int gap = size / fanout;
        int remaining = size % fanout;
        std::vector<int> dividers;
        dividers.push_back(0);
        for (int l = 0; l <= size - 1;) {
            int r = l + gap - 1;
            if (remaining > 0) {
                r++;
                remaining--;
            }
            r = std::min(r, size - 1);  // we want index range [l, r] in this section
            dividers.push_back(r+1);

            // adjust the index so that it cleanly divides the attr values
            // u->childs.push_back(new AttrTreeNode(L_attr, R_attr_open, u->depth + 1, r - l));
            // BuildTree___(u->childs.back(), sorted + l, depth_max);

            l = r + 1;
        }
        int n_segments = dividers.size() - 1;
        my_assert(n_segments == fanout);

        std::vector<float> dividing_values;
        std::vector<int> actual_dividers;
        actual_dividers.push_back(0);
        for (int i = 1; i < dividers.size() - 1; i++) {
            float v = sorted[dividers[i]].first;
            dividing_values.push_back(v);
            int lower = lower_bound(sorted, sorted + size, make_pair(v, 0)) - sorted;
            // std::upper_bound -- Searches for the first element in the partitioned range [first, last) which is ordered after `value`.
            int upper = upper_bound(sorted, sorted + size, make_pair(v, numeric_limits<int>::max())) - sorted;
            // pick one out of lower and upper that is closer to dividers[i]
            int diff1 = abs(lower - dividers[i]);
            int diff2 = abs(upper - dividers[i]);
            int choice = lower;
            if (diff1 <= diff2) {
                choice = lower;
            } else {
                choice = upper;
            }
            if (choice == actual_dividers.back()) {
                continue;
            }
            actual_dividers.push_back(choice);
        }
        if (size != actual_dividers.back())
            actual_dividers.push_back(size);

        // cout << "Dividers are: ";
        // for (int i = 0; i < actual_dividers.size(); i++) {
        //     cout << actual_dividers[i] << ", ";
        // }
        // cout << endl;

        for (int i = 0; i < actual_dividers.size() - 1; i++) {
            int l = actual_dividers[i];
            int r_open = actual_dividers[i+1];
            float L_attr = sorted[l].first;
            float R_attr_open;
            if (r_open == size) {
                R_attr_open = R_open;
            } else {
                R_attr_open = sorted[r_open].first;
            }
            // cout << "Recursing. " << L_attr << ", " << R_attr_open << ", " << R_open << ", " << l << ", " << r_open << endl;
            u->childs.push_back(new AttrTreeNode(L_attr, R_attr_open, u->depth + 1, r_open - l));
            BuildTree_multiway(u->childs.back(), sorted + l, depth_max, fanout);
        }
    } else {    // just divide by attribute
    }
}

// should we consider qr closed?
std::vector<AttrTreeNode *> AttrSegmentTree::range_filter(AttrTreeNode *u, float ql, float qr) {
    if (u->lbound >= ql && u->rbound_open <= qr)    // fully contained
        return {u};
    std::vector<AttrTreeNode *> res;
    if (u->lbound > qr || u->rbound_open < ql)     // no overlap at all
        return res;
    for (auto child : u->childs) {
        auto t = range_filter(child, ql, qr);
        while (t.size()) {
            res.emplace_back(t.back());
            t.pop_back();
        }
    }
    if (u->childs.size() == 0) {
        // the case where a range doesn't quite fit in the query, but has no children
        return {u};
    } else {
        return res;
    }
}

void irg_attr::save_tree(const std::string& index_name) {
    if (!tree->root->g) {
        std::cerr << "This tree probably has been translated to remove indirection, meaning graph object g is released." << std::endl;
        return;
    }

    std::ofstream out(index_name, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open output file for writing.\n";
        return;
    }

    // Write tree metadata
    out.write(reinterpret_cast<const char*>(&tree->size_threshold), sizeof(tree->size_threshold));

    int node_count = tree->treenodes.size();
    out.write(reinterpret_cast<const char*>(&node_count), sizeof(node_count));

    out.write(reinterpret_cast<const char*>(&tree->max_depth), sizeof(tree->max_depth));

    for (AttrTreeNode* node : tree->treenodes) {
        out.write(reinterpret_cast<const char*>(&node->lbound), sizeof(node->lbound));
        out.write(reinterpret_cast<const char*>(&node->rbound_open), sizeof(node->rbound_open));
        out.write(reinterpret_cast<const char*>(&node->depth), sizeof(node->depth));
        out.write(reinterpret_cast<const char*>(&node->size), sizeof(node->size));

        // Write number of children
        int child_count = node->childs.size();
        out.write(reinterpret_cast<const char*>(&child_count), sizeof(child_count));

        // Write vertex_table
        my_assert(node->vertex_table.size() == node->size);
        for (const auto& [k, v] : node->vertex_table) {
            out.write(reinterpret_cast<const char*>(&k), sizeof(k));
            out.write(reinterpret_cast<const char*>(&v), sizeof(v));
        }

        auto &graph = node->g;

        IdxType num_nodes = node->size;
        for (IdxType i = 0; i < num_nodes; ++i) {
            out.write(reinterpret_cast<const char*>(&i), sizeof(IdxType));

            const auto &nbrs = graph->neighbors[i];
            uint32_t num_neighbors = static_cast<uint32_t>(nbrs.size());
            out.write(reinterpret_cast<const char*>(&num_neighbors), sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(nbrs.data()), num_neighbors * sizeof(IdxType));
        }
    }

    out.close();

    save_hnsw_upper_layers(index_name, tree->treenodes);
}

void irg_attr::load_tree(const std::string& index_name,
        std::shared_ptr<IStorage> base_storage,
        std::shared_ptr<DistanceHandler> distance_handler, 
        IdxType M, int ef_con) {

    fs::path file_path(index_name);
    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Failed to open binary file for reading: " << file_path << std::endl;
        return;
    }

    _base_storage = base_storage;
    _distance_handler = distance_handler;
    _M = M;
    _ef_con = ef_con;

    tree = new AttrSegmentTree(0, 0, 0, 0);
    in.read(reinterpret_cast<char*>(&tree->size_threshold), sizeof(tree->size_threshold));

    int node_count = 0;
    in.read(reinterpret_cast<char*>(&node_count), sizeof(node_count));
    tree->treenodes.resize(node_count);

    in.read(reinterpret_cast<char*>(&tree->max_depth), sizeof(tree->max_depth));

    // Temporary store for child counts to reconstruct the tree later
    std::vector<int> child_counts(node_count);

    for (int i = 0; i < node_count; ++i) {
        float lbound, rbound_open;
        int depth, size;
        in.read(reinterpret_cast<char*>(&lbound), sizeof(float));
        in.read(reinterpret_cast<char*>(&rbound_open), sizeof(float));
        in.read(reinterpret_cast<char*>(&depth), sizeof(int));
        in.read(reinterpret_cast<char*>(&size), sizeof(int));

        AttrTreeNode* node = new AttrTreeNode(lbound, rbound_open, depth, size);
        node->vertex_table.clear();
        node->node_id = i;
        node->parent = nullptr;
        tree->treenodes[i] = node;

        // Read child count
        int child_count = 0;
        in.read(reinterpret_cast<char*>(&child_count), sizeof(int));
        child_counts[i] = child_count;

        // Read vertex_table
        for (int j = 0; j < node->size; ++j) {
            IdxType k, v;
            in.read(reinterpret_cast<char*>(&k), sizeof(IdxType));
            in.read(reinterpret_cast<char*>(&v), sizeof(IdxType));
            node->vertex_table[k] = v;
        }

        // Read graph
        node->g = std::make_shared<Graph>(size);
        size_t size_e = 0;
        for (IdxType j = 0; j < static_cast<IdxType>(size); ++j) {
            IdxType id;
            uint32_t num_neighbors;
            in.read(reinterpret_cast<char*>(&id), sizeof(IdxType));
            in.read(reinterpret_cast<char*>(&num_neighbors), sizeof(uint32_t));
            size_e += num_neighbors;
            node->g->neighbors[id].resize(num_neighbors);
            in.read(reinterpret_cast<char*>(node->g->neighbors[id].data()), num_neighbors * sizeof(IdxType));
        }

        // node->map_global_id_directly_to_edge_table();   // hopefully this brings down memory consumption

        node->size_e = size_e;
    }

    // Rebuild tree structure
    int cursor = 0;
    std::function<void(AttrTreeNode*, int&)> rebuild = [&](AttrTreeNode* u, int& cursor) {
        int my_index = cursor++;
        int count = child_counts[my_index];
        for (int j = 0; j < count; ++j) {
            AttrTreeNode* child = tree->treenodes[cursor];
            u->childs.push_back(child);
            child->parent = u;
            my_assert(child->depth == u->depth + 1);
            rebuild(child, cursor);
        }
    };

    tree->root = tree->treenodes[0];
    int start = 0;
    rebuild(tree->root, start);

    // Compute max_depth if needed
    tree->max_depth = 0;
    for (auto* node : tree->treenodes) {
        tree->max_depth = std::max(tree->max_depth, node->depth);
    }
    cout << "Sanity check: max depth = " << tree->max_depth << endl;
    for (auto* node : tree->treenodes) {
        my_assert(node->vertex_table.size() == node->size);
    }
    // PrintTree(tree->root);

    in.close();

    if (_unit_graph_build_type == UNIT_GRAPH_HNSW) {
        load_hnsw_upper_layers(index_name, tree->treenodes);
    }
}

void irg_attr::save_hash(const std::string& index_name) {
    if (!hash->hashednodes.size() || !hash->hashednodes[0]->g) {
        std::cerr << "This tree probably has been translated to remove indirection, meaning graph object g is released." << std::endl;
        return;
    }

    std::ofstream out(index_name, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open output file for writing.\n";
        return;
    }

    int node_count = hash->hash_table.size();
    out.write(reinterpret_cast<const char*>(&node_count), sizeof(node_count));
    out.write(reinterpret_cast<const char*>(&hash->min_val), sizeof(hash->min_val));
    out.write(reinterpret_cast<const char*>(&hash->max_val), sizeof(hash->max_val));

    for (auto [key, node] : hash->hash_table) {
        out.write(reinterpret_cast<const char*>(&key), sizeof(key));

        out.write(reinterpret_cast<const char*>(&node->lbound), sizeof(node->lbound));
        out.write(reinterpret_cast<const char*>(&node->rbound_open), sizeof(node->rbound_open));
        out.write(reinterpret_cast<const char*>(&node->depth), sizeof(node->depth));
        out.write(reinterpret_cast<const char*>(&node->size), sizeof(node->size));

        // Write number of children
        int child_count = node->childs.size();
        out.write(reinterpret_cast<const char*>(&child_count), sizeof(child_count));

        // Write vertex_table
        my_assert(node->vertex_table.size() == node->size);
        for (const auto& [k, v] : node->vertex_table) {
            out.write(reinterpret_cast<const char*>(&k), sizeof(k));
            out.write(reinterpret_cast<const char*>(&v), sizeof(v));
        }

        auto &graph = node->g;

        IdxType num_nodes = node->size;
        for (IdxType i = 0; i < num_nodes; ++i) {
            out.write(reinterpret_cast<const char*>(&i), sizeof(IdxType));

            const auto &nbrs = graph->neighbors[i];
            uint32_t num_neighbors = static_cast<uint32_t>(nbrs.size());
            out.write(reinterpret_cast<const char*>(&num_neighbors), sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(nbrs.data()), num_neighbors * sizeof(IdxType));
        }
    }

    out.close();

    save_hnsw_upper_layers(index_name, hash->hashednodes);
}

void irg_attr::load_hash(const std::string& index_name,
        std::shared_ptr<IStorage> base_storage,
        std::shared_ptr<DistanceHandler> distance_handler, 
        IdxType M, int ef_con) {

    fs::path file_path(index_name);
    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Failed to open binary file for reading: " << file_path << std::endl;
        return;
    }

    _base_storage = base_storage;
    _distance_handler = distance_handler;
    _M = M;
    _ef_con = ef_con;

    int node_count = 0;
    in.read(reinterpret_cast<char*>(&node_count), sizeof(node_count));

    hash = new AttrHashCollection();
    hash->all_keys.resize(node_count);
    hash->hashednodes.resize(node_count);
    hash->hash_table.clear();

    in.read(reinterpret_cast<char*>(&hash->min_val), sizeof(hash->min_val));
    in.read(reinterpret_cast<char*>(&hash->max_val), sizeof(hash->max_val));

    for (int i = 0; i < node_count; ++i) {
        float key;
        in.read(reinterpret_cast<char*>(&key), sizeof(float));

        float lbound, rbound_open;
        int depth, size;
        in.read(reinterpret_cast<char*>(&lbound), sizeof(float));
        in.read(reinterpret_cast<char*>(&rbound_open), sizeof(float));
        in.read(reinterpret_cast<char*>(&depth), sizeof(int));
        in.read(reinterpret_cast<char*>(&size), sizeof(int));

        // NOTE: all non_root nodes have depth = 1
        if (key < 0)
            depth = 0;  // this is the root
        else
            depth = 1;

        AttrTreeNode* node = new AttrTreeNode(lbound, rbound_open, depth, size);
        node->vertex_table.clear();
        node->node_id = (int)key;

        int child_count = 0;
        in.read(reinterpret_cast<char*>(&child_count), sizeof(int));
        // my_assert(child_count == 0);

        for (int j = 0; j < node->size; ++j) {
            IdxType k, v;
            in.read(reinterpret_cast<char*>(&k), sizeof(IdxType));
            in.read(reinterpret_cast<char*>(&v), sizeof(IdxType));
            node->vertex_table[k] = v;
        }

        // Read graph
        node->g = std::make_shared<Graph>(size);
        size_t size_e = 0;
        for (IdxType j = 0; j < static_cast<IdxType>(size); ++j) {
            IdxType id;
            uint32_t num_neighbors;
            in.read(reinterpret_cast<char*>(&id), sizeof(IdxType));
            in.read(reinterpret_cast<char*>(&num_neighbors), sizeof(uint32_t));
            size_e += num_neighbors;
            node->g->neighbors[id].resize(num_neighbors);
            in.read(reinterpret_cast<char*>(node->g->neighbors[id].data()), num_neighbors * sizeof(IdxType));
        }

        // node->map_global_id_directly_to_edge_table();   // hopefully this brings down memory consumption

        node->size_e = size_e;

        hash->all_keys[i] = key;
        hash->hashednodes[i] = node;
        hash->hash_table[key] = node;
    }

    hash->root_collection = hash->hash_table[-1];
    for (auto c : hash->hashednodes) {
        if (c != hash->root_collection) {
            hash->root_collection->childs.push_back(c);
            c->parent = hash->root_collection;
        } else {
            c->parent = nullptr;
        }
    }

    in.close();

    if (_unit_graph_build_type == UNIT_GRAPH_HNSW) {
        load_hnsw_upper_layers(index_name, hash->hashednodes);
    }
}

void AttrTreeNode::export_txt(string filename) {
    auto new_to_old = get_new_to_old();
    std::ofstream out(filename);
    for (IdxType i = 0; i < g->_num_points; i++) {
        out << new_to_old[i] << " ";
        for (auto& neighbor : g->neighbors[i])
            out << neighbor << " ";
        out << std::endl;
    }
    out.close();
}

void AttrTreeNode::export_edge_table(string filename, vector<float> &attr, int na) {
    auto new_to_old = get_new_to_old();
    std::ofstream out(filename);
    for (IdxType i = 0; i < g->_num_points; i++) {
        IdxType global_id = new_to_old[i];
        vector<float> row(na * 2);
        for (int a = 0; a < na; a++) {
            row[a] = attr[global_id * na + a];
        }
        for (auto& neighbor : g->neighbors[i]) {
            for (int a = 0; a < na; a++) {
                row[na + a] = attr[neighbor * na + a];
            }
            out << row[0];
            for (int j = 1; j < row.size(); j++) {
                out << "," << row[j];
            }
            out << std::endl;
        }
    }
    out.close();
}

void CheckTree(AttrTreeNode *u, std::vector<float> &attr, int na, int attr_id) {
    my_assert(u->vertex_table.size() == u->size);
    for (auto kv : u->vertex_table) {
        float a = attr[kv.first * na + attr_id];
        my_assert(u->lbound <= a && a < u->rbound_open);
    }
    int cnt = 0;
    #pragma omp parallel for schedule (static, 1024) reduction (+: cnt)
    for (int i = attr_id; i < attr.size(); i += na) {
        float a = attr[i];
        if (u->lbound <= a && a < u->rbound_open)
            ++cnt;
    }
    my_assert(cnt == u->size);
    for (auto c : u->childs) {
        CheckTree(c, attr, na, attr_id);
    }
}

void irg_attr::build(std::shared_ptr<IStorage> base_storage,
        std::vector<float> &attr,
        int attr_id,
        int depth_max,
        std::shared_ptr<DistanceHandler> distance_handler,
        IdxType M, int ef_con, int n_threads,
        int fanout,
        UnitGraphBuildType unit_graph_build_type) {

    _attr_id = attr_id;
    _base_storage = base_storage;
    _attr = &attr;
    _distance_handler = distance_handler;
    _M = M;
    _ef_con = ef_con;
    _unit_graph_build_type = unit_graph_build_type;

    int size = base_storage->get_num_points();
    int na = attr.size() / size;
    my_assert(na * size == attr.size());

    pfi *sorted = new pfi[size];
    // FIXME: this can also be parallel
    for (int i = 0; i < size; i++) {
        sorted[i].first = attr[i * na + attr_id];
        sorted[i].second = i;
    }
    sort(sorted, sorted + size);    // FIXME: use parallel sort or we'd be stupidly waiting on a single thread

    tree = nullptr; hash = nullptr;

    // bool skip_root = (attr_id != 0);
    bool skip_root = true;

    if (schema == NUMERICAL) {
        float l = sorted[0].first;
        float r = std::max(sorted[size-1].first + 1, std::nextafter(sorted[size-1].first, INFINITY));

        tree = new AttrSegmentTree(l, r, size, 0);  // 0: size threshold
        if (fanout == 2) {
            tree->BuildTree(tree->root, sorted, depth_max);
        } else {
            tree->BuildTree_multiway(tree->root, sorted, depth_max, fanout);
        }
        tree->root->size_e = 0;
        PrintTree(tree->root);
        delete [] sorted;
        // CheckTree(tree->root, attr, na, attr_id);

        if (depth_max < 0) {
            // we just leave the root node empty to save time
            tree->root->g = make_shared<Graph>(tree->root->size);
            cout << "Skip building the graph since depth_max < 0 ..." << endl;
        } else {
            // build the actual graphs
            std::vector<AttrTreeNode*> flat_nodes;
            flat_nodes = tree->treenodes;

            // Sort nodes by ascending layer
            std::sort(flat_nodes.begin(), flat_nodes.end(),
                    [](AttrTreeNode* a, AttrTreeNode* b) { return a->depth < b->depth; });

            for (auto u : flat_nodes) {
                if (skip_root && u->depth == 0) {    // skip root construction
                    u->g = make_shared<Graph>(u->size);
                    u->size_e = 0;
                    cout << "Skip building root since it's not attribute id 0..." << endl;
                } else if (u->depth <= 2) {
                    build_node(u, n_threads, false);
                    // cout << "Done building one node on depth " << u->depth << " ..." << endl;
                } else {
                    build_node(u, n_threads, false);
                }
                u->translate_edge_destination();
            }
        }
    } else {
        hash = new AttrHashCollection();
        hash->BuildTable(sorted, size);

        hash->root_collection = hash->hash_table[-1];
        for (auto c : hash->hashednodes) {
            if (c != hash->root_collection) {
                hash->root_collection->childs.push_back(c);
                c->parent = hash->root_collection;
            } else {
                c->parent = nullptr;
            }
        }

        std::vector<AttrTreeNode*> flat_nodes;
        flat_nodes = hash->hashednodes;

        // for (auto u : flat_nodes) {
        for (int i = 0; i < flat_nodes.size(); i++) {
            auto u = flat_nodes[flat_nodes.size() - i - 1];
            if (skip_root && u->depth == 0) {
                u->g = make_shared<Graph>(u->size);
                u->size_e = 0;
                cout << "Skip building root since it's not attribute id 0..." << endl;
                continue;
            }

            // int category_val = (int)u->lbound;
            // if (category_val == 1 || category_val == 2) {
            //     ;
            // } else {
            //     cout << "skipping category " << category_val << " ..." << endl;
            //     u->g = make_shared<Graph>(u->size);
            //     continue;
            // }

            // cout << "Start building one node for category " << int(u->lbound) << " ..." << endl;
            auto t0 = std::chrono::high_resolution_clock::now();
            build_node(u, n_threads, false);
            auto t1 = std::chrono::high_resolution_clock::now();
            // cout << "Done building one node for category " << int(u->lbound) << " ... Time cost (s) = " << std::chrono::duration<float>(t1 - t0).count() << endl;
            u->translate_edge_destination();    // populates "size_e"
            // cout << "Metadata: size = " << u->size << ", size_e = " << u->size_e << endl;
            // cout << "  avgdeg = " << (double)u->size_e / u->size << endl;
        }
    }
}

void irg_attr::build_print_stats(std::shared_ptr<IStorage> base_storage,
        std::vector<float> &attr,
        int attr_id,
        int depth_max,
        std::shared_ptr<DistanceHandler> distance_handler,
        IdxType M, int ef_con, int n_threads) {

    throw std::runtime_error("This function needs updating since multiway_build");
    _attr_id = attr_id;
    _base_storage = base_storage;
    _attr = &attr;
    _distance_handler = distance_handler;
    _M = M;
    _ef_con = ef_con;

    int size = base_storage->get_num_points();
    int na = attr.size() / size;
    my_assert(na * size == attr.size());

    pfi *sorted = new pfi[size];
    for (int i = 0; i < size; i++) {
        sorted[i].first = attr[i * na + attr_id];
        sorted[i].second = i;
    }
    sort(sorted, sorted + size);

    tree = nullptr; hash = nullptr;

    auto t0 = std::chrono::high_resolution_clock::now();

    if (schema == NUMERICAL) {
        float l = sorted[0].first;
        float r = std::max(sorted[size-1].first + 1, std::nextafter(sorted[size-1].first, INFINITY));

        tree = new AttrSegmentTree(l, r, size, 0);  // 0: size threshold
        tree->BuildTree(tree->root, sorted, depth_max);
        tree->root->size_e = 0;
        delete [] sorted;

        my_assert(depth_max >= 0);

        // build the actual graphs
        std::vector<AttrTreeNode*> flat_nodes;
        flat_nodes = tree->treenodes;

        // Sort nodes by ascending layer
        std::sort(flat_nodes.begin(), flat_nodes.end(),
                [](AttrTreeNode* a, AttrTreeNode* b) { return a->depth < b->depth; });

        int now_depth = 0;
        for (auto u : flat_nodes) {
            if (u->depth > now_depth) {
                auto t1 = std::chrono::high_resolution_clock::now();
                cout << "Done building all nodes in depth " << now_depth << ", time from beginning = " << std::chrono::duration<float>(t1 - t0).count() << endl;
                now_depth = u->depth;
            }
            bool verbose = false;
            build_node(u, n_threads, verbose);
            u->translate_edge_destination();
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        cout << "Done building all nodes in depth " << now_depth << ", time from beginning = " << std::chrono::duration<float>(t1 - t0).count() << endl;
    } else {
        hash = new AttrHashCollection();
        hash->BuildTable(sorted, size);

        std::vector<AttrTreeNode*> flat_nodes;
        flat_nodes = hash->hashednodes;

        // for (auto u : flat_nodes) {
        for (int i = 0; i < flat_nodes.size(); i++) {
            auto u = flat_nodes[flat_nodes.size() - i - 1];

            cout << "Start building one node for category " << int(u->lbound) << " ..." << endl;
            auto t0 = std::chrono::high_resolution_clock::now();
            build_node(u, n_threads, false);
            auto t1 = std::chrono::high_resolution_clock::now();
            cout << "Done building one node for category " << int(u->lbound) << " ... Time cost (s) = " << std::chrono::duration<float>(t1 - t0).count() << endl;
            u->translate_edge_destination();    // populates "size_e"
            cout << "Metadata: size = " << u->size << ", size_e = " << u->size_e << endl;
            cout << "  avgdeg = " << (double)u->size_e / u->size << endl;
        }
    }
}

std::vector<IdxType> irg_attr::entrypoint_selection(float L_attr, float R_attr, int n_entry) {
    // sample some nodes within each unit graph.
    // use the "keys" in the vertex table
    if (this->schema == CATEGORICAL) {
        my_assert(std::isinf(L_attr) || L_attr < hash->min_val);
        my_assert(std::isinf(R_attr) || R_attr > hash->max_val);
        return entrypoint_selection(hash->all_keys, n_entry);
    }

    std::vector<IdxType> entries;
    vector<AttrTreeNode *> nodes = this->tree->range_filter(this->tree->root, L_attr, R_attr);
    for (auto node : nodes) {
        entries.push_back(node->vertex_table.begin()->first);
        // for (auto &global_id_internal_id : node->vertex_table) {
        //     IdxType global_id = global_id_internal_id.first;
        //     float attr_value = _attr->at(global_id * _na + _attr_id);
        //     if (L_attr <= attr_value && attr_value < R_attr) {
        //         entries.push_back(global_id);
        //     }
        // }
    }
    return entries;
}

std::vector<IdxType> irg_attr::entrypoint_selection(const vector<float> &vals, int n_entry) {
    // sample some nodes within each unit graph.
    // use the "keys" in the vertex table

    my_assert(schema == CATEGORICAL);

    std::vector<IdxType> entries;
    for (float v : vals) {
        auto &vtxtable = hash->hash_table[(int)v]->vertex_table;
        entries.push_back(vtxtable.begin()->first);
    }
    return entries;
}

#ifdef REMOVE_INDIRECTION
void irg_attr::map_global_id_directly_to_edge_table() {
    std::vector<AttrTreeNode*> *flat_nodes;
    if (schema == NUMERICAL) {
        flat_nodes = &tree->treenodes;
    } else {
        flat_nodes = &hash->hashednodes;
    }

    #pragma omp parallel for
    for (auto u : *flat_nodes) {
        if (u)
            u->map_global_id_directly_to_edge_table();
    }
}
#else
#endif

void AttrSegmentTree::cutoff_too_narrow(float min_range_width) {
    // after we delete a whole lot of unit graphs, we have to maintain this->treenodes and this->max_depth
    this->max_depth = 0;

    std::function<void(AttrTreeNode *)> free_whole_subtree = [&](AttrTreeNode *node) {
        for (auto c : node->childs) {
            free_whole_subtree(c);
        }
        this->treenodes[node->node_id] = nullptr;
        delete node;
    };

    std::function<void(AttrTreeNode *)> dfs = [&](AttrTreeNode *node) {
        if (node == nullptr) return;
        if (node->childs.size() == 0) return;

        this->max_depth = max(this->max_depth, node->depth);

        for (auto c : node->childs) {
            dfs(c);
        }

        if (node->rbound_open - node->lbound < min_range_width) {
            for (auto c : node->childs) {
                free_whole_subtree(c);
            }
            node->childs.clear();
        }
    };

    dfs(root);
}

void AttrSegmentTree::cutoff_too_small(int min_node_count) {
    // after we delete a whole lot of unit graphs, we have to maintain this->treenodes and this->max_depth
    this->max_depth = 0;

    std::function<void(AttrTreeNode *)> free_whole_subtree = [&](AttrTreeNode *node) {
        for (auto c : node->childs) {
            free_whole_subtree(c);
        }
        this->treenodes[node->node_id] = nullptr;
        delete node;
    };

    std::function<void(AttrTreeNode *)> dfs = [&](AttrTreeNode *node) {
        if (node == nullptr) return;

        this->max_depth = max(this->max_depth, node->depth);
        if (node->childs.size() == 0) return;

        vector<AttrTreeNode *> new_childs;
        for (auto c : node->childs) {
            if (c->size < min_node_count) {
                free_whole_subtree(c);
            } else {
                dfs(c);
                new_childs.push_back(c);
            }
        }
        node->childs = std::move(new_childs);
    };

    dfs(root);
}

void AttrSegmentTree::cutoff_too_deep(int new_max_depth) {
    this->max_depth = new_max_depth;

    std::function<void(AttrTreeNode *)> free_whole_subtree = [&](AttrTreeNode *node) {
        for (auto c : node->childs) {
            free_whole_subtree(c);
        }
        // printf("Freeing subtree on depth %d, id %d\n", (int)node->depth, (int)node->node_id);
        this->treenodes[node->node_id] = nullptr;
        delete node;
    };

    bool found_required_depth = false;

    std::function<void(AttrTreeNode *)> dfs = [&](AttrTreeNode *node) {
        if (node == nullptr) return;

        if (node->depth == new_max_depth) {
            found_required_depth = true;
            // printf("Reached wanted max depth %d, id %d\n", (int)node->depth, (int)node->node_id);
            for (auto c : node->childs) {
                free_whole_subtree(c);
            }
            node->childs.clear();
        } else {
            for (auto c : node->childs) {
                dfs(c);
            }
        }
    };

    dfs(root);

    if (!found_required_depth) {
        throw std::runtime_error("The input tree is not deep enough!");
    }
}

void irg_attr::cutoff_too_narrow(float min_range_width) {
    if (schema == NUMERICAL) {
        tree->cutoff_too_narrow(min_range_width);
    } else {
        ;   // do nothing
    }
}

void irg_attr::cutoff_too_small(int min_node_count) {
    if (schema == NUMERICAL) {
        tree->cutoff_too_small(min_node_count);
    } else {
        ;   // do nothing
    }
}

void irg_attr::cutoff_too_deep(int new_max_depth) {
    if (schema == NUMERICAL) {
        tree->cutoff_too_deep(new_max_depth);
    } else {
        ;   // do nothing
    }
}

std::vector<IdxType> irg_attr::select_edge_infilter(
        IdxType pid,
        float myattr, float L_attr, float R_attr,
        VisitedSet &visited_set,
        std::function<bool(IdxType)> predicate,
        int edge_limit,
        vector<int> *ne_sdnc) {

    my_assert(this->schema == NUMERICAL);

    int scanned=0, dup=0, notpass=0, chosen=0;

    std::vector<IdxType> selected_edges;
    selected_edges.reserve(edge_limit);

    AttrTreeNode *cur_node = nullptr, *nxt_node = tree->root;

    auto GetOverlap = [](float l, float r, float ql, float qr) {
        float L = std::max(l, ql), R = std::min(r, qr);
        return R - L;
    };
    auto find_child_containing = [&](AttrTreeNode *cur_node, float myattr) -> AttrTreeNode * {
        if (cur_node == nullptr) return nullptr;
        if (cur_node->childs.size() == 0) return nullptr;
        for (auto c : cur_node->childs)
            if (c->lbound <= myattr && myattr < c->rbound_open)
                return c;
        return nullptr;
    };

    bool done = false;

    do {
        cur_node = nxt_node;
        if (cur_node == nullptr) break;

        do {
            if (cur_node->childs.size() == 0) {
                nxt_node = nullptr; break;
            } else {
                nxt_node = find_child_containing(cur_node, myattr);

                if (nxt_node == nullptr) {
                    // If no child contains myattr, we are done
                    break;
                }

                if (GetOverlap(cur_node->lbound, cur_node->rbound_open, L_attr, R_attr)
                 == GetOverlap(nxt_node->lbound, nxt_node->rbound_open, L_attr, R_attr)) {
                    cur_node = nxt_node;    // move one layer at a time
                    continue;
                } else {
                    break;  // now cur and nxt are 1 layer apart
                }
            }
        } while (true);

        const vector<IdxType> *edgelist;
        if (cur_node->delegate_root) {
            edgelist = &cur_node->delegate_root->edgelist_hashmap.at_unsafe(pid);
        } else {
            edgelist = &cur_node->edgelist_hashmap.at_unsafe(pid);
        }
        for (IdxType neighborId : *edgelist) {
            scanned++;
            if (visited_set.check(neighborId)) {
                dup++;
                continue;
            }
            visited_set.set(neighborId);
            if (!predicate(neighborId)) {
                notpass++;
                continue;
            }
            chosen++;
            selected_edges.emplace_back(neighborId);
            if (selected_edges.size() >= edge_limit) {
                done = true;
                break;
            }
        }
    } while (!done && (cur_node->lbound <= L_attr || cur_node->rbound_open > R_attr));

    if (ne_sdnc) {
        ne_sdnc->resize(4);
        (*ne_sdnc)[0] = scanned;
        (*ne_sdnc)[1] = dup;
        (*ne_sdnc)[2] = notpass;
        (*ne_sdnc)[3] = chosen;
    }

    return selected_edges;
}

static int ProbFunc(int x, unsigned int &rand_state) {
    // return 1;   // purepost

    if (x >= 20)
        return 0;
    double randNum = (double)rand_r(&rand_state) / RAND_MAX;
    // if (probability.size() == 0) {
    //     setprob();
    // }
    // return randNum < probability[x] ? 1 : 0;
    return randNum < (1 / (1 + exp(x))) ? 1 : 0;
}

std::vector<IdxType> irg_attr::select_edge_infilter_prob(
        IdxType pid,
        float myattr, float L_attr, float R_attr,
        VisitedSet &visited_set,
        std::function<bool(IdxType)> predicate_one,
        std::function<bool(IdxType)> predicate,
        int current_step,   // interpolation parameter
        int edge_limit,
        unsigned int &prob_state,
        std::vector<int> *ne_sdnc) {

    my_assert(this->schema == NUMERICAL);

    int scanned=0, dup=0, notpass=0, chosen=0;

    std::vector<IdxType> selected_edges;
    selected_edges.reserve(edge_limit);

    AttrTreeNode *cur_node = nullptr, *nxt_node = tree->root;

    auto GetOverlap = [](float l, float r, float ql, float qr) {
        float L = std::max(l, ql), R = std::min(r, qr);
        return R - L;
    };
    auto find_child_containing = [&](AttrTreeNode *cur_node, float myattr) -> AttrTreeNode * {
        if (cur_node == nullptr) return nullptr;
        if (cur_node->childs.size() == 0) return nullptr;
        for (auto c : cur_node->childs)
            if (c->lbound <= myattr && myattr < c->rbound_open)
                return c;
        return nullptr;
    };

    bool done = false;

    do {
        cur_node = nxt_node;
        if (cur_node == nullptr) break;

        do {
            if (cur_node->childs.size() == 0) {
                nxt_node = nullptr; break;
            } else {
                nxt_node = find_child_containing(cur_node, myattr);

                if (nxt_node == nullptr) {
                    // If no child contains myattr, we are done
                    break;
                }

                if (GetOverlap(cur_node->lbound, cur_node->rbound_open, L_attr, R_attr)
                 == GetOverlap(nxt_node->lbound, nxt_node->rbound_open, L_attr, R_attr)) {
                    cur_node = nxt_node;    // move one layer at a time
                    continue;
                } else {
                    break;  // now cur and nxt are 1 layer apart
                }
            }
        } while (true);

        const vector<IdxType> *edgelist;
        if (cur_node->delegate_root) {
            edgelist = &cur_node->delegate_root->edgelist_hashmap.at_unsafe(pid);
        } else {
            edgelist = &cur_node->edgelist_hashmap.at_unsafe(pid);
        }

        for (IdxType neighborId : *edgelist) {
                    // int neighborId = *(data + j);
                    // if (neighborId < ql || neighborId > qr)
                    //     continue;
                    // int prob = 1;
                    // int next_step = current_step + 1;
                    // bool inrange = CheckInQueryRange(neighborId, queryrange);
                    // if (!inrange)
                    //     prob = ProbFunc(next_step);
                    // if (!prob)
                    //     continue;

                    // selected_edges.emplace_back(neighborId, inrange);

            scanned++;
#ifdef FIX_THEIR_BUG
            if (visited_set.check(neighborId)) {
                continue;
            }
            visited_set.set(neighborId);
#endif
            if (!predicate_one(neighborId))
                continue;
            int prob = 1;
            int next_step = current_step + 1;
            bool inrange = predicate(neighborId);
            if (!inrange) {
                prob = ProbFunc(next_step, prob_state);
            }
            if (!prob) {
                notpass++;
                continue;
            }
            chosen++;
            selected_edges.emplace_back(neighborId);
            if (selected_edges.size() >= edge_limit) {
                done = true;
                break;
            }
        }
    } while (!done && (cur_node->lbound <= L_attr || cur_node->rbound_open > R_attr));

    if (ne_sdnc) {
        ne_sdnc->resize(4);
        (*ne_sdnc)[0] = scanned;
        (*ne_sdnc)[1] = dup;
        (*ne_sdnc)[2] = notpass;
        (*ne_sdnc)[3] = chosen;
    }

    return selected_edges;
}

void irg_attr::build_node(AttrTreeNode *u, int n_threads, bool verbose) {
    // will need to use the vertex table in reverse...
    vector<IdxType> new_to_old = u->get_new_to_old();
    std::shared_ptr<IStorage> my_storage;
    auto dtype = _base_storage->get_data_type();
    if (dtype == DataType::FLOAT)
        my_storage = std::make_shared<Storage<float>>(_base_storage, new_to_old);
    else if (dtype == DataType::UINT8)
        my_storage = std::make_shared<Storage<uint8_t>>(_base_storage, new_to_old);
    u->g = make_shared<Graph>(u->size);
    Hnsw tmp_index(verbose);
    if (_unit_graph_build_type == UNIT_GRAPH_HNSW) {
        bool randomize = false;
        bool merge_upper_layers = false;
        tmp_index.build_HNSWLib(my_storage, _distance_handler, u->g, _M, _ef_con, n_threads, randomize, merge_upper_layers);
        u->optional_HNSW_upper = tmp_index._upper_layer;
    } else {
        tmp_index.build_NSW(my_storage, _distance_handler, u->g, _M, _ef_con, n_threads);
    }

    my_storage->clean();
}

std::shared_ptr<irg_attr> irg_attr::build_or_load_graph_for_numerical_attr(
        std::vector<float> &attr, int attr_id, int depth_max, std::shared_ptr<ANNS::IStorage> base_storage, std::string idx_path,
        std::shared_ptr<ANNS::DistanceHandler> distance_handler, IdxType M, int ef_con, int num_threads, int fanout,
        UnitGraphBuildType unit_graph_build_type) {

    size_t nb = base_storage->get_num_points();
    size_t na = attr.size() / nb;  // number of attributes

    auto index = std::make_shared<irg_attr>();
    index->schema = NUMERICAL;
    index->_unit_graph_build_type = unit_graph_build_type;

    // bool skip_root = (attr_id != 0);
    bool skip_root = true;  // root graph must be processed separately

    if (std::filesystem::is_regular_file(idx_path)) {
        index->load_tree(idx_path, base_storage, distance_handler, M, ef_con);
    } else {
        index->build(base_storage, attr, attr_id, depth_max, distance_handler, M, ef_con, num_threads, fanout, unit_graph_build_type);
        index->save_tree(idx_path);
    }

    if (skip_root) {
        index->tree->root->g.reset();
        // index->tree->root->vertex_table.clear(); // required by "populate value bounds"
        index->tree->root->edgelist_hashmap.clear();
    }

    return index;
}

std::shared_ptr<irg_attr> irg_attr::build_or_load_graph_for_hash_attr(
        std::vector<float> &attr, int attr_id, std::shared_ptr<ANNS::IStorage> base_storage, std::string idx_path,
        std::shared_ptr<ANNS::DistanceHandler> distance_handler, IdxType M, int ef_con, int num_threads,
        UnitGraphBuildType unit_graph_build_type) {

    size_t nb = base_storage->get_num_points();
    size_t na = attr.size() / nb;  // number of attributes

    auto index = std::make_shared<irg_attr>();
    index->schema = CATEGORICAL;
    index->_unit_graph_build_type = unit_graph_build_type;
    idx_path += "_hash";

    // bool skip_root = (attr_id != 0);
    bool skip_root = true;  // root graph must be processed separately

    if (std::filesystem::is_regular_file(idx_path)) {
        index->load_hash(idx_path, base_storage, distance_handler, M, ef_con);
    } else {
        index->build(base_storage, attr, attr_id, 100, distance_handler, M, ef_con, num_threads, 2, unit_graph_build_type);
        index->save_hash(idx_path);
    }

    if (skip_root) {
        index->hash->root_collection->g.reset();
        // index->hash->root_collection->vertex_table.clear(); // required by "populate value bounds"
        index->hash->root_collection->edgelist_hashmap.clear();
    }

    return index;
}

void AttrHashCollection::BuildTable(const pfi *sorted, int total_size) {
    using namespace std;

    min_val = (int)sorted[0].first;
    max_val = (int)sorted[total_size - 1].first;

    hash_table.clear();
    all_keys.clear();
    hashednodes.clear();

    int section_start = 0;
    int section_rear = 0;

    for (int i = 1; i <= total_size; i++) {
        if (i == total_size || sorted[i].first != sorted[i-1].first) {
            section_rear = i-1;

            int section_val = (int)sorted[i-1].first;
            int section_size = section_rear - section_start + 1;
            int depth = 1;
            my_assert(section_size > 0);

            min_val = min(min_val, section_val);
            max_val = max(max_val, section_val);

            float l = section_val; float r_open = l + 1;

            auto u = new AttrTreeNode(l, r_open, depth, section_size);
            u->node_id = section_val;

            // build the vertex table
            for (int j = section_start; j < section_rear + 1; j++) {
                u->vertex_table[sorted[j].second] = j - section_start;
            }

            hash_table[section_val] = u;
            all_keys.push_back(section_val);
            hashednodes.push_back(u);

            section_start = i;
        }
    }

    {
        int section_size = total_size;
        int section_val = -1;
        int depth = 0;
        auto u = new AttrTreeNode(min_val, max_val + 1, depth, section_size);
        u->node_id = section_val;

        // build the vertex table
        for (int j = 0; j < total_size; ++j) {
            u->vertex_table[sorted[j].second] = j;
        }

        hash_table[section_val] = u;
        all_keys.push_back(section_val);
        for (auto c : hashednodes) {
            u->childs.push_back(c);
        }
        hashednodes.insert(hashednodes.begin(), u);
    }
}

} // namespace ANNS
