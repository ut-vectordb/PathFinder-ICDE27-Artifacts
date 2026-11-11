#include "irangegraph.h"
#include <omp.h>
#include <filesystem>

using namespace ANNS;
namespace fs = std::filesystem;

void iRG::build(std::shared_ptr<IStorage> base_storage,
        std::shared_ptr<DistanceHandler> distance_handler, 
        IdxType M, int ef_con, int n_threads) {

    _base_storage = base_storage;
    _distance_handler = distance_handler;
    _M = M;
    _ef_con = ef_con;
    _num_threads = n_threads;

    int n_pts = _base_storage->get_num_points();
    int n_dim = _base_storage->get_dim();

    tree = new iRangeGraph::SegmentTree(n_pts);
    tree->BuildTree(tree->root);

    std::vector<std::vector<iRangeGraph::TreeNode *>> level_nodes;
    level_nodes.resize(tree->max_depth + 1);
    for (auto node : tree->treenodes) {
        level_nodes[node->depth].emplace_back(node);
        indices.emplace_back(false);
        graphs.emplace_back(nullptr);
    }

    omp_set_num_threads(_num_threads);

    // Flatten all nodes across all layers into a single list
    std::vector<iRangeGraph::TreeNode*> flat_nodes;
    flat_nodes.reserve(tree->treenodes.size());

    for (int layer = 0; layer <= tree->max_depth; ++layer) {
        flat_nodes.insert(flat_nodes.end(), level_nodes[layer].begin(), level_nodes[layer].end());
    }

    // Sort nodes by ascending layer
    std::sort(flat_nodes.begin(), flat_nodes.end(),
            [](iRangeGraph::TreeNode* a, iRangeGraph::TreeNode* b) {
                return a->depth < b->depth;
            });

    if (_verbose) {
        std::cout << "building " << flat_nodes.size() << " nodes in parallel..." << std::endl;
    }

    int total_n = flat_nodes.size();
    int interval = total_n / 100000;
    if (interval == 0) interval = 1;

    // n_threads = 64;
    // int i = 10;
    // auto u = flat_nodes[i];
    // int uid = u->node_id;
    // graphs[uid] = std::make_shared<Graph>(u->rbound - u->lbound + 1);
    // auto range_storage = create_storage(_base_storage, u->lbound, u->rbound + 1);
    // std::cout << "Building #" << i << " using " << n_threads << " threads, n=" << u->rbound - u->lbound + 1 << std::endl;
    // indices[uid].build_NSW(range_storage, _distance_handler, graphs[uid], _M, _ef_con, n_threads);
    // indices[uid].statistics();
    // abort();

    // for the largest indices, we build them each in parallel
    int cutoff = total_n / 50;

    for (int i = 0; i < cutoff; i++) {
        auto u = flat_nodes[i];

        int uid = u->node_id;

        graphs[uid] = std::make_shared<Graph>(u->rbound - u->lbound + 1);
        auto range_storage = create_storage(_base_storage, u->lbound, u->rbound + 1);

        std::cout << "Building #" << i << " using " << n_threads << " threads" << std::endl;
        indices[uid].fixed_alpha = 1.0;
        indices[uid].build_NSW(range_storage, _distance_handler, graphs[uid], _M, _ef_con, n_threads);

        if (i < 10)
            indices[uid].statistics();
    }

    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = cutoff; i < flat_nodes.size(); ++i) {
        // if (i % interval == 0) {
        //     std::cout << "[" << i << "/" << total_n << "] " << 100.00 * i / total_n << "%" << std::endl;
        // }
        build_node(flat_nodes[i]);
    }
}

void iRG::build_node(iRangeGraph::TreeNode *u) {
    int uid = u->node_id;

    graphs[uid] = std::make_shared<Graph>(u->rbound - u->lbound + 1);
    auto range_storage = create_storage(_base_storage, u->lbound, u->rbound + 1);

    indices[uid].build_NSW(range_storage, _distance_handler, graphs[uid], _M, _ef_con, 1);
}

std::vector<IdxType> iRG::select_edge(IdxType pid, IdxType L_closed, IdxType R_closed, VisitedSet &visited_set) {

    int ql = L_closed, qr = R_closed;

    // NOTE: this function has a few tricks, including checking the visited set.
    int edge_limit = _M;
    std::vector<IdxType> selected_edges;
    selected_edges.reserve(_M);

    iRangeGraph::TreeNode *cur_node = nullptr, *nxt_node = tree->root;

    auto GetOverlap = [](int l, int r, int ql, int qr) {
        int L = std::max(l, ql);
        int R = std::min(r, qr);
        return R - L + 1;
    };

    do {
        cur_node = nxt_node;
        if (cur_node == nullptr) break;

        do {
            if (cur_node->childs.size() == 0) {
                nxt_node = nullptr; break;
            } else {
                nxt_node = find_child_containing(cur_node, pid);
                // the proper way to skip layer is to do it here.
                auto shouldskip = [](int d) {
                    // return (d < 10);
                    // return (d > 6);
                    return false;
                };
                while (shouldskip(nxt_node->depth)) {
                    if (nxt_node->childs.size() == 0) { break; }    // inside loop break
                    nxt_node = find_child_containing(nxt_node, pid);
                }
                if (nxt_node->childs.size() == 0) { nxt_node = nullptr; break; }

                if (GetOverlap(cur_node->lbound, cur_node->rbound, ql, qr) == GetOverlap(nxt_node->lbound, nxt_node->rbound, ql, qr)) {
                    cur_node = nxt_node;
                    continue;
                } else {
                    break;
                }
            }
        } while (true);  // go down the tree until the overlap isn't the same

        if (cur_node->depth == tree->max_depth) continue;
        // if (cur_node->depth == 1) continue;

        auto &edgelist = graphs[cur_node->node_id]->neighbors[pid - cur_node->lbound];

        for (int k : edgelist) {
            int neighborId = k + cur_node->lbound;
            if (neighborId < ql || neighborId > qr)
                continue;   // NOTE: Not having to access attr makes this faster... slightly
            if (visited_set.check(neighborId))
                continue;   // NOTE: performance test reveals this has no real impact on perf
            selected_edges.emplace_back(neighborId);
            if (selected_edges.size() == edge_limit)
                return selected_edges;
        }
    } while (cur_node->lbound < ql || cur_node->rbound > qr);

    return selected_edges;
}

std::vector<IdxType> iRG::select_edge_by_attr(
        IdxType pid, IdxType L_closed, IdxType R_closed,
        float *attrs, float L_attr, float R_attr,
        VisitedSet &visited_set) {

    // This function uses ID ranges ("L_closed", "R_closed") to navigate the tree,
    //   but it also uses attribute ranges ("L_attr", "R_attr") to filter the nodes.
    // The core idea of this function will be useful for navigating a graph with node compression or edge compression,
    //   which can support IDs that aren't pre-sorted according to the attribute value.

    int ql = L_closed, qr = R_closed;

    int edge_limit = _M;
    std::vector<IdxType> selected_edges;
    selected_edges.reserve(_M);

    iRangeGraph::TreeNode *cur_node = nullptr, *nxt_node = tree->root;

    auto GetOverlap = [](int l, int r, int ql, int qr) {
        int L = std::max(l, ql);
        int R = std::min(r, qr);
        return R - L + 1;
    };

    do {
        cur_node = nxt_node;
        if (cur_node == nullptr) break;

        do {
            if (cur_node->childs.size() == 0) {
                nxt_node = nullptr; break;
            } else {
                nxt_node = find_child_containing(cur_node, pid);
                if (nxt_node->childs.size() == 0) { nxt_node = nullptr; break; }

                // NOTE: during the tree traversal, we still use the ID ranges.
                // If we use a graph with node compression or edge compression, then there's no need for the tree structure. In that scenario, we can completely ditch ID ranges, and rely on attribute ranges instead.
                if (GetOverlap(cur_node->lbound, cur_node->rbound, ql, qr) == GetOverlap(nxt_node->lbound, nxt_node->rbound, ql, qr)) {
                    cur_node = nxt_node;
                    continue;
                } else {
                    break;
                }
            }
        } while (true);  // go down the tree until the overlap isn't the same

        if (cur_node->depth == tree->max_depth) continue;

        auto &edgelist = graphs[cur_node->node_id]->neighbors[pid - cur_node->lbound];

        for (int k : edgelist) {
            int neighborId = k + cur_node->lbound;
            // NOTE: Here, we lookup the attribute from an array, and compare it to the ranges
            float myattr = attrs[neighborId];
            if (myattr < L_attr || R_attr < myattr)
                continue;
            if (visited_set.check(neighborId))
                continue;
            selected_edges.emplace_back(neighborId);
            if (selected_edges.size() == edge_limit)
                return selected_edges;
        }
    } while (cur_node->lbound < ql || cur_node->rbound > qr);

    return selected_edges;
}

void iRG::load(const std::string& index_name,
               std::shared_ptr<IStorage> base_storage,
               std::shared_ptr<DistanceHandler> distance_handler, 
               IdxType M, int ef_con, int n_threads) {
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
    _num_threads = n_threads;

    {
        int n_pts = _base_storage->get_num_points();
        tree = new iRangeGraph::SegmentTree(n_pts);
    }
    tree->BuildTree(tree->root);

    std::vector<std::vector<iRangeGraph::TreeNode *>> level_nodes;
    level_nodes.resize(tree->max_depth + 1);
    for (auto node : tree->treenodes) {
        level_nodes[node->depth].emplace_back(node);
        indices.emplace_back(_verbose);
        graphs.emplace_back(nullptr);
    }

    uint32_t num_indices = 0;
    in.read(reinterpret_cast<char*>(&num_indices), sizeof(uint32_t));
    if (num_indices != tree->treenodes.size()) {
        std::cerr << "Num of indices does not match the tree structure generated from range!" << std::endl;
        std::cerr << num_indices << " v.s. " << tree->treenodes.size() << std::endl;
        abort();
    }

    for (IdxType uid = 0; uid < num_indices; ++uid) {
        if (uid % 100000 == 0) {
            std::cout << "Loading " << (float) 100 * uid / num_indices << "%" << std::endl;
        }
        iRangeGraph::TreeNode *u = tree->treenodes[uid];
        graphs[uid] = std::make_shared<Graph>(u->rbound - u->lbound + 1);
        std::shared_ptr<IStorage> range_storage = nullptr;
        if (_base_storage) {
            range_storage = create_storage(_base_storage, u->lbound, u->rbound + 1);
        }

        indices[uid]._graph = graphs[uid];
        indices[uid]._base_storage = range_storage; // this was needed for statistics
        indices[uid]._distance_handler = distance_handler;
        indices[uid]._verbose = false;

        IdxType index_id = 0, num_nodes = 0;
        in.read(reinterpret_cast<char*>(&index_id), sizeof(IdxType));
        in.read(reinterpret_cast<char*>(&num_nodes), sizeof(IdxType));
        if (index_id != uid) abort();
        if (num_nodes != graphs[uid]->_num_points) abort();

        for (IdxType i = 0; i < num_nodes; ++i) {
            IdxType id;
            in.read(reinterpret_cast<char*>(&id), sizeof(IdxType));
            if (id != i) abort();

            auto &nbrs = graphs[uid]->neighbors[i];
            uint32_t num_neighbors;
            in.read(reinterpret_cast<char*>(&num_neighbors), sizeof(uint32_t));
            nbrs.resize(num_neighbors);
            in.read(reinterpret_cast<char*>(nbrs.data()), num_neighbors * sizeof(IdxType));
        }
    }
}

void iRG::save(const std::string &index_name) {
    fs::path file_path(index_name);
    std::ofstream out(file_path, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Failed to open binary file for writing: " << file_path << std::endl;
        return;
    }

    uint32_t num_indices = static_cast<uint32_t>(indices.size());
    out.write(reinterpret_cast<const char*>(&num_indices), sizeof(uint32_t));

    for (IdxType uid = 0; uid < num_indices; ++uid) {
        auto &graph = indices[uid]._graph;

        IdxType num_nodes = graph->_num_points;
        out.write(reinterpret_cast<const char*>(&uid), sizeof(IdxType));
        out.write(reinterpret_cast<const char*>(&num_nodes), sizeof(IdxType));

        for (IdxType i = 0; i < num_nodes; ++i) {
            out.write(reinterpret_cast<const char*>(&i), sizeof(IdxType));

            const auto &nbrs = graph->neighbors[i];
            uint32_t num_neighbors = static_cast<uint32_t>(nbrs.size());
            out.write(reinterpret_cast<const char*>(&num_neighbors), sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(nbrs.data()), num_neighbors * sizeof(IdxType));
        }
    }

    out.close();
    std::cout << "- All indices saved (binary) to " << file_path << std::endl;
}

void iRG::statistics() {
    std::vector<std::vector<iRangeGraph::TreeNode *>> level_nodes;
    level_nodes.resize(tree->max_depth + 1);
    for (auto node : tree->treenodes) {
        level_nodes[node->depth].emplace_back(node);
    }

    size_t num_points = _base_storage->get_num_points();
    std::cout << "Number of points: " << num_points << std::endl;

    size_t num_edges_all_layers = 0;

    for (int layer = 0; layer < tree->max_depth; layer++) {
        size_t num_edges = 0;
        IdxType min_degree = std::numeric_limits<IdxType>::max(), max_degree = 0;
        for (int uid = 0; uid < level_nodes[layer].size(); uid++) {
            auto node = level_nodes[layer][uid];
            int L = node->lbound;
            int R = node->rbound;
            auto &graph = graphs[node->node_id];
            for (auto pid = L; pid <= R; ++pid) {
                auto s = graph->neighbors[pid - L].size();
                num_edges += s;
                min_degree = std::min(min_degree, (IdxType)s);
                max_degree = std::max(max_degree, (IdxType)s);
            }
        }
        float avg_degree = (float)num_edges / num_points;

        std::cout << "Layer " << layer << ": " << std::endl;
        std::cout << "  Number of edges: " << num_edges << std::endl;
        std::cout << "  Min/Max/Avg degree: "
            << min_degree << "/"
            << max_degree << "/"
            << avg_degree << std::endl;

        num_edges_all_layers += num_edges;
    }

    std::cout << "Total number of edges: " << num_edges_all_layers << std::endl;
}

IdxType iRG::iterate_to_fixed_point__rangefilter_by_id(
        const char *query,
        IdxType l_closed, IdxType r_closed,
        std::shared_ptr<SearchCache> search_cache,
        std::vector<IdxType> *entry_pts) {

    auto dim = _base_storage->get_dim();
    auto& search_queue = search_cache->search_queue;
    auto& visited_set = search_cache->visited_set;
    search_queue.clear();
    visited_set.clear();

    // entry point
    IdxType num_cmps = 0;
    if (entry_pts == nullptr) {
        // throw std::invalid_argument("Must supply entry points");
        std::random_device rd;
        std::default_random_engine e(rd());
        std::vector<iRangeGraph::TreeNode *> filtered_nodes = tree->range_filter(tree->root, l_closed, r_closed);
        for (auto u : filtered_nodes) {
            std::uniform_int_distribution<int> u_start(u->lbound, u->rbound);
            int pid = u_start(e);
            visited_set.set(pid);
            float dist = _distance_handler->compute(query, _base_storage->get_vector(pid), dim);
            search_queue.insert(pid, dist);
            ++num_cmps;
        }
    } else {
        for (int i = 0; i < entry_pts->size(); i++) {
            visited_set.set(entry_pts->at(i));
            float dist = _distance_handler->compute(query, _base_storage->get_vector(entry_pts->at(i)), dim);
            search_queue.insert(entry_pts->at(i), dist);
            ++num_cmps;
        }
    }

    while (search_queue.has_unexpanded_node()) {
        const Candidate& cur = search_queue.get_closest_unexpanded();
        std::vector<IdxType> neighbors = select_edge(cur.id, l_closed, r_closed, visited_set);

        for (int i = 0; i < neighbors.size(); i++) {
            auto& nbr = neighbors[i];
            if (visited_set.check(nbr)) { continue; }

            visited_set.set(nbr);
            float dist = _distance_handler->compute(query, _base_storage->get_vector(nbr), dim);
            num_cmps++;
            if (search_queue.can_insert(dist)) {
                search_queue.insert(nbr, dist);
            }
        }
    }

    // the SearchQueue abstraction conveniently handles stopping condition and top candidate extraction. The search_ef parameter is also managed by the SearchQueue object.

    return num_cmps;
}

IdxType iRG::iterate_to_fixed_point__rangefilter_by_attr(
        const char *query,
        float *attrs, float L_attr, float R_attr,
        std::shared_ptr<SearchCache> search_cache,
        std::vector<IdxType> *entry_pts) {

    int nb = _base_storage->get_num_points();
    int l_closed = std::lower_bound(attrs, attrs+nb, L_attr) - attrs;
    int r_closed = std::lower_bound(attrs, attrs+nb, R_attr) - attrs - 1;

    auto dim = _base_storage->get_dim();
    auto& search_queue = search_cache->search_queue;
    auto& visited_set = search_cache->visited_set;
    search_queue.clear();
    visited_set.clear();

    // entry point
    IdxType num_cmps = 0;
    if (entry_pts == nullptr) {
        // throw std::invalid_argument("Must supply entry points");
        std::random_device rd;
        std::default_random_engine e(rd());
        std::vector<iRangeGraph::TreeNode *> filtered_nodes = tree->range_filter(tree->root, l_closed, r_closed);
        for (auto u : filtered_nodes) {
            std::uniform_int_distribution<int> u_start(u->lbound, u->rbound);
            int pid = u_start(e);
            visited_set.set(pid);
            float dist = _distance_handler->compute(query, _base_storage->get_vector(pid), dim);
            search_queue.insert(pid, dist);
            ++num_cmps;
        }
    } else {
        for (int i = 0; i < entry_pts->size(); i++) {
            visited_set.set(entry_pts->at(i));
            float dist = _distance_handler->compute(query, _base_storage->get_vector(entry_pts->at(i)), dim);
            search_queue.insert(entry_pts->at(i), dist);
            ++num_cmps;
        }
    }

    while (search_queue.has_unexpanded_node()) {
        const Candidate& cur = search_queue.get_closest_unexpanded();
        std::vector<IdxType> neighbors = select_edge_by_attr(cur.id, l_closed, r_closed, attrs, L_attr, R_attr, visited_set);

        for (int i = 0; i < neighbors.size(); i++) {
            auto& nbr = neighbors[i];
            if (visited_set.check(nbr)) { continue; }

            visited_set.set(nbr);
            float dist = _distance_handler->compute(query, _base_storage->get_vector(nbr), dim);
            num_cmps++;
            if (search_queue.can_insert(dist)) {
                search_queue.insert(nbr, dist);
            }
        }
    }

    // the SearchQueue abstraction conveniently handles stopping condition and top candidate extraction. The search_ef parameter is also managed by the SearchQueue object.

    return num_cmps;
}