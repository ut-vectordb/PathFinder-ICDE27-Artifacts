#ifndef QUERYPLANNING_H

#include <vector>
#include <memory>
#include <functional>
#include <algorithm>

#include "graph.h"
#include "search_cache.h"

#include "querytype/querytype.h"

#define USE_ATTR_TREE
#ifdef USE_ATTR_TREE
#include "irangegraph/attribute_tree.h"
#endif

using namespace ANNS;

/**
 * This struct is a reference to some unit graph. It should provide a method to select edges given a filter.
 * NOTE: we must make sure vector<unit_graph_ref> has a small size: small enough to fit in the cache. Or else
 *       the performance will be very bad. That's why we need to discard unit graphs that are too small.
 */
struct unit_graph_ref {
    AttrTreeNode *ag_ptr;

    // the offset can only be applied after sorting the attributes... which we need to look-up
    int attribute_graph_id;

    enum FILTER_STRATEGY_TYPE { DEFAULT, CHECK_ALL, CHECK_ONE, NOCHECK, SKIP } filter_strategy = DEFAULT;

    inline bool contains(IdxType pid, float attr_value) const {
        bool inrange = (ag_ptr->lbound <= attr_value && attr_value < ag_ptr->rbound_open);
        // bool found = ag_ptr->vertex_table.count(pid);
        return inrange;
    }
};

// conceptual model:
// typedef std::function<vector<unit_graph_ref>(const vector<pair<float, float>> &, vector<IdxType> &)> query_planner_t;

// typedef std::vector<std::pair<float, float>> HyperCube;

// virtual class -- interface only
class IQueryPlanner {
public:
    virtual std::vector<unit_graph_ref> operator()(const QueryType &preds) const = 0;

    virtual void preprocess_for_id(const QueryType &preds, int id) {
        std::vector<unit_graph_ref> tmp = (*this)(preds);
        #pragma omp critical
        {
            cache[id] = std::move(tmp);
        }
    }
    virtual std::vector<unit_graph_ref> retrieve_preprocessed(int id) {
        return cache.at(id);
    }

    // constructors aren't inherited, therefore we don't need to mark a virtual constructor;
    // however we do have to mark the destructor as virtual.
    virtual ~IQueryPlanner() = default;

protected:
    std::unordered_map<int, std::vector<unit_graph_ref>> cache;
};

// class UseSingleUnitGraphOOR : public IQueryPlanner {
// public:
//     UseSingleUnitGraphOOR(AttrTreeNode *g_ptr) { g.ag_ptr = g_ptr; g.filter_strategy = unit_graph_ref::NOCHECK; }
//     std::vector<unit_graph_ref> operator()(const QueryType &preds) const override { return {g}; }
//     unit_graph_ref g;
// };

// class UseSingleUnitGraphInFilter : public IQueryPlanner {
// public:
//     UseSingleUnitGraphInFilter(AttrTreeNode *g_ptr) { g.ag_ptr = g_ptr; g.filter_strategy = unit_graph_ref::CHECK_ALL; }
//     std::vector<unit_graph_ref> operator()(const QueryType &preds) const override { return {g}; }
//     unit_graph_ref g;
// };

class PathFinderPlanner: public IQueryPlanner {
public:
    PathFinderPlanner(const std::vector<std::shared_ptr<irg_attr>> &indices) {
        this->indices = indices;
        na = indices.size();

        rootgraph = nullptr;
        for (int i = 0; i < na; i++) {
            if (indices[i] == nullptr) {
                continue;
            }
            if (indices[i]->schema == irg_attr::NUMERICAL)
                rootgraph = indices[i]->tree->root;
            else
                rootgraph = indices[i]->hash->root_collection;
            break;
        }
        my_assert(rootgraph != nullptr);
        my_assert(rootgraph->delegate_root == nullptr);

        reset_index_mask();
    }

    std::vector<unit_graph_ref> operator()(const QueryType &preds) const override;
    std::vector<unit_graph_ref> conjunctive_index_selection(const QueryType &preds, vector<double> &metrics) const;

    std::vector<unit_graph_ref> optimize_disjunctive_plan(const vector<vector<unit_graph_ref>> &input) const;

    AttrTreeNode *rootgraph = nullptr;

    vector<bool> index_mask;
    void reset_index_mask() {
        index_mask.clear();
        index_mask.resize(na, true);
    }
    void set_index_mask(vector<int> index_ids) {
        index_mask.clear();
        index_mask.resize(na, false);
        for (int i : index_ids)
            index_mask[i] = true;
    }

    // "coverage" means: [overlapping region] / [query region projected onto one attribute]
    double get_range_coverage(const QueryType &preds, int attr_id, AttrTreeNode *p) const;
    double get_tuple_coverage(const QueryType &preds, int attr_id, AttrTreeNode *p) const;

    double alpha = 1.00;
    double alpha_categorical = 0.40;
    int na;
    std::vector<float> *global_attr;
    std::vector<std::shared_ptr<irg_attr>> indices;

    void populate_flattened_lists();
    vector<vector<AttrTreeNode *>> flattened_list;
};

class BorrowPlanner : public PathFinderPlanner {
public:
    BorrowPlanner(const std::vector<std::shared_ptr<irg_attr>> &indices, const vector<float> &global_attr) : PathFinderPlanner(indices) {
        my_assert(indices[missing_id] == nullptr);

        leaf_list.resize(indices.size());
        for (int i = 0; i < indices.size(); i++) {
            if (i != missing_id) {
                if (indices[i] == nullptr || indices[i]->schema != irg_attr::NUMERICAL)
                    continue;
                my_assert(indices[i]->tree);
                populate_value_bounds_recursive(indices[i]->tree->root, global_attr, i);
                sort(leaf_list[i].begin(), leaf_list[i].end(), [&](AttrTreeNode *a, AttrTreeNode *b) { return a->lbound < b->lbound; });
            }
        }

        compute_correlations(global_attr);
    }

    void populate_value_bounds_recursive(AttrTreeNode *p, const vector<float> &global_attr, int index_id);
    void compute_correlations(const vector<float> &global_attr) {
        corr.resize(3);
        corr[0] = {0, 1, 0};
        corr[1] = {1, 0, 0};
        corr[2] = {0, 0, 0};
    }
    std::vector<std::vector<float>> corr;

    std::pair<float, float> derive_bound(const QueryType &preds, int index_id) const;
    std::vector<unit_graph_ref> operator()(const QueryType &preds) const override;

    std::vector<std::vector<AttrTreeNode *>> leaf_list;
    int missing_id = 1;
};

using Aug31Planner = PathFinderPlanner;

#define QUERYPLANNING_H
#endif // QUERYPLANNING_H
