#include "queryplanning/queryplanning.h"
#include <omp.h>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <set>

#include "irangegraph/irangegraph.h"

using namespace ANNS;
using namespace std;

double PathFinderPlanner::get_range_coverage(const QueryType &preds, int attr_id, AttrTreeNode *p) const {
    // kind 1 coverage: overlap / graph range ==> density
    // kind 2 coverage: overlap / query length
    float coverage; // == [overlapping region] / [query region projected onto one attribute]

    int i = attr_id;
    std::shared_ptr<irg_attr> index = this->indices[i];

    float v_L = p->lbound;
    float v_R = p->rbound_open;

    float pL, pR;
    vector<float> predicate_set;

    if (preds.get_range_bounds(i, pL, pR)) {
        bool is_inf = std::isinf(pL) || std::isinf(pR);
        float denom = pR-pL;
        if (is_inf) {
            if (index->schema == irg_attr::CATEGORICAL)
                denom = index->hash->max_val - index->hash->min_val;
            else
                denom = index->tree->root->rbound_open - index->tree->root->lbound;
        }
        float v_L_overlap = std::max(pL, v_L);
        float v_R_overlap = std::min(pR, v_R);
        coverage = (v_R_overlap - v_L_overlap) / denom;
        if (coverage < 0) coverage = 0;
    } else if (preds.get_member_set(i, predicate_set)) {
        bool has_overlap = false;
        for (auto val : predicate_set) {
            if (v_L <= val && val < v_R) {
                has_overlap = true; break;
            }
        }
        if (has_overlap)
            coverage = 1;
        else
            coverage = 0;
    } else {
        throw runtime_error("Malformed query");
    }

    return coverage;
}

double PathFinderPlanner::get_tuple_coverage(const QueryType &preds, int attr_id, AttrTreeNode *p) const {
    float coverage; // == [overlapping tuple count] / [tuple count passing (query projected onto one attribute)]

    std::shared_ptr<irg_attr> index = this->indices[attr_id];

    float v_L = p->lbound;
    float v_R = p->rbound_open;

    int denom = 0;
    int nom = 0;
    int nb = this->global_attr->size() / na;
    for (int i = 0; i < nb; i++) {
        const float *p = this->global_attr->data() + i * na;
        if (preds.check_one(attr_id, p)) {
            denom++;
            if (v_L <= p[attr_id] && p[attr_id] < v_R) {
                nom++;
            }
        }
    }

    return (coverage = (double)nom / denom);
}

std::vector<unit_graph_ref> PathFinderPlanner::conjunctive_index_selection(
        const QueryType &preds,
        vector<double> &metrics) const {

    metrics.clear();
    metrics.resize(preds.na() * 2, -1);

    auto comp_metric = [&](vector<unit_graph_ref> &glist, int attr_id) -> double {
        int N = glist.size();
        if (N == 0) return -100;
        double total_cost = 0;
        for (auto g : glist) {
            total_cost += g.ag_ptr->size;
        }
        if (indices[attr_id]->schema == irg_attr::NUMERICAL)
            total_cost *= pow(N, this->alpha);
        else
            total_cost *= pow(N, this->alpha_categorical);

        // w = #nodes in the graph set/#tottal
        // double w = total_cost / rootgraph->size;
        // total_cost *= pow(N, pow(w, beta));

        return 1.00 / total_cost;
    };

    int inv_cnt = 0;

    auto create_gref = [&](int attr_id, AttrTreeNode* node) {
        unit_graph_ref add;
        add.ag_ptr = node;
        add.attribute_graph_id = attr_id;
        add.filter_strategy = unit_graph_ref::NOCHECK;
        return add;
    };

    vector<pair<double, vector<unit_graph_ref>>> all_plans;
    vector<int> index_candidates;

    for (int attr_id = 0; attr_id < na; attr_id++) {
        float qL, qR; vector<float> qset;
        if (preds.get_range_bounds(attr_id, qL, qR)) {
            bool is_inf = (std::isinf(qL) && std::isinf(qR));
            if (is_inf) continue;   // non-constrained
        }
        if (index_mask[attr_id] == true)
            index_candidates.push_back(attr_id);
    }
    if (index_candidates.size() == 0) {
        unit_graph_ref g;
        g.ag_ptr = rootgraph; g.filter_strategy = unit_graph_ref::NOCHECK;
        return {g};
    }

    for (int attr_id : index_candidates) {
        // find entry
        if (indices[attr_id]->schema == irg_attr::CATEGORICAL) {
            AttrTreeNode *p_entry = indices[attr_id]->hash->root_collection;
            vector<unit_graph_ref> ret_whole, ret_split;
            ret_whole.push_back(create_gref(attr_id, p_entry));

            vector<float> pset;
            if (preds.get_member_set(attr_id, pset) == false) {
                // infinite range
                double score_whole = comp_metric(ret_whole, attr_id);
                metrics[attr_id * 2] = score_whole;
                all_plans.push_back(make_pair(score_whole, ret_whole));
                continue;
            }

            auto lookup_hash_or_root = [&](float v) -> AttrTreeNode * {
                auto &ht = indices[attr_id]->hash->hash_table;
                auto it = ht.find(v);
                return it == ht.end() ? indices[attr_id]->hash->root_collection : it->second;
            };

            if (pset.size() == 1) {
                ret_split.push_back(create_gref(attr_id, lookup_hash_or_root(pset[0])));
                double score_split = comp_metric(ret_split, attr_id);
                metrics[attr_id * 2 + 1] = score_split;
                all_plans.push_back(make_pair(score_split, ret_split));
                continue;
            }

            for (float v : pset) {
                ret_split.push_back(create_gref(attr_id, lookup_hash_or_root(v)));
            }

            double score_whole = comp_metric(ret_whole, attr_id);
            double score_split = comp_metric(ret_split, attr_id);
            metrics[attr_id * 2] = score_whole;
            metrics[attr_id * 2 + 1] = score_split;

            if (score_whole > score_split) {
                all_plans.push_back(make_pair(score_whole, ret_whole));
            } else {
                all_plans.push_back(make_pair(score_split, ret_split));
            }
            continue;
        }

        AttrTreeNode *p = indices[attr_id]->tree->root;
        float qL, qR;
        my_assert(preds.get_range_bounds(attr_id, qL, qR));
        while (true) {
            bool stepped = false;
            for (auto &c : p->childs) {
                if (get_range_coverage(preds, attr_id, c) >= 1) {
                    p = c; stepped = true; break;
                }
            }
            if (!stepped) break;
        }
        AttrTreeNode *p_entry = p;

        /**
         * We have two cases: split or not split.
         * We use the metric score: Avg[density] / N
         */

        vector<unit_graph_ref> ret_whole, ret_split;
        ret_whole.push_back(create_gref(attr_id, p_entry));

        for (size_t i = 0; i < p_entry->childs.size(); i++) {
            AttrTreeNode *cur = p_entry->childs[i];

            float q_cur_L = max(qL, cur->lbound), q_cur_R = min(qR, cur->rbound_open);
            if (q_cur_L >= q_cur_R) { continue; }

            auto preds_child = preds;
            preds_child.set_range(attr_id, q_cur_L, q_cur_R);

            // walk down the tree to the deepest fully covered node
            while (true) {
                bool stepped = false;
                for (auto &c : cur->childs) {
                    if (get_range_coverage(preds_child, attr_id, c) >= 1) {
                        cur = c;
                        stepped = true;
                        break;
                    }
                }
                if (!stepped) break;
            }

            ret_split.push_back(create_gref(attr_id, cur));  // this should NOT make a difference
        }

        double score_whole = comp_metric(ret_whole, attr_id);
        double score_split = comp_metric(ret_split, attr_id);
        metrics[attr_id * 2] = score_whole;
        metrics[attr_id * 2 + 1] = score_split;

        if (score_whole > score_split) {
            all_plans.push_back(make_pair(score_whole, ret_whole));
        } else {
            all_plans.push_back(make_pair(score_split, ret_split));
        }
    }

    if (all_plans.size() == 0) {
        cerr << "No valid plan found." << endl;
        throw runtime_error("No valid plan found.");
    }
    sort(all_plans.begin(), all_plans.end(), [](const pair<double, vector<unit_graph_ref>> &a, const pair<double, vector<unit_graph_ref>> &b) { return a.first > b.first; });
    if (all_plans[0].second.size() == 0) {
        cerr << "Best plan is invalid. total plan count = " << all_plans.size() << endl;
        cerr << "Score = " << all_plans[0].first << endl;
        throw runtime_error("No valid plan found.");
    }

    return all_plans[0].second;
}

void PathFinderPlanner::populate_flattened_lists() {
    flattened_list.resize(na);
    for (int i = 0; i < na; i++) {
        if (indices[i]->schema == irg_attr::CATEGORICAL) {
            flattened_list[i] = indices[i]->hash->hashednodes;  // includes the root partition, and the root partition is always the first one
        } else {
            flattened_list[i] = indices[i]->tree->treenodes;
        }
        // remove nullptrs
        auto &lst = flattened_list[i];
        lst.erase(std::remove_if(lst.begin(), lst.end(), [](AttrTreeNode *p) { return p == nullptr; }), lst.end());
        sort(flattened_list[i].begin(), flattened_list[i].end(), [](AttrTreeNode *a, AttrTreeNode *b) {
            if (a->depth == b->depth) {
                return a->lbound < b->lbound;
            } else {
                return a->depth > b->depth;
            }
        });
    }
}

std::vector<unit_graph_ref> PathFinderPlanner::optimize_disjunctive_plan(const vector<vector<unit_graph_ref>> &input) const {
    if (input.size() == 1) {
        return input[0];
    }

    my_assert(input.size() == 2);

    std::vector<unit_graph_ref> all;
    for (auto &partial_permutation : input)
        for (auto &tmp : partial_permutation)
            all.push_back(tmp);

    if (0) {
        // Version: disables disjunctive optimization
        return all;
    }

    if (1) {
        // Version: preprocess the list to remove any dominated nodes
        AttrTreeNode *root;
        std::vector<unit_graph_ref> new_all;
        if (indices[0]->schema == irg_attr::CATEGORICAL)
            root = indices[0]->hash->root_collection;
        else
            root = indices[0]->tree->root;

        // deal with the case where root is present
        for (int i = 0; i < all.size(); i++) {
            if (all[i].ag_ptr->delegate_root || all[i].ag_ptr->size == root->size) {
                new_all.push_back(all[i]);
                break;
            }
        }
        if (new_all.size() > 0) {
            // found root in it
            all = new_all;
        } else {
            for (int i = 0; i < all.size(); i++) {
                bool has_ancestor = false;
                for (int j = 0; j < all.size(); j++) {
                    if (i == j) continue;
                    if (all[j].attribute_graph_id == all[i].attribute_graph_id && all[i].ag_ptr->node_id == all[j].ag_ptr->node_id) {
                        // i'm not sure if this case is real, but in case they are the same node, we make the first one the ancestor of others
                        if (j < i) {
                            has_ancestor = true;
                            break;
                        }
                        continue;
                    }
                    // test whether j is the ancestor of i
                    if (all[j].attribute_graph_id != all[i].attribute_graph_id) continue;
                    if (all[j].ag_ptr->lbound <= all[i].ag_ptr->lbound && all[i].ag_ptr->rbound_open <= all[j].ag_ptr->rbound_open) {
                        has_ancestor = true;
                        break;
                    }
                }
                if (!has_ancestor) {
                    new_all.push_back(all[i]);
                }
            }

            if (new_all.size() == 0) {
                auto graphstring = [&](unit_graph_ref &g) {
                    string ret;
                    if (g.ag_ptr->depth == 0) {
                        ret = "[Root]";
                    } else {
                        ret += "[G: Attr=" + to_string(g.attribute_graph_id) + ", Depth=" + to_string(g.ag_ptr->depth) + ", L=" + to_string(g.ag_ptr->lbound) + ", R=" + to_string(g.ag_ptr->rbound_open) + "]";
                    }
                    return ret;
                };
                auto planstring = [&](vector<unit_graph_ref> &plan) {
                    string ret;
                    for (auto &gref : plan) {
                        ret += graphstring(gref) + ", ";
                    }
                    return ret;
                };
                #pragma omp critical
                {
                    cerr << planstring(all) << endl;
                    cerr.flush();
                }
            }

            all = new_all;
        }
    }

    int index_id = input[0][0].attribute_graph_id;

    for (auto &clause : input) {
        if (clause[0].attribute_graph_id != index_id) {
            return all;
        }
    }

    // all clauses belong to the same index.
    // following the algorithm-2 in the paper

    // first collect a flattened version of the tree that puts leaf nodes on the front
    my_assert(flattened_list.size() == na);

    // outputs plan \ S + g_p
    auto update_plan = [](const vector<unit_graph_ref> &plan, const vector<AttrTreeNode *> &S, AttrTreeNode *g_p) -> vector<unit_graph_ref> {
        vector<unit_graph_ref> ret;

        unit_graph_ref g_p_ref; g_p_ref.ag_ptr = g_p; g_p_ref.filter_strategy = unit_graph_ref::NOCHECK;
        ret.push_back(g_p_ref);

        for (const unit_graph_ref &gref : plan) {
            auto ptr = gref.ag_ptr;
            if (ptr == g_p) continue;
            bool do_not_add = false;
            for (auto gptr : S) {
                if (gref.ag_ptr == gptr) {
                    do_not_add = true; break;
                }
            }
            if (not do_not_add)
                ret.push_back(gref);
        }
        return ret;
    };

    auto comp_util = [&](vector<AttrTreeNode *> &glist, int attr_id) -> double {
        int N = glist.size();
        if (N == 0) {
            return std::numeric_limits<float>::infinity();
        }
        double total_cost = 0;
        for (auto g : glist) {
            total_cost += g->size;
        }
        if (indices[attr_id]->schema == irg_attr::NUMERICAL)
            total_cost *= pow(N, this->alpha);
        else
            total_cost *= pow(N, this->alpha_categorical);

        // w = #nodes in the graph set/#tottal
        // double w = total_cost / rootgraph->size;
        // total_cost *= pow(N, pow(w, beta));

        return 1 / total_cost;
    };

    unordered_map<int, vector<AttrTreeNode *>> plan;

    // for each leaf node and non leaf node
    for (AttrTreeNode *p : flattened_list.at(index_id)) {
        if (p == nullptr) continue;
        if (p->childs.size() == 0) {    // leaf
            for (auto &gref : all) {
                if (gref.ag_ptr == p) {
                    plan[p->node_id] = {p};
                }
            }
        } else {
            vector<AttrTreeNode *> S;
            for (auto &c : p->childs) {
                if (plan.count(c->node_id)) {
                    auto &c_plan = plan[c->node_id];
                    for (auto &it : c_plan) {
                        S.push_back(it);
                    }
                }
            }

            double util1 = comp_util(S, index_id);
            vector<AttrTreeNode *> pp = {p};
            double util2 = comp_util(pp, index_id);

            if (S.size() > 0 && util2 > util1) {
                plan[p->node_id] = pp;
                all = update_plan(all, S, p);
            } else {
                bool inside = false;
                for (auto &gref : all) {
                    if (gref.ag_ptr == p) {
                        inside = true; break;
                    }
                }
                if (inside) {
                    plan[p->node_id] = pp;
                } else {
                    if (S.size() > 0)
                        plan[p->node_id] = S;
                }
            }
        }
    }

    return all;
}

std::vector<unit_graph_ref> PathFinderPlanner::operator()(const QueryType &preds) const {
    vector<double> metric_scores;
    return conjunctive_index_selection(preds, metric_scores);
}

void BorrowPlanner::populate_value_bounds_recursive(AttrTreeNode *p, const vector<float> &global_attr, int index_id) {
    if (p->childs.size() == 0) {
        p->populate_value_bounds(global_attr, na);
        leaf_list[index_id].push_back(p);
        return;
    }
    p->value_bounds.resize(na);
    float inf = numeric_limits<float>::infinity();
    for (int j = 0; j < na; j++) {
        p->value_bounds[j] = make_pair(inf, -inf);
    }
    for (auto *c : p->childs) {
        populate_value_bounds_recursive(c, global_attr, index_id);
        for (int j = 0; j < na; j++) {
            p->value_bounds[j].first = min(p->value_bounds[j].first, c->value_bounds[j].first);
            p->value_bounds[j].second = max(p->value_bounds[j].second, c->value_bounds[j].second);
        }
    }
}

std::pair<float, float> BorrowPlanner::derive_bound(const QueryType &preds, int index_id) const {
    float Q_min, Q_max;
    my_assert(preds.get_range_bounds(missing_id, Q_min, Q_max));    // takes in old bounds

    float inf = std::numeric_limits<float>::infinity();

    // derive the new range to pass into the index selection subroutine
    // scan the leaf nodes
    float A_min = inf, A_max = -inf;
    for (AttrTreeNode *leaf : leaf_list[index_id]) {
        // check whether one leaf "overlaps" with the old bound. If so, absorb the node into the new bound
        if (Q_min <= leaf->value_bounds[missing_id].second && leaf->value_bounds[missing_id].first <= Q_max) {
            A_min = min(A_min, leaf->value_bounds[index_id].first);
            A_max = max(A_max, leaf->value_bounds[index_id].second);
        }
    }

    return make_pair(A_min, A_max);
}

std::vector<unit_graph_ref> BorrowPlanner::operator()(const QueryType &preds) const {
    float inf = std::numeric_limits<float>::infinity();

    int index_id;   // <- choose this
    float max_sim = -1;
    for (int i = 0; i < corr.size(); i++) {
        if (corr[missing_id][i] >= max_sim) {
            max_sim = corr[missing_id][i];
            index_id = i;
        }
    }

    const auto &[A_min, A_max] = derive_bound(preds, index_id);
    QueryType derived_pred(na);
    for (int i = 0; i < na; i++) {
        derived_pred.set_range(i, -inf, inf);
    }
    derived_pred.set_range(index_id, A_min, A_max);

    vector<double> dummy_metrics;
    vector<unit_graph_ref> plan = conjunctive_index_selection(derived_pred, dummy_metrics);

    return plan;
}