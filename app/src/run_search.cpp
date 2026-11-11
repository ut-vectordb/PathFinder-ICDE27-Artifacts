#include "run_search.h"

// #define BREAK_QPS

int flag13 = 0; // choose random index among constrained attributes

#include <set>

void save_recall(const std::pair<IdxType, float>* gt, const std::pair<IdxType, float>* results, uint32_t num_queries, uint32_t K) {
    float total_correct = 0;
    vector<float> recalls;
    for (uint32_t i = 0; i < num_queries; i++) {
        float this_recall = 0;
        // prepare ground truth set, offset records the last valid gt index
        std::set<IdxType> gt_set;
        int32_t offset = -1;
        for (uint32_t j = 0; j < K; j++) {
            if (gt[i * K + j].first != -1) {
                offset = j;
                gt_set.insert(gt[i * K + j].first);     
            }
        }
        
        // count the correct
        float this_correct = 0;
        float this_valid = offset + 1;
        std::vector<std::pair<IdxType, float>> res, cleaned;
        for (uint32_t j = 0; j < K; j++) {
            res.push_back(results[i * K + j]);
        }
        ANNS::clean_results(res, cleaned, K);
        for (uint32_t j = 0; j < cleaned.size(); j++) {
            if (cleaned[j].first == -1)
                break;
            if (offset >=0 && cleaned[j].second == gt[i * K + offset].second) {           // for ties
                this_correct++;
                offset--;
            } else {
                if (gt_set.find(cleaned[j].first) != gt_set.end())
                this_correct++;
            }
        }
        total_correct += this_correct;
        if (this_valid == 0)
            this_recall = 1;
        else
            this_recall = this_correct / this_valid;
        recalls.push_back(this_recall);
    }

    static int numbering=0;
    fs::create_directories("raw_data/");
    string rawdataname = "raw_data/recalls_" + to_string(numbering) + ".csv";
    ofstream fout(rawdataname);
    numbering++;
    fout << "query_index,recall\n";
    for (size_t i = 0; i < recalls.size(); i++) {
        fout << i << "," << recalls[i] << "\n";
    }
    cerr << "[+] Wrote recall data to '" << rawdataname << "'\n";
}

static vector<double> calc_selectivity_each_axis(
        // TableStorage &tbstorage,
        const vector<float> &attrs,
        const QueryType &q
    ) {

    // int nb = tbstorage.nb;
    // int na = tbstorage.na;
    int na = q.na();
    int nb = attrs.size() / na;

    vector<size_t> cnt(na, 0);

    for (int i = 0; i < nb; i++) {
        const float *ptr = attrs.data() + i * na;
        for (int j = 0; j < na; j++) {
            if (q.check_one(j, ptr)) {
                cnt[j]++;
            }
        }
    }

    vector<double> sels(na, 0);
    for (int i = 0; i < na; i++) {
        sels[i] = (double)cnt[i] / nb;
    }

    return sels;
}

void choose_indexes_using_min_sel(
        TableStorage &tbstorage,
        const vector<float> &attrs,
        const vector<QueryType> &range,
        vector<int> &result_index_ids
    ) {

    int nb = tbstorage.nb;
    int na = tbstorage.na;
    int dim = tbstorage.n_dim;
    int nq = range.size();

    result_index_ids.resize(nq);

    if (flag13 == 1) {
        for (int qi = 0; qi < nq; qi++) {
            const auto &q = range[qi];
            vector<int> constrained_ids;
            vector<float> qset; float ql, qr;
            for (int i = 0; i < na; i++) {
                if (q.get_member_set(i, qset)) {
                    constrained_ids.push_back(i);
                } else if (q.get_range_bounds(i, ql, qr)) {
                    if (!std::isinf(ql) && !std::isinf(qr)) {
                        constrained_ids.push_back(i);
                    }
                }
            }
            result_index_ids[qi] = constrained_ids.at(rand() % constrained_ids.size());
        }

        return;
    }

    #pragma omp parallel for
    for (int qi = 0; qi < nq; qi++) {
        const auto &q = range[qi];
        auto sels = calc_selectivity_each_axis(attrs, q);
        int best_id = 0;
        double best_sel = 10;
        for (int i = 0; i < na; i++) {
            if (sels[i] < best_sel) {
                best_sel = sels[i]; best_id = i;    // chooses the smallest
            }
        }
        result_index_ids[qi] = best_id;
    }
}

void choose_indexes_using_min_sel(
        TableStorage &tbstorage,
        const vector<float> &attrs,
        const vector<DisjunctiveQueryType> &range,
        vector<int> &result_index_ids
    ) {

    int nb = tbstorage.nb;
    int na = tbstorage.na;
    int dim = tbstorage.n_dim;
    int nq = range.size();

    result_index_ids.resize(nq * 2);    // NOTE: We assume DisjunctiveQueryType always has 2 clauses

    if (flag13 == 1) {
        for (int qi = 0; qi < nq; qi++) {
            vector<int> constrained_ids;
            vector<float> qset; float ql, qr;
            my_assert(range[qi].clauses.size() == 2);
            for (int j = 0; j < range[qi].clauses.size(); ++j) {
                const auto &q = range[qi].clauses[j];
                for (int i = 0; i < na; i++) {
                    if (q.get_member_set(i, qset)) {
                        constrained_ids.push_back(i);
                    } else if (q.get_range_bounds(i, ql, qr)) {
                        if (!std::isinf(ql) && !std::isinf(qr)) {
                            constrained_ids.push_back(i);
                        }
                    }
                }
                result_index_ids[qi * 2 + j] = constrained_ids.at(rand() % constrained_ids.size());
            }
        }

        return;
    }

    #pragma omp parallel for
    for (int qi = 0; qi < nq; qi++) {
        my_assert(range[qi].clauses.size() == 2);
        for (int j = 0; j < range[qi].clauses.size(); ++j) {
            const auto &q = range[qi].clauses[j];
            auto sels = calc_selectivity_each_axis(attrs, q);
            int best_id = 0;
            double best_sel = 10;
            for (int i = 0; i < na; i++) {
                if (sels[i] < best_sel) {
                    best_sel = sels[i]; best_id = i;    // chooses the smallest
                }
            }
            result_index_ids[qi * 2 + j] = best_id;
        }
    }
}

void choose_indexes_using_min_sel(
        TableStorage &tbstorage,
        const vector<float> &attrs,
        const QueryCollection &range,
        vector<int> &result_index_ids
    ) {

    if (range.dq_)
        choose_indexes_using_min_sel(tbstorage, attrs, *range.dq_, result_index_ids);
    else
        choose_indexes_using_min_sel(tbstorage, attrs, *range.qt_, result_index_ids);
}

void search_unit_graph_prob(
        TableStorage &tbstorage,
        char *query,    // query_vec
        std::function<bool(int)> predicate,
        std::function<bool(int)> one_attr_predicate, // We don't have functool in C++... Should we use a functor instead?
        int K, int n_entry,

        const AttrTreeNode *graph,
        SearchCacheList &search_cache_list,
        unsigned int &random_state,
        vector<pair<IdxType, float>> &all,
        unordered_set<IdxType> &seen
    ) {

    auto ProbFunc = [&](int x, unsigned int &rand_state) -> int {
        if (x >= 20)
            return 0;
        double randNum = (double)rand_r(&rand_state) / RAND_MAX;
        return randNum < (1 / (1 + exp(x))) ? 1 : 0;
    };

    auto compdist = [&](IdxType pid) {
        // n_cmps++;
        auto *vec_ptr = tbstorage.get_vector(tbstorage.lookup(pid));
        return distance_handler->compute(query, (const char *)vec_ptr, dim);
    };

    priority_queue<pair<float, IdxType>> top_candidates;    // a dedicated list for valid results in post-filtering
    auto record_top = [&](IdxType pid, float dist) {    // only call if the pid passes predicate
        top_candidates.emplace(dist, pid);
        while (top_candidates.size() > K) { top_candidates.pop(); }
    };

    auto search_cache = search_cache_list.get_free_cache();

    auto &search_queue = search_cache->search_queue;
    auto &visited_set = search_cache->visited_set;
    search_queue.clear(); visited_set.clear();

    int cntr = 5;   // NOTE: n_entry is not honored
    for (auto it : graph->edgelist_hashmap) {
        IdxType entry = it.first;
        if (visited_set.check(entry)) continue;
        visited_set.set(entry);
        float dist = compdist(entry);
        search_queue.insert(entry, dist, -1);
        if (predicate(entry)) { record_top(entry, dist); }
        --cntr;
        if (cntr <= 0) break;
    }

    while (search_queue.has_unexpanded_node()) {
        const ANNS::Candidate cur = search_queue.get_closest_unexpanded();

        float *myattrs = tbstorage.get_attr(tbstorage.lookup(cur.id));

        std::vector<IdxType> edgelist_gref;
        { auto guard = graph->edgelist_hashmap.read_lock(cur.id); edgelist_gref = *guard.edges; }
        vector<IdxType> edgelist;

        for (const IdxType nbr : edgelist_gref) {
            int max_e = 32;

            if (!one_attr_predicate(nbr))
                continue;
            int prob = 1;
            int next_step = cur.extra_meta + 1;
            bool inrange = predicate(nbr);
            if (!inrange) {
                prob = ProbFunc(next_step, random_state);
            }
            if (!prob) { continue; }
            edgelist.emplace_back(nbr);
            if (edgelist.size() >= max_e) {
                break;
            }
        }

        for (auto neighbor : edgelist) {
            if (visited_set.check(neighbor)) {
                continue;
            }
            visited_set.set(neighbor);
            float dist = compdist(neighbor);

            int next_step = cur.extra_meta + 1;
            if (predicate(neighbor)) {
                record_top(neighbor, dist);
                search_queue.insert(neighbor, dist, -1);
            } else {
                search_queue.insert(neighbor, dist, next_step);
            }
        }
    }

    // aggregate results from multiple graph runs
    while (!top_candidates.empty()) {
        auto [dist, id] = top_candidates.top(); top_candidates.pop();
        if (seen.insert(id).second) { // first time seeing this id
            all.emplace_back(id, dist);
        }
    }

    search_cache_list.release_cache(search_cache);
}

void search_irangegraph_prob(
        TableStorage &tbstorage,
        char *query,    // query_vec
        std::function<bool(int)> predicate,
        std::function<bool(int)> one_attr_predicate, // We don't have functool in C++... Should we use a functor instead?
        int K, int n_entry,

        float L, float R,
        int attr_id,
        const shared_ptr<irg_attr> index,
        SearchCacheList &search_cache_list,
        unsigned int &random_state,
        vector<pair<IdxType, float>> &all,
        unordered_set<IdxType> &seen
    ) {

    vector<IdxType> entries;
    auto tmp = index->entrypoint_selection(L, R, n_entry);
    for (auto u : tmp) {
        entries.push_back(u);
    }

    priority_queue<pair<float, IdxType>> top_candidates;    // a dedicated list for valid results in post-filtering
    auto compdist = [&](IdxType pid) {
        auto *vec_ptr = tbstorage.get_vector(tbstorage.lookup(pid));
        return distance_handler->compute(query, (const char *)vec_ptr, dim);
    };
    auto record_top = [&](IdxType pid, float dist) {    // only call if the pid passes predicate
        top_candidates.emplace(dist, pid);
        while (top_candidates.size() > K) { top_candidates.pop(); }
    };

    auto search_cache = search_cache_list.get_free_cache();
    auto &search_queue = search_cache->search_queue;
    auto &visited_set = search_cache->visited_set;
    search_queue.clear(); visited_set.clear();

    for (IdxType entry : entries) {
        float dist = compdist(entry);
        if (visited_set.check(entry))
            continue;
        search_queue.insert(entry, dist, -1);
        if (predicate(entry)) {
            record_top(entry, dist);
        }
        visited_set.set(entry);
    }

    while (search_queue.has_unexpanded_node()) {    // NOTE: we need to end when we have |top_candidates| == ef_search
        const ANNS::Candidate cur = search_queue.get_closest_unexpanded();
        float *myattrs = tbstorage.get_attr(tbstorage.lookup(cur.id));

        int max_e = 32;

        vector<int> ne_sdnc;    // scanned, dup, notpass, chosen
        vector<IdxType> edgelist;

        float myattr = myattrs[attr_id];

        // if (postfiltering) {
            // if (use_prob) {
                edgelist = index->select_edge_infilter_prob(
                    cur.id, myattr, L, R,
                    visited_set, one_attr_predicate, predicate, cur.extra_meta, max_e, random_state, &ne_sdnc);
            // } else {
            //     edgelist = index->select_edge_infilter(
            //         cur.id, myattr, L, R,
            //         visited_set, one_attr_predicate, max_e, &ne_sdnc);
            // }

        for (auto neighbor : edgelist) {
            if (visited_set.check(neighbor)) {
                continue;
            }
            visited_set.set(neighbor);
            float dist = compdist(neighbor);

            int next_step = cur.extra_meta + 1;
            if (predicate(neighbor)) {
                record_top(neighbor, dist);
                search_queue.insert(neighbor, dist, -1);
            } else {
                search_queue.insert(neighbor, dist, next_step);
            }
        }
    }

    // aggregate results from multiple graph runs
    while (!top_candidates.empty()) {
        auto [dist, id] = top_candidates.top(); top_candidates.pop();
        if (seen.insert(id).second) { // first time seeing this id
            all.emplace_back(id, dist);
        }
    }

    search_cache_list.release_cache(search_cache);
}

int baseline_index_selection(
        const vector<bool> index_mask,
        const QueryType &conj_clause,
        unsigned int &random_state) {

    vector<double> sels(na, 0);
    int best_id = 0;
    double best_sel = nb * 10.0;

    vector<int> constrained_ids;
    vector<float> qset; float ql, qr;
    for (int i = 0; i < na; i++) {
        if (conj_clause.get_member_set(i, qset)) {
            if (index_mask[i])
                constrained_ids.push_back(i);
        } else if (conj_clause.get_range_bounds(i, ql, qr)) {
            if (!std::isinf(ql) && !std::isinf(qr)) {
                if (index_mask[i])
                    constrained_ids.push_back(i);
            }
        }
    }
    if (constrained_ids.size() > 0) {
        my_assert(constrained_ids.size() > 0);
        best_id = constrained_ids.at(rand_r(&random_state) % constrained_ids.size());
        return best_id;
    } else {
        return -100;
    }
}

void test_search_IRG_collection(
        TableStorage &tbstorage,
        shared_ptr<ANNS::IStorage> query_storage,
        const vector<int> &index_ids,
        const vector<shared_ptr<irg_attr>> &irg_indices,
        const vector<float> &attrs,
        const QueryCollection &range,
        int K, const vector<int> &ef_searches, int n_entry,
        pair<IdxType, float> *gt,
        AttrTreeNode *rootgraph
    ) {

    my_assert(rootgraph->delegate_root == nullptr); // we need a non-empty rootgraph for this

    int nq = query_storage->get_num_points();
    int nb = tbstorage.nb;
    int na = tbstorage.na;
    int dim = tbstorage.n_dim;
    vector<bool> index_mask(na, false);
    for (int i : index_ids)
        index_mask[i] = true;

    auto results = new std::pair<ANNS::IdxType, float>[nq * K];

    vector<double> qpss, recalls, cmpss;
    for (auto ef_search : ef_searches) {
        ANNS::SearchCacheList search_cache_list(num_threads + 5, nb, ef_search);

        auto start_time = std::chrono::high_resolution_clock::now();
        vector<int> num_cmps(nq, 0);

#ifndef PRINT_TRACE_ID
        #pragma omp parallel for schedule (dynamic, 1)
        for (int q = 0; q < nq; q++) {
#else
        for (int q = PRINT_TRACE_ID; q == PRINT_TRACE_ID; ++q) {
#endif

            vector<pair<IdxType, float>> all;
            unordered_set<IdxType> seen;
            int n_cmps = 0;

            int cid = -1;
            for (const QueryType &conj_clause : range.get_clauses(q)) {
                ++cid;

                unsigned int random_state = 0;

                int my_index_id = baseline_index_selection(index_mask, conj_clause, random_state);
                if (my_index_id == -1) {
                }

                auto query = query_storage->get_vector(q);

                auto predicate = [&](int pid) {
                    const float *attr_ptr = tbstorage.get_attr(tbstorage.lookup(pid));
                    return conj_clause.check(attr_ptr);
                };

                int one_attr_k = 0;
                auto one_attr_predicate = [&](int pid) {
                    const float *attr_ptr = tbstorage.get_attr(tbstorage.lookup(pid));
                    return conj_clause.check_one(one_attr_k, attr_ptr);
                };

                if (my_index_id == -100) {
                    // use the root graph and OOR
                    unit_graph_ref gref;
                    gref.ag_ptr = rootgraph; my_assert(rootgraph != nullptr);
                    gref.filter_strategy = unit_graph_ref::NOCHECK;
                    search_graph_reference_OOR(tbstorage, query, predicate, [](int pid) {return true;}, K, 5, gref, search_cache_list, n_cmps, all, seen);
                    continue;
                }

                int sort_by = my_index_id;

                if (irg_indices[sort_by]->schema == irg_attr::CATEGORICAL) {
                    auto index = irg_indices[sort_by];

                    // do in-filtering in each of the little graphs

                    vector<AttrTreeNode *> graphs;
                    vector<float> qset;
                    my_assert(conj_clause.get_member_set(sort_by, qset));
                    for (auto val : qset) {
                        auto ptr = index->hash->hash_table.at(val);
                        my_assert(ptr != nullptr);
                        if (ptr->delegate_root)
                            graphs.push_back(ptr->delegate_root);
                        else
                            graphs.push_back(ptr);
                    }

                    for (const AttrTreeNode *gg : graphs) {
                        one_attr_k = sort_by;
                        search_unit_graph_prob(
                            tbstorage, query, predicate, one_attr_predicate,
                            K, n_entry,
                            gg,
                            search_cache_list, random_state, all, seen
                        );
                    }
                } else {
                    auto index = irg_indices[sort_by];
                    float L, R;
                    bool success = conj_clause.get_range_bounds(sort_by, L, R);
                    my_assert(success);

                    one_attr_k = sort_by;

                    search_irangegraph_prob(
                        tbstorage, query, predicate, one_attr_predicate,
                        K, n_entry, L, R, sort_by,
                        index,
                        search_cache_list, random_state, all, seen
                    );
                } // CATEGORICAL or NUMERICAL
            } // for conj clause

            // Aggregate all results
            std::sort(all.begin(), all.end(), [](auto& a, auto& b) { return a.second < b.second; });
            if ((int)all.size() > K) {
                all.resize(K);
            }

            for (int k = all.size(); k < K; k++) { results[q*K+k].first = -1; }
            for (int k = 0; k < all.size(); k++) { results[q*K+k] = all[k]; }
        } // for query ID

        auto time_cost = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
        double QPS = nq * 1e6 / time_cost;

        double avg_cmps = (double)accumulate(num_cmps.begin(), num_cmps.end(), 0) / nq;

        double recall = calculate_recall(gt, results, nq, K, false);

        if (true) {
            cerr << endl;
            cerr << "ef_search = " << ef_search << endl;
            cerr << "- Time cost (us): " << time_cost << endl;
            cerr << "- Time per comp (ns): " << 1e6 * time_cost / (avg_cmps * nq) << endl;
            cerr << "- QPS: " << QPS << endl;
            cerr << "- Average number of comparisons: " << avg_cmps << endl;
            cerr << "- Recall: " << recall << endl;
        }
        if (flag3 >= 0 && ef_search == 100) {
            save_recall(gt, results, nq, K);
        }

        qpss.push_back(QPS);
        cmpss.push_back(avg_cmps);
        recalls.push_back(recall);
        // if (avg_cmps > 8000) break;
#ifdef BREAK_QPS
        if (QPS < 1000) break;
#endif
    }

    cout << "[";
    for (double q : cmpss) cout << q << ",";
    cout << "],[";
    for (double q : qpss) cout << q << ",";
    cout << "],[";
    for (double r : recalls) cout << r << ",";
    cout << "]," << endl;

    delete[] results;
}

void search_graph_reference_simple_postfiltering(
        TableStorage &tbstorage,
        char *query,
        std::function<bool(int)> predicate,
        int K,

        const unit_graph_ref &graphref,
        SearchCacheList &search_cache_list,
        int &n_cmps,
        vector<pair<IdxType, float>> &all,
        unordered_set<IdxType> &seen) {

    auto search_cache = search_cache_list.get_free_cache();

    auto compdist = [&](IdxType pid) {
        ++n_cmps;
        auto *vec_ptr = tbstorage.get_vector(tbstorage.lookup(pid));
        return distance_handler->compute(query, (const char *)vec_ptr, dim);
    };

    priority_queue<pair<float, IdxType>> top_candidates; 
    auto record_top = [&](IdxType pid, float dist) {
        top_candidates.emplace(dist, pid);
        while (top_candidates.size() > K) { top_candidates.pop(); }
    };

    auto &search_queue = search_cache->search_queue;
    auto &visited_set = search_cache->visited_set;
    search_queue.clear(); visited_set.clear();

    int cntr = 5;

    AttrTreeNode *ag_ptr = graphref.ag_ptr;
    if (ag_ptr->delegate_root)
        ag_ptr = ag_ptr->delegate_root;

    for (auto it : ag_ptr->edgelist_hashmap) {
        IdxType entry = it.first;
        if (visited_set.check(entry)) continue;
        visited_set.set(entry);

        float dist = compdist(entry);
        search_queue.insert(entry, dist, -1);
        record_top(entry, dist);

        --cntr;
        if (cntr <= 0) break;
    }

    while (search_queue.has_unexpanded_node()) {
        const ANNS::Candidate cur = search_queue.get_closest_unexpanded();

        std::vector<IdxType> edgelist_gref;
        { auto guard = ag_ptr->edgelist_hashmap.read_lock(cur.id); edgelist_gref = *guard.edges; }
        for (const IdxType nbr : edgelist_gref) {
            if (visited_set.check(nbr)) { continue; }
            visited_set.set(nbr);

            float dist = compdist(nbr);
            search_queue.insert(nbr, dist, cur.id);
            record_top(nbr, dist);  // search frontier is added to results
        }
    } // while (search_queue.has_unexpanded_node())

    search_cache_list.release_cache(search_cache);

    vector<pair<float, IdxType>> unfiltered_candidates;
    while (!top_candidates.empty()) {
        auto [dist, id] = top_candidates.top(); top_candidates.pop();
        unfiltered_candidates.emplace_back(dist, id);
    }

    vector<pair<float, IdxType>> filtered_candidates;
    for (int i = unfiltered_candidates.size() - 1; i >= 0; i--) {
        auto [dist, id] = unfiltered_candidates[i];
        if (predicate(id)) {
            filtered_candidates.emplace_back(dist, id);
        }
    }

    for (auto [dist, id] : filtered_candidates) {
        if (seen.insert(id).second) { // first time seeing this id
            all.emplace_back(id, dist);
        }
    }
}

void search_graph_reference_doubling_postfiltering(
        TableStorage &tbstorage,
        char *query,
        std::function<bool(int)> predicate,
        int K,

        const unit_graph_ref &graphref,
        SearchCacheList &search_cache_list,
        vector<pair<IdxType, float>> &all,
        unordered_set<IdxType> &seen) {

    size_t K_prime = K;
    my_assert(K_prime > 0);

    while (true) {
        // auto search_cache = std::make_shared<SearchCache>(tbstorage.nb, K_prime);   // ef_search = K_prime
        auto search_cache = search_cache_list.get_free_cache();

        auto compdist = [&](IdxType pid) {
            auto *vec_ptr = tbstorage.get_vector(tbstorage.lookup(pid));
            return distance_handler->compute(query, (const char *)vec_ptr, dim);
        };

        priority_queue<pair<float, IdxType>> top_candidates; 
        auto record_top = [&](IdxType pid, float dist) {
            top_candidates.emplace(dist, pid);
            while (top_candidates.size() > K_prime) { top_candidates.pop(); }
        };

        auto &search_queue = search_cache->search_queue;
        auto &visited_set = search_cache->visited_set;
        search_queue.clear(); visited_set.clear();

        int cntr = 5;

        AttrTreeNode *ag_ptr = graphref.ag_ptr;
        if (ag_ptr->delegate_root)
            ag_ptr = ag_ptr->delegate_root;

        for (auto it : ag_ptr->edgelist_hashmap) {
            IdxType entry = it.first;
            if (visited_set.check(entry)) continue;
            visited_set.set(entry);

            float dist = compdist(entry);
            search_queue.insert(entry, dist, -1);
            record_top(entry, dist);

            --cntr;
            if (cntr <= 0) break;
        }

        while (search_queue.has_unexpanded_node()) {
            const ANNS::Candidate cur = search_queue.get_closest_unexpanded();

            std::vector<IdxType> edgelist_gref;
            { auto guard = ag_ptr->edgelist_hashmap.read_lock(cur.id); edgelist_gref = *guard.edges; }
            for (const IdxType nbr : edgelist_gref) {
                if (visited_set.check(nbr)) { continue; }
                visited_set.set(nbr);

                float dist = compdist(nbr);
                search_queue.insert(nbr, dist, cur.id);
                record_top(nbr, dist);  // search frontier is added to results
            }
        } // while (search_queue.has_unexpanded_node())

        search_cache_list.release_cache(search_cache);

        vector<pair<float, IdxType>> unfiltered_candidates;
        while (!top_candidates.empty()) {
            auto [dist, id] = top_candidates.top(); top_candidates.pop();
            unfiltered_candidates.emplace_back(dist, id);
        }

        vector<pair<float, IdxType>> filtered_candidates;
        for (int i = unfiltered_candidates.size() - 1; i >= 0; i--) {
            auto [dist, id] = unfiltered_candidates[i];
            if (predicate(id)) {
                filtered_candidates.emplace_back(dist, id);
            }
        }

        // cerr << "Found unfiltered_candidates = " << unfiltered_candidates.size() << endl;
        // cerr << "Found filtered_candidates = " << filtered_candidates.size() << endl;

        if (filtered_candidates.size() < K && unfiltered_candidates.size() == K_prime) {
            size_t new_K_prime = K_prime * 2;
            if (new_K_prime < K_prime) {
                cerr << "Integer overflow" << endl;
                abort();
            }
            K_prime = new_K_prime;
            continue;
        } else {
            for (auto [dist, id] : filtered_candidates) {
                if (seen.insert(id).second) { // first time seeing this id
                    all.emplace_back(id, dist);
                }
            }
            break;
        }
    }
}

void search_graph_reference_OOR(
        TableStorage &tbstorage,
        char *query,    // query_vec
        std::function<bool(int)> predicate,
        std::function<bool(int)> predicate_one, // We don't have functool in C++... Should we use a functor instead?
        int K, int n_entry,

        const unit_graph_ref &graphref,
        SearchCacheList &search_cache_list,
        int &n_cmps,
        vector<pair<IdxType, float>> &all,
        unordered_set<IdxType> &seen) {

    unit_graph_ref resolved = graphref;
    if (resolved.ag_ptr->delegate_root) {
        resolved.ag_ptr = resolved.ag_ptr->delegate_root;
    }

    if (resolved.ag_ptr->optional_HNSW_upper) {
        search_HNSW_reference_OOR(tbstorage, query, predicate, predicate_one, K, resolved, search_cache_list, n_cmps, all, seen);
        return;
    }

    auto search_cache = search_cache_list.get_free_cache();

    auto compdist = [&](IdxType pid) {
        ++n_cmps;
        auto *vec_ptr = tbstorage.get_vector(tbstorage.lookup(pid));
        return distance_handler->compute(query, (const char *)vec_ptr, dim);
    };

    priority_queue<pair<float, IdxType>> top_candidates;    // a dedicated list for valid results in post-filtering
    auto record_top = [&](IdxType pid, float dist) {    // only call if the pid passes predicate
        top_candidates.emplace(dist, pid);
        while (top_candidates.size() > K) { top_candidates.pop(); }
    };

    auto &search_queue = search_cache->search_queue;
    auto &visited_set = search_cache->visited_set;
    search_queue.clear(); visited_set.clear();

    int cntr = 5;

    AttrTreeNode *ag_ptr = resolved.ag_ptr;

    for (auto it : ag_ptr->edgelist_hashmap) {
        IdxType entry = it.first;
        if (visited_set.check(entry)) continue;
        visited_set.set(entry);
        float dist = compdist(entry);
        search_queue.insert(entry, dist, -1);
        if (predicate(entry)) { record_top(entry, dist); }
        --cntr;
        if (cntr <= 0) break;
    }

    while (search_queue.has_unexpanded_node()) {
        const ANNS::Candidate cur = search_queue.get_closest_unexpanded();

#ifdef JUST_LOOK_AT
        cerr << "Expanding " << cur.id << ", src=" << cur.extra_meta << " (" << range[q].boolean_string(tbstorage.get_attr(tbstorage.lookup(cur.id))) << ") with a distance of " << fixed << setprecision(3) << cur.distance << endl;
#endif

        // float *myattrs = tbstorage.get_attr(tbstorage.lookup(cur.id));

        std::vector<IdxType> edgelist_gref;
        { auto guard = ag_ptr->edgelist_hashmap.read_lock(cur.id); edgelist_gref = *guard.edges; }
        for (const IdxType nbr : edgelist_gref) {
            if (visited_set.check(nbr)) { continue; }
            visited_set.set(nbr);

            if (resolved.filter_strategy == unit_graph_ref::NOCHECK) {
                float dist = compdist(nbr);

                // candidate
                search_queue.insert(nbr, dist, cur.id);
                if (predicate(nbr)) {
                    record_top(nbr, dist);
                }
            } else if (resolved.filter_strategy == unit_graph_ref::CHECK_ONE) {
                if (predicate_one(nbr)) {
                    float dist = compdist(nbr);
                    search_queue.insert(nbr, dist, cur.id);
                    if (predicate(nbr)) {
                        record_top(nbr, dist);
                    }
                }
            } else if (resolved.filter_strategy == unit_graph_ref::CHECK_ALL) {
                if (predicate(nbr)) {
                    float dist = compdist(nbr);
                    search_queue.insert(nbr, dist, cur.id);
                    record_top(nbr, dist);
                }
            }
        }
    } // while (search_queue.has_unexpanded_node())

    ////////// end each unit graph

    while (!top_candidates.empty()) {
        auto [dist, id] = top_candidates.top(); top_candidates.pop();
        if (seen.insert(id).second) { // first time seeing this id
            all.emplace_back(id, dist);
        }

#ifdef JUST_LOOK_AT
        bool is_correct = false;
        for (int k = 0; k < K; k++) {
            IdxType ID_GroundTruth = gt[q * K + k].first;
            if (id == ID_GroundTruth) {
                cerr << "Groundtruth #" << k << " = '" << ID_GroundTruth << "' (" << range[q].boolean_string(tbstorage.get_attr(tbstorage.lookup(id))) << "), dist = " << gt[q * K + k].second << " is found! gid = " << ag_ptr->node_id << endl;
                found[k] = 1;
                is_correct = true;
            }
        }
        if (!is_correct) {
            cerr << "Search result '" << id << "' (" << range[q].boolean_string(tbstorage.get_attr(tbstorage.lookup(id))) << "), dist = " << dist << " is not an answer! gid = " << ag_ptr->node_id << endl;
        }
#endif
    }

    search_cache_list.release_cache(search_cache);
}

void search_HNSW_reference_OOR(
        TableStorage &tbstorage,
        char *query,    // query_vec
        std::function<bool(int)> predicate,
        std::function<bool(int)> predicate_one, // We don't have functool in C++... Should we use a functor instead?
        int K,

        const unit_graph_ref &graphref,
        SearchCacheList &search_cache_list,
        int &n_cmps,
        vector<pair<IdxType, float>> &all,
        unordered_set<IdxType> &seen) {

    unit_graph_ref resolved = graphref;
    if (resolved.ag_ptr->delegate_root) {
        resolved.ag_ptr = resolved.ag_ptr->delegate_root;
    }

    shared_ptr<HNSWUpperLayer> HNSW_upper = resolved.ag_ptr->optional_HNSW_upper;
    my_assert(HNSW_upper);

    auto search_cache = search_cache_list.get_free_cache();

    auto compdist = [&](IdxType pid) {
        ++n_cmps;
        auto *vec_ptr = tbstorage.get_vector(tbstorage.lookup(pid));
        return distance_handler->compute(query, (const char *)vec_ptr, dim);
    };

    priority_queue<pair<float, IdxType>> top_candidates;    // a dedicated list for valid results in post-filtering
    auto record_top = [&](IdxType pid, float dist) {    // only call if the pid passes predicate
        top_candidates.emplace(dist, pid);
        while (top_candidates.size() > K) { top_candidates.pop(); }
    };

    auto &search_queue = search_cache->search_queue;
    auto &visited_set = search_cache->visited_set;
    search_queue.clear(); visited_set.clear();

    AttrTreeNode *ag_ptr = resolved.ag_ptr;

/*  [hnswalg.h, searchStopConditionClosest()]
        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }
*/

    IdxType currObj = HNSW_upper->enterpoint_node_;
    float curdist = compdist(currObj);

    // pure greedy, non-beam routing on the upper levels
    for (int level = HNSW_upper->maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            vector<IdxType> &data_all_layers = HNSW_upper->edgelist.at(currObj);
            IdxType offset = HNSW_upper->n_links_per_element_ * (level - 1);
            int size = data_all_layers[offset];
            IdxType *datal = &data_all_layers[offset + 1];
            for (int i = 0; i < size; i++) {
                IdxType cand = datal[i];
                float d = compdist(cand);
                if (d < curdist) {
                    curdist = d;
                    currObj = cand;
                    changed = true;
                }
            }
        }
    }

    visited_set.set(currObj);
    search_queue.insert(currObj, curdist, -1);
    if (predicate(currObj)) { record_top(currObj, curdist); }
    // NOTE: NaviX also uses this single entry point like I do, but they have an additional logic:
    /**
     * // Add some random filtered nodes to candidates and results pq
     *  navix_add_filtered_nodes_to_candidates(
     *      qdis, candidates, res, vt, filter_id_map, 10);
     */
    // Here, "10" is the target number of nodes passing the filter. The above function simply scans all attributes in order to gather passing nodes.

    while (search_queue.has_unexpanded_node()) {
        const ANNS::Candidate cur = search_queue.get_closest_unexpanded();

        std::vector<IdxType> edgelist_gref;
        { auto guard = ag_ptr->edgelist_hashmap.read_lock(cur.id); edgelist_gref = *guard.edges; }
        for (const IdxType nbr : edgelist_gref) {
            if (visited_set.check(nbr)) { continue; }
            visited_set.set(nbr);

            if (resolved.filter_strategy == unit_graph_ref::NOCHECK) {
                float dist = compdist(nbr);

                // candidate
                search_queue.insert(nbr, dist, cur.id);
                if (predicate(nbr)) {
                    record_top(nbr, dist);
                }
            } else if (resolved.filter_strategy == unit_graph_ref::CHECK_ONE) {
                if (predicate_one(nbr)) {
                    float dist = compdist(nbr);
                    search_queue.insert(nbr, dist, cur.id);
                    if (predicate(nbr)) {
                        record_top(nbr, dist);
                    }
                }
            } else if (resolved.filter_strategy == unit_graph_ref::CHECK_ALL) {
                if (predicate(nbr)) {
                    float dist = compdist(nbr);
                    search_queue.insert(nbr, dist, cur.id);
                    record_top(nbr, dist);
                }
            }
        }
    } // while (search_queue.has_unexpanded_node())

    while (!top_candidates.empty()) {
        auto [dist, id] = top_candidates.top(); top_candidates.pop();
        if (seen.insert(id).second) { // first time seeing this id
            all.emplace_back(id, dist);
        }
    }

    search_cache_list.release_cache(search_cache);
}

void search_graph_reference_OOR_twophase(
        TableStorage &tbstorage,
        char *query,    // query_vec
        std::function<bool(int)> predicate,
        std::function<bool(int)> predicate_one, // We don't have functool in C++... Should we use a functor instead?
        int K, int n_entry,

        const unit_graph_ref &graphref,
        SearchCacheList &search_cache_list,
        vector<pair<IdxType, float>> &all,
        unordered_set<IdxType> &seen) {

    auto search_cache = search_cache_list.get_free_cache();

    int sort_by = graphref.attribute_graph_id;

    auto compdist = [&](IdxType pid) {
        auto *vec_ptr = tbstorage.get_vector(tbstorage.lookup(pid));
        return distance_handler->compute(query, (const char *)vec_ptr, dim);
    };

    priority_queue<pair<float, IdxType>> top_candidates;    // a dedicated list for valid results in post-filtering
    auto record_top = [&](IdxType pid, float dist) {    // only call if the pid passes predicate
        top_candidates.emplace(dist, pid);
        while (top_candidates.size() > K) { top_candidates.pop(); }
    };

    auto &search_queue = search_cache->search_queue;
    auto &visited_set = search_cache->visited_set;
    search_queue.clear(); visited_set.clear();

    int cntr = 5;
    for (auto it : graphref.ag_ptr->edgelist_hashmap) {
        IdxType entry = it.first;
        if (visited_set.check(entry)) continue;
        visited_set.set(entry);
        float dist = compdist(entry);
        search_queue.insert(entry, dist, -1);
        if (predicate(entry)) { record_top(entry, dist); }
        --cntr;
        if (cntr <= 0) break;
    }

    int w_CGIF = 3;
    std::queue<float> moving_window;
    float window_sum = 0;
    bool finished_descent = false;  // if true, then we use acorn; else we use OOR

    while (search_queue.has_unexpanded_node()) {
        const ANNS::Candidate cur = search_queue.get_closest_unexpanded();

        if (!finished_descent) {
            float this_dist = cur.distance;
            if (moving_window.size() == w_CGIF) {
                float window_avg = window_sum / moving_window.size();
                if (this_dist > window_avg) {
                    finished_descent = true;
                }

                float popped_dist = moving_window.front(); moving_window.pop();
                window_sum -= popped_dist;
                moving_window.push(this_dist);
                window_sum += this_dist;
            } else {
                moving_window.push(this_dist);
                window_sum += this_dist;
                my_assert(moving_window.size() <= w_CGIF);
            }
        }

        if (!finished_descent) {
            // this branch uses NOCHECK similar to OOR, for the descent phase.
            // NOTE: all neighbors are visited at the expense of high degree

            std::vector<IdxType> edgelist_gref;
            { auto guard = graphref.ag_ptr->edgelist_hashmap.read_lock(cur.id); edgelist_gref = *guard.edges; }
            for (const IdxType nbr : edgelist_gref) {
                if (visited_set.check(nbr)) { continue; }
                visited_set.set(nbr);

                float dist = compdist(nbr);

                // candidate
                search_queue.insert(nbr, dist);
                if (predicate(nbr)) {
                    record_top(nbr, dist);
                }
            }

            continue;
        }

        // NOTE: when we go out of the descent phase, should we purge the queue? skip the front of the queue that don't pass the filter?
        std::vector<IdxType> edgelist_gref;
        { auto guard = graphref.ag_ptr->edgelist_hashmap.read_lock(cur.id); edgelist_gref = *guard.edges; }
        for (const IdxType nbr : edgelist_gref) {
            if (visited_set.check(nbr)) { continue; }
            visited_set.set(nbr);

            // once descent is finished, we use 2-hop search
            if (predicate(nbr)) {
                float dist = compdist(nbr);
                record_top(nbr, dist);
                search_queue.insert(nbr, dist);
            }

            std::vector<IdxType> edgelist2;
            { auto guard = graphref.ag_ptr->edgelist_hashmap.read_lock(nbr); edgelist2 = *guard.edges; }
            for (const IdxType nbr2 : edgelist2) {
                if (visited_set.check(nbr2)) { continue; }
                visited_set.set(nbr2);

                if (predicate(nbr2)) {
                    float dist2 = compdist(nbr2);
                    record_top(nbr2, dist2);
                    search_queue.insert(nbr2, dist2);
                }
            }
        }
    } // while (search_queue.has_unexpanded_node())

    ////////// end each unit graph

    while (!top_candidates.empty()) {
        auto [dist, id] = top_candidates.top(); top_candidates.pop();
        if (seen.insert(id).second) { // first time seeing this id
            all.emplace_back(id, dist);
        }
    }

    search_cache_list.release_cache(search_cache);
}

void test_search_using_plan_separate_post(
        TableStorage &tbstorage,
        shared_ptr<ANNS::IStorage> query_storage,
        IQueryPlanner &query_planner,
        const vector<float> &attrs,
        const QueryCollection &range,
        int K, const vector<int> &ef_searches, int n_entry,
        pair<IdxType, float> *gt
    ) {

    int nq = query_storage->get_num_points();
    int nb = tbstorage.nb;
    int na = tbstorage.na;
    int dim = tbstorage.n_dim;

    auto results = new std::pair<ANNS::IdxType, float>[nq * K];

// #define DO_NOT_COUNT_PLANNING_TIME
#ifdef DO_NOT_COUNT_PLANNING_TIME
    for (int i = 0; i < nq; ++i) {
        query_planner.preprocess_for_id(range[i], i);
    }
#endif

    vector<double> qpss, recalls, cmpss;

    // to record average query planning overhead, we record the total execution time and the total query planning time of **one thread**
    double total_execution_time = 0.00;
    double total_planning_time_one_thread = 0.00;

    for (auto ef_search : ef_searches) {
        vector<int> num_cmps; num_cmps.resize(nq, 0);
        vector<int> n_edges_scanned; n_edges_scanned.resize(nq, 0);
        vector<int> n_edges_dup; n_edges_dup.resize(nq, 0);
        vector<int> n_edges_npass; n_edges_npass.resize(nq, 0);
        vector<int> n_edges_chosen; n_edges_chosen.resize(nq, 0);
        vector<int> n_hops; n_hops.resize(nq, 0);
        ANNS::SearchCacheList search_cache_list(num_threads + 5, nb, ef_search);

// #ifdef SOMEINDEX
#if 0
        // adaptive splitting
        auto ptr = dynamic_cast<Aug31Planner *>(&query_planner);
        if (ptr) {
            if (ef_search <= 100) {
                ptr->alpha = 1.0;   // penalize splitting
            } else if (ef_search <= 1000) {
                ptr->alpha = 0.4;
            } else {
                // encourage splitting: qps will be lower but we are able to reach the left upper corner of the trade-off curve
                ptr->alpha = -0.5;
            }
        }
#endif

        auto start_time = std::chrono::high_resolution_clock::now();
        double time_query_planner_ns = 0.0;

// #define JUST_LOOK_AT 31
#ifdef JUST_LOOK_AT
#error "support for run_and_report is removed"
        auto *planner = dynamic_cast<Aug31Planner *>(&query_planner);
        my_assert(planner != nullptr);
        auto report = planner->run_and_report(range[JUST_LOOK_AT], 100);
        cerr << endl << report << endl;
#endif

#ifndef JUST_LOOK_AT
        #pragma omp parallel for schedule (dynamic, 1)
        for (int q = 0; q < nq; q++) {
#else
        for (int q = JUST_LOOK_AT; q == JUST_LOOK_AT; ++q) {
#endif

            int n_cmps = 0;

            int one_predicate_k = 0;
            auto predicate_one = [&](int pid) {
                const float *attr_ptr = tbstorage.get_attr(tbstorage.lookup(pid));
                // return conj_clause.check_one(one_predicate_k, attr_ptr);
                return range.check_one(q, one_predicate_k, attr_ptr);
            };

            auto predicate = [&](int pid) {
                const float *attr_ptr = tbstorage.get_attr(tbstorage.lookup(pid));
                // return conj_clause.check(attr_ptr);
                return range.check(q, attr_ptr);
            };

            auto query = query_storage->get_vector(q);

            vector<vector<unit_graph_ref>> permutation_each_clause;

            for (const QueryType conj_clause : range.get_clauses(q)) {
                // get the permutation for this query
                auto t0 = std::chrono::high_resolution_clock::now();
    #ifdef DO_NOT_COUNT_PLANNING_TIME
                    const auto &partial_permutation = query_planner.retrieve_preprocessed(q);
    #else
                    auto partial_permutation = query_planner(conj_clause);
                    permutation_each_clause.push_back(move(partial_permutation));
    #endif
                auto t1 = std::chrono::high_resolution_clock::now();
                if (omp_get_thread_num() == 0) {
                    time_query_planner_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                }
            }

            vector<unit_graph_ref> permutation;
            if (permutation_each_clause.size() > 1) {
                auto ptr = dynamic_cast<Aug31Planner *>(&query_planner);
                if (ptr) {
                    // ptr->ef_search_reference = ef_search;
                    permutation = ptr->optimize_disjunctive_plan(permutation_each_clause);
                } else {    // simple wrapper planners that only return a single graph
                    permutation.push_back(permutation_each_clause[0][0]);
                }
            } else
                permutation = permutation_each_clause[0];

            vector<int> found(K, 0);

            vector<pair<IdxType, float>> all;
            unordered_set<IdxType> seen;
            ////////// start each unit graph
            my_assert(permutation.size() > 0);
            for (const auto &graphref : permutation) {
                one_predicate_k = graphref.attribute_graph_id;
                search_graph_reference_OOR(tbstorage, query, predicate, predicate_one, K, 5, graphref, search_cache_list, n_cmps, all, seen);
            }

#ifdef JUST_LOOK_AT
            for (int k = 0; k < K; k++) {
                IdxType ID_GroundTruth = gt[q * K + k].first;
                if (found[k] == 0) {
                    cerr << "Groundtruth #" << k << " = '" << ID_GroundTruth << "' (" << range[q].boolean_string(tbstorage.get_attr(tbstorage.lookup(ID_GroundTruth))) << "), dist = " << gt[q * K + k].second << " is NOT found!" << endl;
                }
            }
#endif

            std::sort(all.begin(), all.end(), [](auto& a, auto& b) { return a.second < b.second; });
            if ((int)all.size() > K) {
                all.resize(K);
            }

            for (int k = all.size(); k < K; k++) { results[q*K+k].first = -1; }
            for (int k = 0; k < all.size(); k++) { results[q*K+k] = all[k]; }
            num_cmps[q] = n_cmps;
        }

        auto time_cost = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
        auto time_cost_query_planner = time_query_planner_ns / 1e6; // convert to ms
        total_planning_time_one_thread += time_cost_query_planner;
        total_execution_time += time_cost / 1e3;
        double QPS = nq * 1e6 / time_cost;

        double avg_cmps = (double)accumulate(num_cmps.begin(), num_cmps.end(), 0) / nq;

        bool verbose = false;
        double recall = calculate_recall(gt, results, nq, K, verbose);

        if (true) {
            cerr << endl;
            cerr << "ef_search = " << ef_search << endl;
            cerr << "- Time cost (us): " << time_cost << endl;
            cerr << "- Time cost query planner (ms): " << time_cost_query_planner << " (" << 100.0 * time_cost_query_planner * num_threads / (time_cost / 1e3) << "%)" << endl;
            // cerr << "- Time cost entry finding (ms): " << time_cost_entry_finding << endl;
            cerr << "- Time per comp (ns): " << 1e6 * time_cost / (avg_cmps * nq) << endl;
            cerr << "- QPS: " << QPS << endl;
            cerr << "- Average number of comparisons: " << avg_cmps << endl;
            cerr << "- Recall: " << recall << endl;
        }
        if (flag3 >= 0 && ef_search == 100) {
            // additionally, save the recall statistics for each query row
            save_recall(gt, results, nq, K);
        }

        qpss.push_back(QPS);
        cmpss.push_back(avg_cmps);
        recalls.push_back(recall);
        // if (avg_cmps > 8000) break;
#ifdef BREAK_QPS
        if (QPS < 1000) break;
#endif
    }

    cerr << "Query planning overhead: "
         << total_planning_time_one_thread * num_threads << "ms / "
         << total_execution_time << "ms = "
         << 100.0 * total_planning_time_one_thread * num_threads / total_execution_time
         << "%" << endl;

    cout << "[";
    for (double q : cmpss) cout << q << ",";
    cout << "],[";
    for (double q : qpss) cout << q << ",";
    cout << "],[";
    for (double r : recalls) cout << r << ",";
    cout << "]," << endl;

    delete[] results;
};

void test_search_postfiltering_baseline(
        TableStorage &tbstorage,
        shared_ptr<ANNS::IStorage> query_storage,
        IQueryPlanner &query_planner,
        const vector<float> &attrs,
        const QueryCollection &range,
        int K, const vector<int> &ef_searches, int n_entry,
        pair<IdxType, float> *gt
    ) {

    int nq = query_storage->get_num_points();
    int nb = tbstorage.nb;
    int na = tbstorage.na;
    int dim = tbstorage.n_dim;

    auto results = new std::pair<ANNS::IdxType, float>[nq * K];

    // assume we know the selectivity.
    vector<double> s(nq, 1.0);
    #pragma omp parallel for schedule (dynamic, 1)
    for (int q = 0; q < nq; q++) {
        auto predicate = [&](int pid) {
            const float *attr_ptr = tbstorage.get_attr(tbstorage.lookup(pid));
            return range.check(q, attr_ptr);
        };
        size_t num_pass = 0;
        for (size_t i = 0; i < nb; i++) {
            if (predicate(i)) num_pass++;
        }
        s[q] = (double)num_pass / nb;
    }

    vector<double> qpss, recalls, cmpss;
    for (auto ef_search : ef_searches) {
        vector<int> num_cmps; num_cmps.resize(nq, 0);
        vector<int> n_edges_scanned; n_edges_scanned.resize(nq, 0);
        vector<int> n_edges_dup; n_edges_dup.resize(nq, 0);
        vector<int> n_edges_npass; n_edges_npass.resize(nq, 0);
        vector<int> n_edges_chosen; n_edges_chosen.resize(nq, 0);
        vector<int> n_hops; n_hops.resize(nq, 0);
        ANNS::SearchCacheList search_cache_list(num_threads + 5, nb, ef_search);

        auto start_time = std::chrono::high_resolution_clock::now();
        double time_query_planner_ns = 0.0;

        #pragma omp parallel for schedule (dynamic, 1)
        for (int q = 0; q < nq; q++) {
            auto predicate = [&](int pid) {
                const float *attr_ptr = tbstorage.get_attr(tbstorage.lookup(pid));
                return range.check(q, attr_ptr);
            };

            auto query = query_storage->get_vector(q);

            vector<vector<unit_graph_ref>> permutation_each_clause;

            for (const QueryType conj_clause : range.get_clauses(q)) {
                // get the permutation for this query
                auto t0 = std::chrono::high_resolution_clock::now();
                    auto partial_permutation = query_planner(conj_clause);
                    permutation_each_clause.push_back(move(partial_permutation));
                auto t1 = std::chrono::high_resolution_clock::now();
                if (omp_get_thread_num() == 0) {
                    time_query_planner_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                }
            }

            vector<unit_graph_ref> permutation;
            if (permutation_each_clause.size() > 1) {
                permutation.push_back(permutation_each_clause[0][0]);
            } else
                permutation = permutation_each_clause[0];

            vector<int> found(K, 0);

            vector<pair<IdxType, float>> all;
            unordered_set<IdxType> seen;
            my_assert(permutation.size() > 0);
            for (const auto &graphref : permutation) {
                // search_graph_reference_doubling_postfiltering(tbstorage, query, predicate, K, graphref, search_cache_list, all, seen);
                int K_prime = K / s[q];
                search_graph_reference_simple_postfiltering(tbstorage, query, predicate, K_prime, graphref, search_cache_list, num_cmps[q], all, seen);
            }

            std::sort(all.begin(), all.end(), [](auto& a, auto& b) { return a.second < b.second; });
            if ((int)all.size() > K) {
                all.resize(K);
            }

            for (int k = all.size(); k < K; k++) { results[q*K+k].first = -1; }
            for (int k = 0; k < all.size(); k++) { results[q*K+k] = all[k]; }
        }

        auto time_cost = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
        auto time_cost_query_planner = time_query_planner_ns / 1e6; // convert to ms
        double QPS = nq * 1e6 / time_cost;

        double avg_cmps = (double)accumulate(num_cmps.begin(), num_cmps.end(), 0) / nq;

        bool verbose = false;
        double recall = calculate_recall(gt, results, nq, K, verbose);

        if (true) {
            cerr << endl;
            cerr << "ef_search = " << ef_search << endl;
            cerr << "- Time cost (us): " << time_cost << endl;
            cerr << "- Time cost query planner (ms): " << time_cost_query_planner << endl;
            // cerr << "- Time cost entry finding (ms): " << time_cost_entry_finding << endl;
            cerr << "- Time per comp (ns): " << 1e6 * time_cost / (avg_cmps * nq) << endl;
            cerr << "- QPS: " << QPS << endl;
            cerr << "- Average number of comparisons: " << avg_cmps << endl;
            cerr << "- Recall: " << recall << endl;
        }
        if (flag3 >= 0 && ef_search == 100) {
            // additionally, save the recall statistics for each query row
            save_recall(gt, results, nq, K);
        }

        qpss.push_back(QPS);
        cmpss.push_back(avg_cmps);
        recalls.push_back(recall);
        // if (avg_cmps > 8000) break;
#ifdef BREAK_QPS
        if (QPS < 1000) break;
#endif
    }

    cout << "[";
    for (double q : cmpss) cout << q << ",";
    cout << "],[";
    for (double q : qpss) cout << q << ",";
    cout << "],[";
    for (double r : recalls) cout << r << ",";
    cout << "]," << endl;

    delete[] results;
};

void test_search_using_plan_separate_post_twophase(
        TableStorage &tbstorage,
        shared_ptr<ANNS::IStorage> query_storage,
        IQueryPlanner &query_planner,
        const vector<float> &attrs,
        const QueryCollection &range,
        int K, const vector<int> &ef_searches, int n_entry,
        pair<IdxType, float> *gt
    ) {

    int nq = query_storage->get_num_points();
    int nb = tbstorage.nb;
    int na = tbstorage.na;
    int dim = tbstorage.n_dim;

    auto results = new std::pair<ANNS::IdxType, float>[nq * K];

    vector<double> qpss, recalls, cmpss;
    for (auto ef_search : ef_searches) {
        vector<int> num_cmps; num_cmps.resize(nq, 0);
        vector<int> n_edges_scanned; n_edges_scanned.resize(nq, 0);
        vector<int> n_edges_dup; n_edges_dup.resize(nq, 0);
        vector<int> n_edges_npass; n_edges_npass.resize(nq, 0);
        vector<int> n_edges_chosen; n_edges_chosen.resize(nq, 0);
        vector<int> n_hops; n_hops.resize(nq, 0);
        ANNS::SearchCacheList search_cache_list(num_threads + 5, nb, ef_search);

        auto start_time = std::chrono::high_resolution_clock::now();
        double time_query_planner_ns = 0.0;

        #pragma omp parallel for schedule (dynamic, 1)
        for (int q = 0; q < nq; q++) {
            int one_predicate_k = 0;
            auto predicate_one = [&](int pid) {
                const float *attr_ptr = tbstorage.get_attr(tbstorage.lookup(pid));
                // return conj_clause.check_one(one_predicate_k, attr_ptr);
                return range.check_one(q, one_predicate_k, attr_ptr);
            };

            auto predicate = [&](int pid) {
                const float *attr_ptr = tbstorage.get_attr(tbstorage.lookup(pid));
                // return conj_clause.check(attr_ptr);
                return range.check(q, attr_ptr);
            };

            auto query = query_storage->get_vector(q);

            vector<vector<unit_graph_ref>> permutation_each_clause;

            for (const QueryType conj_clause : range.get_clauses(q)) {
                // get the permutation for this query
                auto t0 = std::chrono::high_resolution_clock::now();
                    auto partial_permutation = query_planner(conj_clause);
                    permutation_each_clause.push_back(move(partial_permutation));
                auto t1 = std::chrono::high_resolution_clock::now();
                if (omp_get_thread_num() == 0) {
                    time_query_planner_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                }
            }

            vector<unit_graph_ref> permutation;
            if (permutation_each_clause.size() > 1) {
                auto ptr = dynamic_cast<Aug31Planner *>(&query_planner);
                if (ptr) {
                    // ptr->ef_search_reference = ef_search;
                    permutation = ptr->optimize_disjunctive_plan(permutation_each_clause);
                } else {    // simple wrapper planners that only return a single graph
                    permutation.push_back(permutation_each_clause[0][0]);
                }
            } else
                permutation = permutation_each_clause[0];

            vector<int> found(K, 0);

            vector<pair<IdxType, float>> all;
            unordered_set<IdxType> seen;
            ////////// start each unit graph
            my_assert(permutation.size() > 0);
            for (const auto &graphref : permutation) {
                one_predicate_k = graphref.attribute_graph_id;
                search_graph_reference_OOR_twophase(tbstorage, query, predicate, predicate_one, K, 5, graphref, search_cache_list, all, seen);
            }

            std::sort(all.begin(), all.end(), [](auto& a, auto& b) { return a.second < b.second; });
            if ((int)all.size() > K) {
                all.resize(K);
            }

            for (int k = all.size(); k < K; k++) { results[q*K+k].first = -1; }
            for (int k = 0; k < all.size(); k++) { results[q*K+k] = all[k]; }
        }

        auto time_cost = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
        auto time_cost_query_planner = time_query_planner_ns / 1e6; // convert to ms
        double QPS = nq * 1e6 / time_cost;

        double avg_cmps = (double)accumulate(num_cmps.begin(), num_cmps.end(), 0) / nq;

        bool verbose = false;
        double recall = calculate_recall(gt, results, nq, K, verbose);

        if (true) {
            cerr << endl;
            cerr << "ef_search = " << ef_search << endl;
            cerr << "- Time cost (us): " << time_cost << endl;
            cerr << "- Time cost query planner (ms): " << time_cost_query_planner << endl;
            // cerr << "- Time cost entry finding (ms): " << time_cost_entry_finding << endl;
            cerr << "- Time per comp (ns): " << 1e6 * time_cost / (avg_cmps * nq) << endl;
            cerr << "- QPS: " << QPS << endl;
            cerr << "- Average number of comparisons: " << avg_cmps << endl;
            cerr << "- Recall: " << recall << endl;
        }
        if (flag3 >= 0 && ef_search == 100) {
            // additionally, save the recall statistics for each query row
            save_recall(gt, results, nq, K);
        }

        qpss.push_back(QPS);
        cmpss.push_back(avg_cmps);
        recalls.push_back(recall);
        // if (avg_cmps > 8000) break;
#ifdef BREAK_QPS
        if (QPS < 1000) break;
#endif
    }

    cout << "[";
    for (double q : cmpss) cout << q << ",";
    cout << "],[";
    for (double q : qpss) cout << q << ",";
    cout << "],[";
    for (double r : recalls) cout << r << ",";
    cout << "]," << endl;

    delete[] results;
};

void test_search_acorn(
        TableStorage &tbstorage,
        shared_ptr<ANNS::IStorage> query_storage,
        shared_ptr<Graph> acorn_graph, int M_ACORN, int Mb_ACORN, int gamma_ACORN,
        const vector<float> &attrs,                 // for predicate = [&]()
        const QueryCollection &range,
        int K, const vector<int> &ef_searches, int n_entry,
        pair<IdxType, float> *gt,
        vector<int> *bitmap
    ) {

    int nq = query_storage->get_num_points();
    int nb = tbstorage.nb;
    int na = tbstorage.na;
    int dim = tbstorage.n_dim;

    auto results = new std::pair<ANNS::IdxType, float>[nq * K];
    std::random_device rd;
    std::mt19937 gen(rd());

    vector<double> qpss, recalls, cmpss;
    for (auto ef_search : ef_searches) {
        vector<int> num_cmps; num_cmps.resize(nq, 0);
        vector<int> n_edges_scanned; n_edges_scanned.resize(nq, 0);
        vector<int> n_edges_dup; n_edges_dup.resize(nq, 0);
        vector<int> n_edges_npass; n_edges_npass.resize(nq, 0);
        vector<int> n_edges_chosen; n_edges_chosen.resize(nq, 0);
        vector<int> n_hops; n_hops.resize(nq, 0);
        ANNS::SearchCacheList search_cache_list(num_threads + 5, nb, ef_search);

        auto start_time = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule (dynamic, 1)
        for (int q = 0; q < nq; q++) {

            std::function<bool(int)> pred_bitmap = [&](int pid) {
                return (bool)(bitmap->at(pid));
            };
            std::function<bool(int)> predicate = [&](int pid) {
                const float *attr_ptr = tbstorage.get_attr(tbstorage.lookup(pid));
                // return range[q].check(attr_ptr);
                return range.check(q, attr_ptr);
            };
            if (bitmap != nullptr)
                predicate = pred_bitmap;

            vector<IdxType> entries;
            {
                std::uniform_int_distribution<IdxType> dis(0, nb - 1);
                entries.reserve(n_entry);
                for (int i = 0; i < n_entry; i++) {
                    entries.push_back(dis(gen));
                }
            }

            priority_queue<pair<float, IdxType>> top_candidates;    // a dedicated list for valid results in post-filtering
            auto query = query_storage->get_vector(q);
            int n_cmps = 0;
            int ne_s=0, ne_d=0, ne_np=0, ne_c=0;
            int n_hops_ = 0;
            auto compdist = [&](IdxType pid) {
                n_cmps++;
                auto *vec_ptr = tbstorage.get_vector(tbstorage.lookup(pid));
                return distance_handler->compute(query, (const char *)vec_ptr, dim);
            };
            auto record_top = [&](IdxType pid, float dist) {    // only call if the pid passes predicate
                top_candidates.emplace(dist, pid);
                while (top_candidates.size() > K) { top_candidates.pop(); }
            };

            auto search_cache = search_cache_list.get_free_cache();
            auto &search_queue = search_cache->search_queue;
            auto &visited_set = search_cache->visited_set;
            search_queue.clear(); visited_set.clear();

            for (IdxType entry : entries) {
                float dist = compdist(entry);
                search_queue.insert(entry, dist);
                if (!visited_set.check(entry) && predicate(entry))
                    record_top(entry, dist);
                visited_set.set(entry);
            }

            while (search_queue.has_unexpanded_node()) {
                ++n_hops_;
                const ANNS::Candidate cur = search_queue.get_closest_unexpanded();

                int num_found = 0;
                bool keep_expanding = true;

                for (int j = 0; keep_expanding && j < acorn_graph->neighbors[cur.id].size(); ++j) {
                    auto nbr = acorn_graph->neighbors[cur.id][j];

                    // scanned
                    ++ne_s;
                    // duplicate with previously seen edges
                    if (visited_set.check(nbr)) { ++ne_d; continue; }
                    visited_set.set(nbr);

                    if (!predicate(nbr)) {
                        // not pass filter
                        ++ne_np;
                    } else {
                        // candidate
                        ++ne_c;
                        ++num_found;    // # of chosen neighbors for the current hop
                        float dist = compdist(nbr);
                        record_top(nbr, dist);
                        search_queue.insert(nbr, dist);

                        if (num_found >= M_ACORN * 2) {
                            keep_expanding = false;
                            break;
                        }
                    }

                    // regardless of filter pass, we do two hop when we reach the Mb_ACORN threshold without collecting enough neighbors
                    bool do_2hop_for_this_node = (gamma_ACORN == 1 || (keep_expanding && (j >= Mb_ACORN)));

                    if (do_2hop_for_this_node) {
                        for (int k = 0; k < acorn_graph->neighbors[nbr].size(); ++k) {
                            auto nbr2 = acorn_graph->neighbors[nbr][k];

                            // scanned
                            ++ne_s;
                            // duplicate with previously seen edges
                            if (visited_set.check(nbr2)) { ++ne_d; continue; }
                            visited_set.set(nbr2);

                            if (!predicate(nbr2)) {
                                // not pass filter
                                ++ne_np;
                            } else {
                                // candidate
                                ++ne_c;
                                ++num_found;
                                float dist = compdist(nbr2);
                                record_top(nbr2, dist);
                                search_queue.insert(nbr2, dist);

                                if (num_found >= M_ACORN * 2) {
                                    keep_expanding = false;
                                    break;
                                }
                            }
                        }
                    } // if (2hop activated)
                } // for neighbor in neighborhood
            } // while unexpanded node

            n_edges_scanned[q] = ne_s;
            n_edges_dup[q] = ne_d;
            n_edges_npass[q] = ne_np;
            n_edges_chosen[q] = ne_c;

            int ptr = top_candidates.size();
            for (int k = ptr; k < K; k++) { results[q*K+k].first = -1; }
            --ptr;
            for (; ptr >= 0; --ptr) {
                auto tmp = top_candidates.top(); top_candidates.pop();
                results[q*K+ptr] = make_pair(tmp.second, tmp.first);
            }
            num_cmps[q] = n_cmps;
            n_hops[q] = n_hops_;

            search_cache_list.release_cache(search_cache);
        }

        auto time_cost = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
        double QPS = nq * 1e6 / time_cost;

        double avg_cmps = (double)accumulate(num_cmps.begin(), num_cmps.end(), 0) / nq;

        double recall = calculate_recall(gt, results, nq, K, false);

        if (true) {
            cerr << endl;
            cerr << "ef_search = " << ef_search << endl;
            cerr << "- Time cost (us): " << time_cost << endl;
            cerr << "- Time per comp (ns): " << 1e6 * time_cost / (avg_cmps * nq) << endl;
            cerr << "- QPS: " << QPS << endl;
            cerr << "- Average number of comparisons: " << avg_cmps << endl;
            cerr << "- Recall: " << recall << endl;
        }

        qpss.push_back(QPS);
        cmpss.push_back(avg_cmps);
        recalls.push_back(recall);
        // if (avg_cmps > 8000) break;
#ifdef BREAK_QPS
        if (QPS < 1000) break;
#endif
    }

    cout << "[";
    for (double q : cmpss) cout << q << ",";
    cout << "],[";
    for (double q : qpss) cout << q << ",";
    cout << "],[";
    for (double r : recalls) cout << r << ",";
    cout << "]," << endl;

    delete[] results;
}

void test_search_acorn_twophase(
        TableStorage &tbstorage,
        shared_ptr<ANNS::IStorage> query_storage,
        shared_ptr<Graph> acorn_graph, int M_ACORN, int Mb_ACORN, int gamma_ACORN,
        const vector<float> &attrs,                 // for predicate = [&]()
        const QueryCollection &range,
        int K, const vector<int> &ef_searches, int n_entry,
        pair<IdxType, float> *gt,
        vector<int> *bitmap
    ) {

    int nq = query_storage->get_num_points();
    int nb = tbstorage.nb;
    int na = tbstorage.na;
    int dim = tbstorage.n_dim;

    auto results = new std::pair<ANNS::IdxType, float>[nq * K];
    std::random_device rd;
    std::mt19937 gen(rd());

    vector<double> qpss, recalls, cmpss;
    for (auto ef_search : ef_searches) {
        vector<int> num_cmps; num_cmps.resize(nq, 0);
        vector<int> n_edges_scanned; n_edges_scanned.resize(nq, 0);
        vector<int> n_edges_dup; n_edges_dup.resize(nq, 0);
        vector<int> n_edges_npass; n_edges_npass.resize(nq, 0);
        vector<int> n_edges_chosen; n_edges_chosen.resize(nq, 0);
        vector<int> n_hops; n_hops.resize(nq, 0);
        ANNS::SearchCacheList search_cache_list(num_threads + 5, nb, ef_search);

        auto start_time = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule (dynamic, 1)
        for (int q = 0; q < nq; q++) {

            std::function<bool(int)> pred_bitmap = [&](int pid) {
                return (bool)(bitmap->at(pid));
            };
            std::function<bool(int)> predicate = [&](int pid) {
                const float *attr_ptr = tbstorage.get_attr(tbstorage.lookup(pid));
                // return range[q].check(attr_ptr);
                return range.check(q, attr_ptr);
            };
            if (bitmap != nullptr)
                predicate = pred_bitmap;

            vector<IdxType> entries;
            {
                std::uniform_int_distribution<IdxType> dis(0, nb - 1);
                entries.reserve(n_entry);
                for (int i = 0; i < n_entry; i++) {
                    entries.push_back(dis(gen));
                }
            }

            priority_queue<pair<float, IdxType>> top_candidates;    // a dedicated list for valid results in post-filtering
            auto query = query_storage->get_vector(q);
            int n_cmps = 0;
            int ne_s=0, ne_d=0, ne_np=0, ne_c=0;
            int n_hops_ = 0;
            auto compdist = [&](IdxType pid) {
                n_cmps++;
                auto *vec_ptr = tbstorage.get_vector(tbstorage.lookup(pid));
                return distance_handler->compute(query, (const char *)vec_ptr, dim);
            };
            auto record_top = [&](IdxType pid, float dist) {    // only call if the pid passes predicate
                top_candidates.emplace(dist, pid);
                while (top_candidates.size() > K) { top_candidates.pop(); }
            };

            auto search_cache = search_cache_list.get_free_cache();
            auto &search_queue = search_cache->search_queue;
            auto &visited_set = search_cache->visited_set;
            search_queue.clear(); visited_set.clear();

            for (IdxType entry : entries) {
                float dist = compdist(entry);
                search_queue.insert(entry, dist);
                if (!visited_set.check(entry) && predicate(entry))
                    record_top(entry, dist);
                visited_set.set(entry);
            }

            // the criteria for entering finished_descent is taken from CGIF paper
            int w_CGIF = 3;
            std::queue<float> moving_window;
            float window_sum = 0;
            bool finished_descent = false;  // if true, then we use acorn; else we use OOR

            while (search_queue.has_unexpanded_node()) {
                ++n_hops_;
                const ANNS::Candidate cur = search_queue.get_closest_unexpanded();

                if (!finished_descent) {
                    float this_dist = cur.distance;
                    if (moving_window.size() == w_CGIF) {
                        float window_avg = window_sum / moving_window.size();
                        if (this_dist > window_avg) {
                            finished_descent = true;
                        }

                        float popped_dist = moving_window.front(); moving_window.pop();
                        window_sum -= popped_dist;
                        moving_window.push(this_dist);
                        window_sum += this_dist;
                    } else {
                        moving_window.push(this_dist);
                        window_sum += this_dist;
                        my_assert(moving_window.size() <= w_CGIF);
                    }
                }

                if (!finished_descent) {
                    // this branch uses NOCHECK similar to OOR, for the descent phase.
                    // NOTE: all neighbors are visited at the expense of high degree
                    const auto &neighbors = acorn_graph->neighbors[cur.id];
                    for (int j = 0; j < neighbors.size(); ++j) {
                        auto nbr = neighbors[j];
                        if (visited_set.check(nbr)) continue;
                        visited_set.set(nbr);
                        float dist = compdist(nbr);
                        search_queue.insert(nbr, dist);
                        if (predicate(nbr)) {
                            record_top(nbr, dist);
                        }
                    }

                    continue;
                }

                int num_found = 0;
                bool keep_expanding = true;

                for (int j = 0; keep_expanding && j < acorn_graph->neighbors[cur.id].size(); ++j) {
                    auto nbr = acorn_graph->neighbors[cur.id][j];

                    // scanned
                    ++ne_s;
                    // duplicate with previously seen edges
                    if (visited_set.check(nbr)) { ++ne_d; continue; }
                    visited_set.set(nbr);

                    if (!predicate(nbr)) {
                        // not pass filter
                        ++ne_np;
                    } else {
                        // candidate
                        ++ne_c;
                        ++num_found;    // # of chosen neighbors for the current hop
                        float dist = compdist(nbr);
                        record_top(nbr, dist);
                        search_queue.insert(nbr, dist);

                        if (num_found >= M_ACORN * 2) {
                            keep_expanding = false;
                            break;
                        }
                    }

                    // regardless of filter pass, we do two hop when we reach the Mb_ACORN threshold without collecting enough neighbors
                    bool do_2hop_for_this_node = (gamma_ACORN == 1 || (keep_expanding && (j >= Mb_ACORN)));

                    if (do_2hop_for_this_node) {
                        for (int k = 0; k < acorn_graph->neighbors[nbr].size(); ++k) {
                            auto nbr2 = acorn_graph->neighbors[nbr][k];

                            // scanned
                            ++ne_s;
                            // duplicate with previously seen edges
                            if (visited_set.check(nbr2)) { ++ne_d; continue; }
                            visited_set.set(nbr2);

                            if (!predicate(nbr2)) {
                                // not pass filter
                                ++ne_np;
                            } else {
                                // candidate
                                ++ne_c;
                                ++num_found;
                                float dist = compdist(nbr2);
                                record_top(nbr2, dist);
                                search_queue.insert(nbr2, dist);

                                if (num_found >= M_ACORN * 2) {
                                    keep_expanding = false;
                                    break;
                                }
                            }
                        }
                    } // if (2hop activated)
                } // for neighbor in neighborhood
            } // while unexpanded node

            n_edges_scanned[q] = ne_s;
            n_edges_dup[q] = ne_d;
            n_edges_npass[q] = ne_np;
            n_edges_chosen[q] = ne_c;

            int ptr = top_candidates.size();
            for (int k = ptr; k < K; k++) { results[q*K+k].first = -1; }
            --ptr;
            for (; ptr >= 0; --ptr) {
                auto tmp = top_candidates.top(); top_candidates.pop();
                results[q*K+ptr] = make_pair(tmp.second, tmp.first);
            }
            num_cmps[q] = n_cmps;
            n_hops[q] = n_hops_;

            search_cache_list.release_cache(search_cache);
        }

        auto time_cost = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
        double QPS = nq * 1e6 / time_cost;

        double avg_cmps = (double)accumulate(num_cmps.begin(), num_cmps.end(), 0) / nq;

        double recall = calculate_recall(gt, results, nq, K, false);

        if (true) {
            cerr << endl;
            cerr << "ef_search = " << ef_search << endl;
            cerr << "- Time cost (us): " << time_cost << endl;
            cerr << "- Time per comp (ns): " << 1e6 * time_cost / (avg_cmps * nq) << endl;
            cerr << "- QPS: " << QPS << endl;
            cerr << "- Average number of comparisons: " << avg_cmps << endl;
            cerr << "- Recall: " << recall << endl;
        }

        qpss.push_back(QPS);
        cmpss.push_back(avg_cmps);
        recalls.push_back(recall);
        // if (avg_cmps > 8000) break;
#ifdef BREAK_QPS
        if (QPS < 1000) break;
#endif
    }

    cout << "[";
    for (double q : cmpss) cout << q << ",";
    cout << "],[";
    for (double q : qpss) cout << q << ",";
    cout << "],[";
    for (double r : recalls) cout << r << ",";
    cout << "]," << endl;

    delete[] results;
}

void test_search_navix(
        TableStorage &tbstorage,
        shared_ptr<ANNS::IStorage> query_storage,
        const unit_graph_ref &graphref,
        int M,
        const vector<float> &attrs,                 // for predicate = [&]()
        std::function<bool(int, int)> predicate_wrap,
        // const QueryCollection &range,
        int K, const vector<int> &ef_searches, int n_entry,
        pair<IdxType, float> *gt,
        int mode
    ) {

    int nq = query_storage->get_num_points();
    int nb = tbstorage.nb;
    int na = tbstorage.na;
    int dim = tbstorage.n_dim;

    auto results = new std::pair<ANNS::IdxType, float>[nq * K];
    std::random_device rd;
    std::mt19937 gen(rd());

    vector<double> qpss, recalls, cmpss;
    for (auto ef_search : ef_searches) {
        vector<int> num_cmps; num_cmps.resize(nq, 0);
        vector<int> n_hops; n_hops.resize(nq, 0);
        ANNS::SearchCacheList search_cache_list(num_threads + 5, nb, ef_search);

        auto start_time = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule (dynamic, 1)
        for (int q = 0; q < nq; q++) {

            // auto predicate = [&](int pid) {
            //     const float *attr_ptr = tbstorage.get_attr(tbstorage.lookup(pid));
            //     return range.check(q, attr_ptr);
            // };
            auto predicate = [&](int pid) {
                return predicate_wrap(q, pid);
            };

            priority_queue<pair<float, IdxType>> top_candidates;    // a dedicated list for valid results in post-filtering
            auto query = query_storage->get_vector(q);
            int n_cmps = 0;
            int ne_s=0, ne_d=0, ne_np=0, ne_c=0;
            int n_hops_ = 0;
            auto compdist = [&](IdxType pid) {
                n_cmps++;
                auto *vec_ptr = tbstorage.get_vector(tbstorage.lookup(pid));
                return distance_handler->compute(query, (const char *)vec_ptr, dim);
            };
            auto record_top = [&](IdxType pid, float dist) {    // only call if the pid passes predicate
                top_candidates.emplace(dist, pid);
                while (top_candidates.size() > K) { top_candidates.pop(); }
            };

            auto search_cache = search_cache_list.get_free_cache();
            auto &search_queue = search_cache->search_queue;
            auto &visited_set = search_cache->visited_set;
            search_queue.clear(); visited_set.clear();

            int cntr = n_entry;
            for (auto it : graphref.ag_ptr->edgelist_hashmap) {
                IdxType entry = it.first;
                if (visited_set.check(entry)) continue;
                visited_set.set(entry);
                float dist = compdist(entry);
                search_queue.insert(entry, dist, -1);
                if (predicate(entry)) { record_top(entry, dist); }
                --cntr;
                if (cntr <= 0) break;
            }

            while (search_queue.has_unexpanded_node()) {
                ++n_hops_;
                const ANNS::Candidate cur = search_queue.get_closest_unexpanded();

                int visitedSetSize = 0;

                int pass_count = 0;
                std::vector<IdxType> edgelist_gref;
                { auto guard = graphref.ag_ptr->edgelist_hashmap.read_lock(cur.id); edgelist_gref = *guard.edges; }
                int nbr_count = edgelist_gref.size();
                int filterNbrsToFind = nbr_count;
                vector<int> pass_mask(nbr_count, 0);

                int cntr = -1;
                for (const IdxType nbr : edgelist_gref) {
                    ++cntr;
                    bool pass = predicate(nbr);
                    if (pass) visitedSetSize++;
                    if (visited_set.check(nbr)) { continue; }
                    pass_mask[cntr] = pass;
                    if (pass_mask[cntr])
                        pass_count++;
                } // for neighbor in neighborhood

                double sel_local = (double)pass_count / nbr_count;
                if (mode == 1 || (mode == 0 && sel_local >= 0.4)) { // value 0.4 taken from git repo
                // if (mode == 1 || (mode == 0 && sel_local >= 0.1)) { // NOTE:
                    // onehop-s
                    int cntr = -1;
                    for (const IdxType nbr : edgelist_gref) {
                        ++cntr;
                        if (!pass_mask[cntr]) { continue; }
                        visited_set.set(nbr);
                        float dist = compdist(nbr);
                        record_top(nbr, dist);
                        search_queue.insert(nbr, dist);
                    }
                } else {
                    double esv = sel_local * (M + 1) * M;
                    double lf = 3;  // leniency
                    if (mode == 2 || (mode == 0 && esv < M * lf)) {
                    // if (mode == 2 || false) {       // NOTE:
                        // blind
                        int cntr = -1;
                        for (const IdxType nbr : edgelist_gref) {
                            ++cntr;
                            if (visited_set.check(nbr) == false && pass_mask[cntr]) {
                                visited_set.set(nbr);
                                float dist = compdist(nbr);
                                record_top(nbr, dist);
                                search_queue.insert(nbr, dist);
                            }

                            std::vector<IdxType> edgelist2;
                            { auto guard = graphref.ag_ptr->edgelist_hashmap.read_lock(nbr); edgelist2 = *guard.edges; }
                            for (const IdxType nbr2 : edgelist2) {
                                        if (visitedSetSize >= filterNbrsToFind) {
                                            break;
                                        }
                                if (visited_set.check(nbr2)) continue;
                                if (predicate(nbr2)) {
                                        ++visitedSetSize;
                                    visited_set.set(nbr2);
                                    float dist = compdist(nbr2);
                                    record_top(nbr2, dist);
                                    search_queue.insert(nbr2, dist);
                                }
                            }
                                        if (visitedSetSize >= filterNbrsToFind) {
                                            break;
                                        }
                        }
                    } else {
                        // directed
                        std::priority_queue<pair<float, IdxType>> nbrsToExplore;    // closest first
                        int cntr = -1;
                        for (const IdxType nbr : edgelist_gref) {
                            ++cntr;
                            if (visited_set.check(nbr)) continue;
                            visited_set.set(nbr);
                            float dist = compdist(nbr); // pay upfront to compute 1-hop distances
                            nbrsToExplore.emplace(-dist, nbr);
                            if (pass_mask[cntr]) {
                                record_top(nbr, dist);
                                search_queue.insert(nbr, dist);
                            }
                        }
                        while (!nbrsToExplore.empty()) {
                            auto neighbor = nbrsToExplore.top();
                            nbrsToExplore.pop();    // Note: nbrsToExplore can be out of range
                            if (visitedSetSize >= filterNbrsToFind) {
                                break;
                            }
                            // if (visited_set.check(neighbor.second)) {
                            //     continue;
                            // }
                            // visited_set.set(neighbor.second);    // already set

                            std::vector<IdxType> edgelist2;
                            { auto guard = graphref.ag_ptr->edgelist_hashmap.read_lock(neighbor.second); edgelist2 = *guard.edges; }
                            for (const IdxType nbr2 : edgelist2) {
                                bool pass = predicate(nbr2);
                                if (pass) visitedSetSize++;
                                if (visited_set.check(nbr2)) continue;
                                if (pass) {
                                    visited_set.set(nbr2);
                                    float dist = compdist(nbr2);
                                    record_top(nbr2, dist);
                                    search_queue.insert(nbr2, dist);
                                }
                            }
                        }
                    }
                }
            } // while unexpanded node

            int ptr = top_candidates.size();
            for (int k = ptr; k < K; k++) { results[q*K+k].first = -1; }
            --ptr;
            for (; ptr >= 0; --ptr) {
                auto tmp = top_candidates.top(); top_candidates.pop();
                results[q*K+ptr] = make_pair(tmp.second, tmp.first);
            }
            num_cmps[q] = n_cmps;
            n_hops[q] = n_hops_;

            search_cache_list.release_cache(search_cache);
        }

        auto time_cost = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
        double QPS = nq * 1e6 / time_cost;

        double avg_cmps = (double)accumulate(num_cmps.begin(), num_cmps.end(), 0) / nq;

        double recall = calculate_recall(gt, results, nq, K, false);

        if (true) {
            cerr << endl;
            cerr << "ef_search = " << ef_search << endl;
            cerr << "- Time cost (us): " << time_cost << endl;
            cerr << "- Time per comp (ns): " << 1e6 * time_cost / (avg_cmps * nq) << endl;
            cerr << "- QPS: " << QPS << endl;
            cerr << "- Average number of comparisons: " << avg_cmps << endl;
            cerr << "- Recall: " << recall << endl;
        }

        qpss.push_back(QPS);
        cmpss.push_back(avg_cmps);
        recalls.push_back(recall);
        // if (avg_cmps > 8000) break;
//         if (QPS < 10) break;    // seriously... you guys are taking 2-hop too far...
// #ifdef BREAK_QPS
//         if (QPS < 1000) break;
// #endif
    }

    cout << "[";
    for (double q : cmpss) cout << q << ",";
    cout << "],[";
    for (double q : qpss) cout << q << ",";
    cout << "],[";
    for (double r : recalls) cout << r << ",";
    cout << "]," << endl;

    delete[] results;
}

void test_search_navix_global(
        TableStorage &tbstorage,
        shared_ptr<ANNS::IStorage> query_storage,
        const unit_graph_ref &graphref,
        int M,
        const vector<float> &attrs,                 // for predicate = [&]()
        std::function<bool(int, int)> predicate_wrap,
        // const QueryCollection &range,
        int K, const vector<int> &ef_searches, int n_entry,
        pair<IdxType, float> *gt,
        int mode
    ) {

    int nq = query_storage->get_num_points();
    int nb = tbstorage.nb;
    int na = tbstorage.na;
    int dim = tbstorage.n_dim;

    auto results = new std::pair<ANNS::IdxType, float>[nq * K];
    std::random_device rd;
    std::mt19937 gen(rd());

    vector<double> sel_globals(nq, 0);
    if (mode == 0)
        for (int q = 0; q < nq; q++) {
            size_t total_pass_count = 0;
            #pragma omp parallel for
            for (size_t i = 0; i < nb; i++) {
                if (predicate_wrap(q, i)) {
                    ++total_pass_count;
                }
            }
            double sel_global_ = (double)total_pass_count / nb;
            sel_globals[q] = sel_global_;
        }

    vector<double> qpss, recalls, cmpss;
    for (auto ef_search : ef_searches) {
        vector<int> num_cmps; num_cmps.resize(nq, 0);
        vector<int> n_hops; n_hops.resize(nq, 0);
        ANNS::SearchCacheList search_cache_list(num_threads + 5, nb, ef_search);

        auto start_time = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for schedule (dynamic, 1)
        for (int q = 0; q < nq; q++) {

            auto predicate = [&](int pid) -> bool {
                return predicate_wrap(q, pid);
            };

            double sel_global = sel_globals[q];

            priority_queue<pair<float, IdxType>> top_candidates;
            auto query = query_storage->get_vector(q);
            int n_cmps = 0;
            int ne_s=0, ne_d=0, ne_np=0, ne_c=0;
            int n_hops_ = 0;
            auto compdist = [&](IdxType pid) {
                n_cmps++;
                auto *vec_ptr = tbstorage.get_vector(tbstorage.lookup(pid));
                return distance_handler->compute(query, (const char *)vec_ptr, dim);
            };
            auto record_top = [&](IdxType pid, float dist) {    // only call if the pid passes predicate
                top_candidates.emplace(dist, pid);
                while (top_candidates.size() > K) { top_candidates.pop(); }
            };

            auto search_cache = search_cache_list.get_free_cache();
            auto &search_queue = search_cache->search_queue;
            auto &visited_set = search_cache->visited_set;
            search_queue.clear(); visited_set.clear();

            int cntr = n_entry;
            for (auto it : graphref.ag_ptr->edgelist_hashmap) {
                IdxType entry = it.first;
                if (visited_set.check(entry)) continue;
                visited_set.set(entry);
                float dist = compdist(entry);
                search_queue.insert(entry, dist, -1);
                if (predicate(entry)) { record_top(entry, dist); }
                --cntr;
                if (cntr <= 0) break;
            }

            while (search_queue.has_unexpanded_node()) {
                ++n_hops_;
                const ANNS::Candidate cur = search_queue.get_closest_unexpanded();

                int visitedSetSize = 0;

                std::vector<IdxType> edgelist_gref;
                { auto guard = graphref.ag_ptr->edgelist_hashmap.read_lock(cur.id); edgelist_gref = *guard.edges; }
                int nbr_count = edgelist_gref.size();
                int filterNbrsToFind = nbr_count;

                // if (mode == 1 || (mode == 0 && sel_global >= 0.4)) { // value 0.4 taken from git repo
                // if (mode == 1 || (mode == 0 && sel_global >= 0.1)) { // NOTE:
                if (mode == 1 || (mode == 0 && sel_global >= 0.3)) { // Nov 24: a second time tuning this
                    // onehop-s
                    for (const IdxType nbr : edgelist_gref) {
                        if (visited_set.check(nbr)) continue;
                        visited_set.set(nbr);           // swap these two
                        if (!predicate(nbr)) continue;  // swap these two
                        float dist = compdist(nbr);
                        record_top(nbr, dist);
                        search_queue.insert(nbr, dist);
                    }
                } else {

        for (const IdxType nbr : edgelist_gref) {
            bool pass = predicate(nbr);
            if (pass) visitedSetSize++;
        } // for neighbor in neighborhood

                    // double esv = sel_local * (M + 1) * M;
                    // double lf = 3;  // leniency
                    // if (mode == 2 || (mode == 0 && esv < M * lf)) {
                    // if (mode == 2 || false) {       // NOTE:
                    if (mode == 2 || (mode == 0 && sel_global < 0.03)) {       // Nov 24: a second time tuning this
                        // blind
                        for (const IdxType nbr : edgelist_gref) {
                            if (visited_set.check(nbr)) continue;
                            if (predicate(nbr)) {
                                visited_set.set(nbr);
                                float dist = compdist(nbr);
                                record_top(nbr, dist);
                                search_queue.insert(nbr, dist);
                            }

                            std::vector<IdxType> edgelist2;
                            { auto guard = graphref.ag_ptr->edgelist_hashmap.read_lock(nbr); edgelist2 = *guard.edges; }
                            for (const IdxType nbr2 : edgelist2) {
                                        if (visitedSetSize >= filterNbrsToFind) {
                                            break;
                                        }
                                if (visited_set.check(nbr2)) continue;
                                if (predicate(nbr2)) {
                                        ++visitedSetSize;
                                    visited_set.set(nbr2);
                                    float dist = compdist(nbr2);
                                    record_top(nbr2, dist);
                                    search_queue.insert(nbr2, dist);
                                }
                            }
                                        if (visitedSetSize >= filterNbrsToFind) {
                                            break;
                                        }
                        }
                    } else {
                        // directed
                        std::priority_queue<pair<float, IdxType>> nbrsToExplore;    // closest first
                        for (const IdxType nbr : edgelist_gref) {
                            if (visited_set.check(nbr)) continue;
                            visited_set.set(nbr);
                            float dist = compdist(nbr); // pay upfront to compute 1-hop distances
                            nbrsToExplore.emplace(-dist, nbr);
                            if (predicate(nbr)) {
                                record_top(nbr, dist);
                                search_queue.insert(nbr, dist);
                            }
                        }
                        while (!nbrsToExplore.empty()) {
                            auto neighbor = nbrsToExplore.top();
                            nbrsToExplore.pop();    // Note: nbrsToExplore can be out of range
                            if (visitedSetSize >= filterNbrsToFind) {
                                break;
                            }
                            std::vector<IdxType> edgelist2;
                            { auto guard = graphref.ag_ptr->edgelist_hashmap.read_lock(neighbor.second); edgelist2 = *guard.edges; }
                            for (const IdxType nbr2 : edgelist2) {
                                bool pass = predicate(nbr2);
                                if (pass) visitedSetSize++;
                                if (visited_set.check(nbr2)) continue;
                                if (pass) {
                                    visited_set.set(nbr2);
                                    float dist = compdist(nbr2);
                                    record_top(nbr2, dist);
                                    search_queue.insert(nbr2, dist);
                                }
                            }
                        }
                    }
                }
            } // while unexpanded node

            int ptr = top_candidates.size();
            for (int k = ptr; k < K; k++) { results[q*K+k].first = -1; }
            --ptr;
            for (; ptr >= 0; --ptr) {
                auto tmp = top_candidates.top(); top_candidates.pop();
                results[q*K+ptr] = make_pair(tmp.second, tmp.first);
            }
            num_cmps[q] = n_cmps;
            n_hops[q] = n_hops_;

            search_cache_list.release_cache(search_cache);
        }

        auto time_cost = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
        double QPS = nq * 1e6 / time_cost;

        double avg_cmps = (double)accumulate(num_cmps.begin(), num_cmps.end(), 0) / nq;

        double recall = calculate_recall(gt, results, nq, K, false);

        if (true) {
            cerr << endl;
            cerr << "ef_search = " << ef_search << endl;
            cerr << "- Time cost (us): " << time_cost << endl;
            cerr << "- Time per comp (ns): " << 1e6 * time_cost / (avg_cmps * nq) << endl;
            cerr << "- QPS: " << QPS << endl;
            cerr << "- Average number of comparisons: " << avg_cmps << endl;
            cerr << "- Recall: " << recall << endl;
        }

        qpss.push_back(QPS);
        cmpss.push_back(avg_cmps);
        recalls.push_back(recall);
        // if (avg_cmps > 8000) break;
//         if (QPS < 10) break;    // seriously... you guys are taking 2-hop too far...
// #ifdef BREAK_QPS
//         if (QPS < 1000) break;
// #endif
    }

    cout << "[";
    for (double q : cmpss) cout << q << ",";
    cout << "],[";
    for (double q : qpss) cout << q << ",";
    cout << "],[";
    for (double r : recalls) cout << r << ",";
    cout << "]," << endl;

    delete[] results;
}
