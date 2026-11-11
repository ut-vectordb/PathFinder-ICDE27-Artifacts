#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <functional>

#include <boost/program_options.hpp>

#include <omp.h>
#include <filesystem>

#include "run_search.h"
#include "artifact_paths.h"

namespace po = boost::program_options;
using namespace std;
using namespace ANNS;

int flag0 = 0;
int flag1 = 0;
int flag2 = 0;
int flag3 = 0;
int flag4 = 0;
int flag5 = 0;
int flag6 = 0;
int flag7 = 0;
double flag8 = 0;
int flag9 = 0;

std::shared_ptr<ANNS::DistanceHandler> distance_handler;

int nb, nq, dim, na;
ANNS::IdxType M, ef_con;
uint32_t num_threads;

void calc_ground_truth(
    shared_ptr<ANNS::IStorage> base_storage, shared_ptr<ANNS::IStorage> query_storage,
    int K,
    std::function<bool(int, int)> predicate,
    pair<IdxType, float> *gt) {

    int missed_gt = 0;

    vector<pair<int, int>> high_selectivity_ids;

    #pragma omp parallel for schedule (dynamic, 4) reduction(+:missed_gt)
    for (int q = 0; q < nq; q++) {
        auto query = query_storage->get_vector(q);
        auto compdist = [&](IdxType pid) {
            return distance_handler->compute(query, base_storage->get_vector(pid), dim);
        };
        vector<pair<IdxType, float>> ranking;
        for (int i = 0; i < nb; ++i) {
            if (predicate(i, q)) {
                ranking.emplace_back(i, compdist(i));
            }
        }
        sort(ranking.begin(), ranking.end(), [](pair<IdxType, float> a, pair<IdxType, float> b){
            return a.second < b.second;
        });
        for (auto k=0; k<K; ++k) {
            if (k < ranking.size()) {
                gt[q*K+k] = ranking[k];
            } else {
                gt[q*K+k].first = -1;
            }
        }
        if (ranking.size() < K) {
            std::cerr << "Warning: Query " << q << " has only " << ranking.size()
                  << " ground truth entries, less than K=" << K
                  << ". Maximum recall will be less than 100% for this query." << std::endl;
            missed_gt += (K - ranking.size());
            #pragma omp critical
            high_selectivity_ids.emplace_back(ranking.size(), q);
        }
    }

    if (missed_gt > 0) {
        std::cout << "Warning: Total of " << missed_gt << "/" << nq * K << " ground truth entries were missing across all queries." << std::endl;
        double miss_rate = (double)missed_gt / nq / K;
        if (miss_rate > 0.3) {
            std::cout << "Valid results too few (" << miss_rate * 100 << "% missing), refusing to proceed." << std::endl;
            abort();
        }
    }
}

// Helper: escape commas and quotes for CSV
std::string csv_escape(const std::string &s) {
    if (s.find_first_of(",\"\n") != std::string::npos) {
        std::string escaped = "\"";
        for (char c : s) {
            if (c == '"') escaped += "\"\""; // escape quotes
            else escaped += c;
        }
        escaped += "\"";
        return escaped;
    }
    return s;
}

class UseSingleUnitGraphOOR : public IQueryPlanner {
public:
    UseSingleUnitGraphOOR(AttrTreeNode *g_ptr) { g.ag_ptr = g_ptr; g.filter_strategy = unit_graph_ref::NOCHECK; }
    std::vector<unit_graph_ref> operator()(const QueryType &preds) const override { return {g}; }
    unit_graph_ref g;
};

class UseSingleUnitGraphInFilter : public IQueryPlanner {
public:
    UseSingleUnitGraphInFilter(AttrTreeNode *g_ptr) { g.ag_ptr = g_ptr; g.filter_strategy = unit_graph_ref::CHECK_ALL; }
    std::vector<unit_graph_ref> operator()(const QueryType &preds) const override { return {g}; }
    unit_graph_ref g;
};

int main(int argc, char** argv) {

    // common auguments
    std::string data_type("float"), dist_fn("L2"), base_bin_file, query_bin_file, index_path_prefix;
    std::string attr_bin_file;
    std::string query_tsv_file;
    string dataset; // "sift", "gist"

    // parameters for HNSW
    // ANNS::IdxType M;        // "degree bound"
    // ANNS::IdxType ef_con;   // "queue length for construction"
    int n_entry = 15;

    int K = 10;

    try {
        po::options_description desc{"Arguments"};

        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("num_threads", po::value<uint32_t>(&num_threads)->default_value(64),
                           "Number of threads to use");
        desc.add_options()("dataset", po::value<string>(&dataset)->default_value("sift"),
                           "Dataset to use, either 'sift' or 'gist'");
        desc.add_options()("flag0", po::value<int>(&flag0)->default_value(102));
        desc.add_options()("flag1", po::value<int>(&flag1)->default_value( 0)); // for 2-hop
        desc.add_options()("flag2", po::value<int>(&flag2)->default_value(-1)); // for building histogram: releases vector mem
        desc.add_options()("flag3", po::value<int>(&flag3)->default_value(-1)); // for dumping "raw_data" that records edge selection statistics
        desc.add_options()("flag4", po::value<int>(&flag4)->default_value(-1)); // for cutoff
        desc.add_options()("flag5", po::value<int>(&flag5)->default_value(-1)); // for building new graphs: bypasses ground truth
        desc.add_options()("flag6", po::value<int>(&flag6)->default_value(-1)); // for building similarity: bypasses ground truth, releases vector mem
        desc.add_options()("flag7", po::value<int>(&flag7)->default_value( 5)); // for tree height cutoff
        desc.add_options()("flag8", po::value<double>(&flag8)->default_value(1e-2)); // for unit graph density threshold [NOCHECK <> CHECK_ONE]
        desc.add_options()("flag9", po::value<int>(&flag9)->default_value(-1)); // for taking just leaf (1) / non_leaf (2)

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) { std::cout << desc; return 0; }
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        return -1;
    }

    omp_set_num_threads(num_threads);

    string home(pf_path(""));
    string home_redcaps(pf_path("datasets/redcaps/"));
    string home_siftgist(pf_path("datasets/sift-gist-synthetic/"));
    string home_wit(pf_path("datasets/wit/"));

    // int M_ACORN = 32, Mb_ACORN = 64, gamma_ACORN = 12;
    // int M_ACORN = 32, Mb_ACORN = 64, gamma_ACORN = 80;
    int M_ACORN = 32, Mb_ACORN = 128, gamma_ACORN = 80;
    string acorn_index_path(pf_path("data/index_files/acorn/txt/"));

    // string base_fvecs_file;
    attr_bin_file = "Synthetic";
    if (dataset == "redcaps") {
        base_bin_file = home_redcaps + "redcaps-embeddings.bin";
        query_bin_file = home_redcaps + "redcaps-query.bin";
        attr_bin_file = home_redcaps + "redcaps-attributes-subreddit-time-upvote.bin";
        query_tsv_file = "./queries.tsv";

        // index_path_prefix = home + "data/index_files/dec27_HNSWroot_with_NSW/redcaps-" + "4way/";
        index_path_prefix = home + "data/index_files/dec24_fanout_experiment/redcaps-" + "6way/";

        M_ACORN = 64;
        string acorn_postfix = to_string(M_ACORN) + "-" + to_string(Mb_ACORN) + "-" + to_string(gamma_ACORN);
        acorn_index_path += "redcaps-" + acorn_postfix + ".txt";

        M = 64;
        ef_con = 200;
    } else if (dataset == "wit") {
        base_bin_file = pf_path("datasets/wit/embeddings_normalized.bin");
        query_bin_file = pf_path("datasets/wit/embeddings_query_1K_normalized.bin");
        attr_bin_file = pf_path("datasets/wit/df_blob_float32.bin");
        query_tsv_file = "./queries.tsv";

        index_path_prefix = home + "data/index_files/dec27_HNSWroot_with_NSW/WIT-" + "4way/";

        M_ACORN = 64;
        string acorn_postfix = to_string(M_ACORN) + "-" + to_string(Mb_ACORN) + "-" + to_string(gamma_ACORN);
        acorn_index_path += "wit-" + acorn_postfix + ".txt";

        M = 64;
        ef_con = 200;
    } else if (dataset == "sift3") {
        base_bin_file = home + "data/sift/sift_base.bin";
        query_bin_file = home + "data/sift/sift_query.bin";
        attr_bin_file = pf_path("workloads/mar30_new_mixed_attributes/step0_attributes/SIFT-mixed_attrs.bin");
        query_tsv_file = "./queries.tsv";

        index_path_prefix = home + "data/index_files/mar30_mixed_attrs/sift-3attr-" + "4way/";

        string acorn_postfix = to_string(M_ACORN) + "-" + to_string(Mb_ACORN) + "-" + to_string(gamma_ACORN);
        acorn_index_path += "sift-" + acorn_postfix + ".txt";

        M = 32;
        ef_con = 200;
    } else if (dataset == "gist3") {
        base_bin_file = home + "data/gist/gist_base.bin";
        query_bin_file = home + "data/gist/gist_query.bin";
        attr_bin_file = pf_path("workloads/mar30_new_mixed_attributes/step0_attributes/GIST-mixed_attrs.bin");
        query_tsv_file = "./queries.tsv";

        index_path_prefix = home + "data/index_files/mar30_mixed_attrs/gist-3attr-" + "4way/";

        string acorn_postfix = to_string(M_ACORN) + "-" + to_string(Mb_ACORN) + "-" + to_string(gamma_ACORN);
        acorn_index_path += "gist-" + acorn_postfix + ".txt";

        M = 32;
        ef_con = 200;
    } else {
        cerr << "Unknown dataset: " << dataset << endl;
        return -1;
    }
    filesystem::create_directories(index_path_prefix);
    string gt_path_prefix = index_path_prefix + "gt/";
    filesystem::create_directories(gt_path_prefix);

    vector<string> idx_path;
    string gt_path = "./groundtruth.bin";

    vector<int> ef_searches = {10,50,100,200,300,400,500,1000,1500,2000,2500,3000};
    // vector<int> ef_searches = {};
    // vector<int> ef_searches = {10,50,100,200,400,1000,3000};
    // vector<int> ef_searches = {10,50,100};

    vector<int> ef_searches_acorn = {10,50,100,200,300,400,500,1000,1500,2000};
    // vector<int> ef_searches_acorn = {10,50,100,200,400,1000};

    // if (flag3 >= 0) { ef_searches = {100}; }

    std::shared_ptr<ANNS::IStorage> base_storage = ANNS::create_storage(data_type);
    base_storage->load_from_file(base_bin_file, "");
    std::shared_ptr<ANNS::IStorage> query_storage = ANNS::create_storage(data_type);
    query_storage->load_from_file(query_bin_file, "");

    vector<float> attr;
    if (attr_bin_file == "Synthetic") {
        cerr << "Provide a prepared data source!" << endl;
        return -1;
    } else {
        attr = read_binary_float32(attr_bin_file);
    }

    nb = base_storage->get_num_points();
    nq = query_storage->get_num_points();
    dim = base_storage->get_dim();
    na = attr.size() / nb;

    auto [preds, schema] = read_queries_from_tsv(query_tsv_file);
    if (preds.size() > nq)
        preds.resize(nq);   // for GIST, preds is 10K but nq is 1K
    schema.describe();
    my_assert(na == schema.names.size());

    distance_handler = ANNS::get_distance_handler(data_type, dist_fn);

    if (flag2 >= 0 || flag6 >= 0) {
        // release some memory...
        base_storage->clean();
    }

    TableStorage tbstorage;
    TableStorage::load_table_storage_helper(base_storage, attr, tbstorage);
    tbstorage.describe();
    for (int i = 0; i < na; i++) {
        if (schema.types[i] == QueryType::PredType::RANGE) {
            tbstorage.table_schema[i] = TableStorage::ColumnType::NUMERICAL;
        } else if (schema.types[i] == QueryType::PredType::MEMBER) {
            tbstorage.table_schema[i] = TableStorage::ColumnType::CATEGORICAL;
        } else { my_assert(false); }
    }

    auto predicate_direct_access = [&](int pid, int q) -> bool {
        return preds[q].check(attr.data() + pid * na);
    };

    // calculate the ground truth
    // rm -rf /mydata/index_files/jun12_avgdeg-sift/gt

    cout << "\nLoading ground truth for " << nq << " queries...\n" << endl;
    auto gt = new pair<IdxType, float>[nq * K];

    auto truncate_input_query_vecs_and_predicates_and_nq_before_ground_truth = [&](vector<IdxType> &ids_subset) {
        query_storage->reorder_data(ids_subset);

        vector<QueryType> new_preds;
        for (auto i : ids_subset) {
            new_preds.push_back(preds[i]);
        }
        preds = new_preds;
        nq = ids_subset.size();
        cout << "Truncated to " << nq << " queries before groundtruth." << endl;
    };

    if (preds.size() < nq) {
        vector<IdxType> ids_subset;
        for (int i = 0; i < preds.size(); i++) {
            ids_subset.push_back(i);
        }
        truncate_input_query_vecs_and_predicates_and_nq_before_ground_truth(ids_subset);
    }

    if ((flag0 == 102 || (flag0 >= 100 && flag0 < 200)) && !(flag5 != -1) && !(flag6 != -1)) {
        if (load_gt_file(gt_path, gt, nq, K)) {
            ;
        } else {
            omp_set_num_threads(96);
            auto gt_start = std::chrono::high_resolution_clock::now();
            calc_ground_truth(base_storage, query_storage, K, predicate_direct_access, gt);
            auto gt_end = std::chrono::high_resolution_clock::now();
            auto gt_time = std::chrono::duration_cast<std::chrono::milliseconds>(gt_end - gt_start).count();
            double time_s = gt_time / 1e3f;
            double qps = query_storage->get_num_points() / time_s;
            std::cout << "Ground truth calculation took " << gt_time << " ms. That's equivalent to a QPS of " << qps << std::endl;
            write_gt_file(gt_path, gt, nq, K);
        }
    }

    auto truncate_input_query_vecs_and_predicates_and_nq_after_ground_truth = [&](vector<IdxType> &ids_subset) {
        query_storage->reorder_data(ids_subset);

        vector<QueryType> new_preds;
        for (auto i : ids_subset) {
            new_preds.push_back(preds[i]);
        }
        preds = new_preds;

        IdxType counter = 0;
        for (auto i : ids_subset) {
            memmove(counter * K + gt, i * K + gt, K * sizeof(gt[0]));
            ++counter;
        }
        nq = ids_subset.size();
        cout << "Truncated to " << nq << " queries." << endl;
    };

    if (true) {
        vector<IdxType> sample_1K;
        for (int i = 0; i < nq; i++) {
            sample_1K.push_back(i);
        }
        int limit = 1000;
        // int limit = 200;
        if (nq > limit) {
            mt19937 g(42);
            shuffle(sample_1K.begin(), sample_1K.end(), g);
            sample_1K.resize(limit);
        }
        sort(sample_1K.begin(), sample_1K.end());
        truncate_input_query_vecs_and_predicates_and_nq_after_ground_truth(sample_1K);
        // filesystem::create_directories("./raw_data");
        // ofstream fout("./raw_data/selected_query_ids.txt");
        // for (auto i : sample_1K) {
        //     fout << i << "\n";
        // }
        // fout.close();
    }

    string idx_and_estimator_prefix = index_path_prefix + "attr" + "_equidepth";
    vector<shared_ptr<irg_attr>> indices;
    for (int i = 0; i < na; i++) {
        idx_path.push_back(idx_and_estimator_prefix + to_string(i));
        indices.emplace_back();
    }
    omp_set_nested(1);

    #pragma omp parallel for
    for (int i = 0; i < na; i++) {
        if (flag2 != -1 && i != flag2)
            continue;
        if (flag5 != -1 && i != flag5)
            continue;
        cout << "Building graph for attribute #" << i << " using " << num_threads << " threads" << endl;
        if (schema.types[i] == QueryType::PredType::RANGE) {
            // max depth = 9 meaning total of 10 layers
            // int maxdepth = 8;
            int maxdepth = 4;
            int fanout = 4;
            indices[i] = irg_attr::build_or_load_graph_for_numerical_attr(attr, i, maxdepth, base_storage, idx_path[i], distance_handler, M, ef_con, num_threads, fanout);
        } else
            indices[i] = irg_attr::build_or_load_graph_for_hash_attr(attr, i, base_storage, idx_path[i], distance_handler, M, ef_con, num_threads);

        indices[i]->_attr_id = i;
        indices[i]->_attr = &attr;
        indices[i]->_na = na;
    }

    // load root: if we run normally ("-1") or request to build attr #0 ("0")
    if (flag5 == -1 || flag5 == 0) {
        string root_path = idx_and_estimator_prefix + "_root.txt";
        string root_upper_path = idx_and_estimator_prefix + "_root_upper.txt";
        AttrTreeNode *rootnode = (schema.types[0] == QueryType::PredType::RANGE) ? indices[0]->tree->root : indices[0]->hash->root_collection;
        if (filesystem::is_regular_file(root_upper_path) && filesystem::is_regular_file(root_path)) {
            cout << "Loading HNSW root graph" << endl;
            rootnode->g = make_shared<Graph>(nb);
            rootnode->g->load(root_path);
            rootnode->optional_HNSW_upper = make_shared<HNSWUpperLayer>();
            rootnode->optional_HNSW_upper->load(root_upper_path);
            rootnode->vertex_table.clear();
        } else {
            cout << "Building HNSW root graph using " << num_threads << " threads" << endl;
            Hnsw h;
            auto g1 = make_shared<Graph>(nb);
            bool randomize = false; bool merge_upper_layers = false;
            h.build_HNSWLib(base_storage, distance_handler, g1, M, ef_con, num_threads, randomize, merge_upper_layers);
            g1->save(root_path);
            h._upper_layer->save(root_upper_path);

            rootnode->g = g1;
            rootnode->optional_HNSW_upper = h._upper_layer;
        }
    }

    if (flag5 != -1) {
        return 0;
    }

    Aug31Planner density_planner_split(indices);

    if (flag6 >= 0)
        return 0;

    for (int i = 0; i < na; i++) {
        if (flag5 != -1 && i != flag5)
            continue;

        indices[i]->cutoff_too_deep(flag7);

        cout << "Map global ID to edge table and free the graph object..." << endl;
        omp_set_num_threads(96);
        indices[i]->map_global_id_directly_to_edge_table();
        omp_set_num_threads(num_threads);
        cout << "Done mapping" << endl;

        int unit_graph_count = 0;
        int maxdepth = 0;
        size_t total_edge_count = 0;
        if (indices[i]->schema == irg_attr::NUMERICAL) {
            for (auto p : indices[i]->tree->treenodes) {
                if (p) {
                    unit_graph_count++;
                    total_edge_count += p->size_e;
                }
            }
            maxdepth = indices[i]->tree->max_depth;
        } else {
            unit_graph_count = indices[i]->hash->hashednodes.size();
            for (auto p : indices[i]->hash->hashednodes) {
                if (p) {
                    total_edge_count += p->size_e;
                }
            }
            maxdepth = 1;
        }
        cout << "Index #" << i << " has " << unit_graph_count << " unit graphs; layer count (=maxdepth+1) = " << maxdepth + 1 << "; total_edges = " << total_edge_count << endl;
    }

    // ANNS::PrintTree(indices[0]->hash->root_collection);
    // ANNS::PrintTree(indices[2]->tree->root);

    if (flag9 >= 0) {
        // to take a subset of all queries: take query_storage, preds, gt
        vector<IdxType> ids_subset; // new_to_old
        {
            #pragma omp parallel for
            for (int i = 0; i < preds.size(); i++) {
                auto plan = density_planner_split(preds[i]);
                #pragma omp critical
                if (plan.size() == 1) {
                    ids_subset.push_back(i);
                }
            }
            sort(ids_subset.begin(), ids_subset.end());

            if (flag9 == 2) {   // non leaf only, take the other subset
                vector<IdxType> the_other_subset;
                for (int i = 0; i < nq; i++) {
                    bool found = false;
                    for (auto j : ids_subset) {
                        if (i == j) { found = true; break; }
                    }
                    if (!found) {
                        the_other_subset.push_back(i);
                        cerr << "got " << i << endl;
                    }
                }
                ids_subset = the_other_subset;
            }
        }

        query_storage->reorder_data(ids_subset);
        vector<QueryType> new_preds;
        for (auto i : ids_subset) {
            new_preds.push_back(preds[i]);
        }
        preds = new_preds;
        IdxType counter = 0;
        for (auto i : ids_subset) {
            memmove(counter * K + gt, i * K + gt, K * sizeof(gt[0]));
            ++counter;
        }
        nq = ids_subset.size();
    }

    if (flag0 == 111) {
        // report the time overhead of query planning for this workload
        omp_set_num_threads(num_threads);
        cout << "Testing query planning overhead using " << num_threads << " threads..." << endl;
        auto t0 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (int i = 0; i < nq; i++) {
            auto plan = density_planner_split(preds[i]);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        auto time_s = std::chrono::duration<float>(t1 - t0).count();
        cout << "Query planning for " << nq << " queries took " << time_s << " seconds." << endl;
        return 0;
    }

#ifdef PRINT_CARDEST
    // test the time consumption of query planner and invocation
    if (1) {
        vector<int> test_q_id = {4};
        // for (int i = 0; i < 10; i++) test_q_id.push_back(i);

        std::cout << std::left
          << std::setw(8)  << "q_id"
          << std::setw(16) << "time(µs)"
          << std::setw(15) << "#bins"
          << std::setw(15) << "#evokes"
          << std::setw(16) << "µs/bin"
          << std::setw(16) << "µs/evoke"
          << "\n";

        for (int q : test_q_id) {
            density_planner_split.clean_estimator_stats();
            int counter_bin, counter_evoke;
            density_planner_split.get_estimator_stats(counter_bin, counter_evoke);
            my_assert(counter_bin == 0);
            my_assert(counter_evoke == 0);

            auto t0 = std::chrono::high_resolution_clock::now();
            auto plan = density_planner_split(preds[q]);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            double time_us = time_ns / 1000.0;

            density_planner_split.get_estimator_stats(counter_bin, counter_evoke);
            double us_per_bin   = counter_bin   > 0 ? time_us / counter_bin   : std::numeric_limits<double>::quiet_NaN();
            double us_per_evoke = counter_evoke > 0 ? time_us / counter_evoke : std::numeric_limits<double>::quiet_NaN();
            std::cout << std::left
              << std::setw(8)  << q
              << std::setw(15) << time_us
              << std::setw(15) << counter_bin
              << std::setw(15) << counter_evoke
              << std::setw(15) << us_per_bin
              << std::setw(15) << us_per_evoke
              << "\n";
        }

        return 0;
    }
#endif

    cout << "histogram done" << endl;

    AttrTreeNode *rootgraph = nullptr;
    for (int i = 0; i < na; i++) {
        if (indices[i]->schema == irg_attr::NUMERICAL)
            rootgraph = indices[i]->tree->root;
        else
            rootgraph = indices[i]->hash->root_collection;
        break;
    }
    for (int i = 1; i < na; i++) {
        if (indices[i]->schema == irg_attr::NUMERICAL)
            indices[i]->tree->root->delegate_root = rootgraph;
        else
            indices[i]->hash->root_collection->delegate_root = rootgraph;
    }

    unit_graph_ref rootgraph_ref; { rootgraph_ref.ag_ptr = rootgraph; }

    auto work_baseline = [&](const vector<QueryType> &preds, vector<int> irg_id) {
        test_search_IRG_collection(
            tbstorage, query_storage,
            irg_id, indices,
            attr,
            preds, K, ef_searches, n_entry, gt,
            rootgraph
        );
    };

    auto work_split = [&](
            IQueryPlanner &planner,
            const vector<QueryType> &preds) {
        test_search_using_plan_separate_post(
            tbstorage, query_storage,
            planner,
            attr,
            preds, K, ef_searches, n_entry, gt
        );
    };

    auto work_prefilter = [&](
            const vector<QueryType> &preds) {
        auto gt = new pair<IdxType, float>[nq * K];
        auto gt_start = std::chrono::high_resolution_clock::now();
        calc_ground_truth(base_storage, query_storage, K, predicate_direct_access, gt);
        auto gt_end = std::chrono::high_resolution_clock::now();
        auto gt_time = std::chrono::duration_cast<std::chrono::milliseconds>(gt_end - gt_start).count();
        double time_s = gt_time / 1e3f;
        double qps = query_storage->get_num_points() / time_s;
        cout << "[" << qps << ",],[" << 100.00 << ",]," << endl;
    };

    vector<int> all_attrs;
    for (int i = 0; i < na; i++) all_attrs.push_back(i);

    omp_set_num_threads(num_threads);

    try {
        // work_split(density_planner_split, preds);  /////////////////
        // cerr << "^ split" << endl;

        cout << "\n================================\nWarm-up done.\n================================\n" << endl;

        UseSingleUnitGraphInFilter infiltering_on_vamanagraph(rootgraph);
        UseSingleUnitGraphOOR simple_out_of_range_search(rootgraph);

        if (flag0 == 108) {
            // SOMEINDEX, but iterates over all combinations
            my_assert(dataset == "redcaps");
            density_planner_split.alpha = 0.4;

            int n = 3;  // number of elements
            std::vector<int> items = {0, 1, 2};

            // loop over all non-empty subsets
            for (int mask = 1; mask < (1 << n); ++mask) {
                std::vector<int> index_ids;
                for (int i = 0; i < n; ++i) {
                    if (mask & (1 << i)) {
                        index_ids.push_back(items[i]);
                        // cout << items[i] << ",";
                    }
                }
                // cout << endl;

                density_planner_split.set_index_mask(index_ids);
                work_split(density_planner_split, preds);

                flag13 = 1;
                work_baseline(preds, index_ids);
                flag13 = 0;
            }

            return 0;
        }

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

        if (flag0 == 109) {
            // SOMEINDEX, but iterates over all combinations
            my_assert(dataset == "redcaps");
            density_planner_split.alpha = 0.4;

            int n = 3;  // number of elements
            std::vector<int> items = {0, 1, 2};

            // loop over all non-empty subsets
            for (int mask = 1; mask < (1 << n); ++mask) {
                std::vector<int> index_ids;
                cout << "Enabled indexes: ";
                for (int i = 0; i < n; ++i) {
                    if (mask & (1 << i)) {
                        index_ids.push_back(items[i]);
                        cout << items[i] << ",";
                    }
                }
                cout << "\n";
                density_planner_split.set_index_mask(index_ids);
                for (int j = 0; j < 5; j++) {
                    auto plan = density_planner_split(preds[j]);
                    cout << "plan #" << j << ": " << planstring(plan) << endl;
                }
                cout << "\n";
            }

            return 0;
        }

        vector<double> alphas = {0, 0.2, 0.4, 0.6, 0.8, 1.0};
        // vector<double> alphas = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};  // 11
        // vector<double> alphas = {0, 0.2, 0.4, 0.6, 0.8, 1.0, 2, 3, 5, 10};
        for (auto alpha : alphas) {
            // density_planner_split.alpha = alpha;
            // density_planner_split.alpha_categorical = alpha;
            density_planner_split.alpha = 1.0;
            density_planner_split.alpha_categorical = alpha;
            work_split(density_planner_split, preds);   // Our method (unified index selection)
            cerr << "^ split (alpha = " << alpha << ")" << endl;
        }

        // work_split(simple_out_of_range_search, preds);
    } catch (std::exception const &e) {
        cout << "Caught error!" << endl;
        cout << e.what() << endl;
        cerr << "Caught error!" << endl;
        cerr << e.what() << endl;
        return -1;
    }

    return 0;
}
