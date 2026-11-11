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

int flag0 = 102;
int flag1 = -1;
int flag2 = -1;
int flag3 = -1;
int flag4 = -1;
int flag5 = -1;
int flag6 = -1;
int flag7 = 0;
double flag8 = 1e-2;
int flag9 = -1;

int fanout;

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

        // vector<pair<IdxType, float>> ranking;
        priority_queue<pair<float, IdxType>> ranking;
        for (int i = 0; i < nb; ++i) {
            if (predicate(i, q)) {
                // ranking.emplace_back(i, compdist(i));
                ranking.push(make_pair(compdist(i), i));
                if (ranking.size() > K) { ranking.pop(); }
            }
        }

        std::vector<std::pair<IdxType, float>> sorted;
        sorted.reserve(ranking.size());

        while (!ranking.empty()) {
            auto [dist, id] = ranking.top();
            ranking.pop();
            sorted.emplace_back(id, dist);
        }

        // Sort by increasing distance
        std::sort(sorted.begin(), sorted.end(),
                [](const std::pair<IdxType, float>& a,
                    const std::pair<IdxType, float>& b) {
                    return a.second < b.second;
                });

        for (auto k=0; k<K; ++k) {
            if (k < sorted.size()) {
                gt[q*K+k] = sorted[k];
            } else {
                gt[q*K+k].first = -1;
            }
        }
        if (sorted.size() < K) {
            std::cerr << "Warning: Query " << q << " has only " << sorted.size()
                  << " ground truth entries, less than K=" << K
                  << ". Maximum recall will be less than 100% for this query." << std::endl;
            missed_gt += (K - sorted.size());
            #pragma omp critical
            high_selectivity_ids.emplace_back(sorted.size(), q);
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

        desc.add_options()("flag7", po::value<int>(&flag7)->default_value( 5)); // for tree height cutoff
        desc.add_options()("fanout", po::value<int>(&fanout)->default_value(2),
                           "fanout to use for test");

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

    attr_bin_file = "Synthetic";
    string index_dir_prefix = home + "data/index_files/mar30_mixed_attrs_fanout/";
    if (dataset == "sift3") {
        base_bin_file = home + "data/sift/sift_base.bin";
        query_bin_file = home + "data/sift/sift_query.bin";
        attr_bin_file = pf_path("workloads/mar30_new_mixed_attributes/step0_attributes/SIFT-mixed_attrs.bin");
        query_tsv_file = "./queries.tsv";

        index_path_prefix = index_dir_prefix + "sift-3attr-" + to_string(fanout) + "way/";

        M = 32;
        ef_con = 200;
    } else if (dataset == "redcaps") {
        base_bin_file = home_redcaps + "redcaps-embeddings.bin";
        query_bin_file = home_redcaps + "redcaps-query.bin";
        attr_bin_file = home_redcaps + "redcaps-attributes-subreddit-time-upvote.bin";
        query_tsv_file = "./queries.tsv";

        index_path_prefix = index_dir_prefix + "redcaps-" + to_string(fanout) + "way/";

        M = 64;
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

    {
        // I think after this point there's no need for the vectors inside base_storage
        base_storage->clean();
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
            int maxdepth = 2;
            // int fanout = 4;
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

    AttrTreeNode *rootgraph = nullptr;
    if (indices[0]->schema == irg_attr::NUMERICAL)
        rootgraph = indices[0]->tree->root;
    else
        rootgraph = indices[0]->hash->root_collection;
    for (int i = 1; i < na; i++) {
        if (indices[i]->schema == irg_attr::NUMERICAL)
            indices[i]->tree->root->delegate_root = rootgraph;
        else
            indices[i]->hash->root_collection->delegate_root = rootgraph;
    }

    Aug31Planner density_planner_split(indices);

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

    vector<int> all_attrs;
    for (int i = 0; i < na; i++) all_attrs.push_back(i);

    omp_set_num_threads(num_threads);

    try {
        cout << "\n================================\nWarm-up done.\n================================\n" << endl;

        // work_prefilter(preds);

        // flag13 = 1;
        // work_baseline(preds, all_attrs);
        // flag13 = 0;
        // cerr << "^ baseline prob, random index selection (within the constrained ones)" << endl;

        vector<double> alphas = {0.4};
        for (auto alpha : alphas) {
            density_planner_split.alpha = alpha;
            work_split(density_planner_split, preds);   // Our method (unified index selection)
            cerr << "^ split (alpha = " << alpha << ")" << endl;
        }
    } catch (std::exception const &e) {
        cout << "Caught error!" << endl;
        cout << e.what() << endl;
        cerr << "Caught error!" << endl;
        cerr << e.what() << endl;
        return -1;
    }

    return 0;
}
