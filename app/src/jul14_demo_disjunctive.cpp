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
#include "icde_search.h"
#include "prefilter_scan.h"
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

int calc_ground_truth(
    shared_ptr<ANNS::IStorage> base_storage, shared_ptr<ANNS::IStorage> query_storage,
    int K,
    std::function<bool(int, int)> predicate,
    pair<IdxType, float> *gt) {

    int missed_gt = 0;
    int n_cmps = 0;

    vector<pair<int, int>> high_selectivity_ids;

    #pragma omp parallel for schedule (dynamic, 4) reduction(+:missed_gt) reduction(+:n_cmps)
    for (int q = 0; q < nq; q++) {
        auto query = query_storage->get_vector(q);
        auto compdist = [&](IdxType pid) {
            ++n_cmps;
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

    return n_cmps;
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
    int fanout = -1;

    int K = 10;
    IcdeSearchOptions icde_options;
    string base_override, query_override, attr_override, index_override, acorn_override;

    try {
        po::options_description desc{"Arguments"};

        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("num_threads", po::value<uint32_t>(&num_threads)->default_value(64),
                           "Number of threads to use");
        desc.add_options()("dataset", po::value<string>(&dataset)->default_value("sift"),
                           "Dataset to use, either 'sift' or 'gist'");
        desc.add_options()("flag0", po::value<int>(&flag0)->default_value(102));
        desc.add_options()("flag1", po::value<int>(&flag1)->default_value( 0)); // for 2-hop
        desc.add_options()("flag2", po::value<int>(&flag2)->default_value(-1)); // repurpose: when=1, load just PathFinder; when=2, evaluate Vamana index and Prefiltering
        desc.add_options()("flag3", po::value<int>(&flag3)->default_value(-1)); // for dumping "raw_data" that records edge selection statistics
        desc.add_options()("flag4", po::value<int>(&flag4)->default_value(-1)); // for cutoff
        desc.add_options()("flag5", po::value<int>(&flag5)->default_value(-1)); // for building new graphs: bypasses ground truth
        desc.add_options()("flag6", po::value<int>(&flag6)->default_value(-1)); // for building similarity: bypasses ground truth, releases vector mem
        desc.add_options()("flag7", po::value<int>(&flag7)->default_value( 5)); // for tree height cutoff
        desc.add_options()("flag8", po::value<double>(&flag8)->default_value(1e-2)); // for unit graph density threshold [NOCHECK <> CHECK_ONE]
        desc.add_options()("flag9", po::value<int>(&flag9)->default_value(-1)); // for taking just leaf (1) / non_leaf (2)
        desc.add_options()("fanout", po::value<int>(&fanout)->default_value(-1), "fanout to use for test");
        desc.add_options()
            ("base_file", po::value<string>(&base_override), "Override base vector file")
            ("query_file", po::value<string>(&query_override), "Override query vector file")
            ("attribute_file", po::value<string>(&attr_override), "Override attribute file")
            ("index_dir", po::value<string>(&index_override), "Override dataset index directory")
            ("acorn_index_file", po::value<string>(&acorn_override), "Override ACORN index file")
            ("query_tsv", po::value<string>(&query_tsv_file), "Override disjunctive query TSV (default ./queries_disjunctive.tsv)");
        desc.add_options()("icde_baseline", po::value<string>(&icde_options.algorithm)->default_value(""), "Search-only port: favor, cgif, pathseer; offline favor-calibrate");
        desc.add_options()("favor_delta_d", po::value<float>(&icde_options.favor_delta_d), "Required offline FAVOR calibration statistic");
        desc.add_options()("pathseer_graph", po::value<string>(&icde_options.pathseer_graph), "PathSeer graph-only search export");
        desc.add_options()("cgif_clusters", po::value<string>(&icde_options.cgif_clusters), "CGIF clustering-only search export");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) { std::cout << desc; return 0; }
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        return -1;
    }

    if (!icde_options.algorithm.empty()) {
        // comma-separated list allowed; entries validated in test_search_icde
        auto has = [&](const string& a) { return ("," + icde_options.algorithm + ",").find("," + a + ",") != string::npos; };
        if (flag0 != 102 || flag5 != -1 || flag6 != -1 || flag9 != -1 || num_threads == 0)
            throw runtime_error("ICDE ports require flag0=102, flag5/6/9=-1 and positive num_threads");
        if (has("favor") && (!std::isfinite(icde_options.favor_delta_d) || icde_options.favor_delta_d < 0))
            throw runtime_error("FAVOR calibration pending: supply --favor_delta_d");
        if (has("pathseer") && icde_options.pathseer_graph.empty())
            throw runtime_error("supply --pathseer_graph");
        if (has("cgif") && icde_options.cgif_clusters.empty())
            throw runtime_error("supply --cgif_clusters");
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

    attr_bin_file = "Synthetic";
    if (query_tsv_file.empty()) query_tsv_file = "./queries_disjunctive.tsv";

    if (dataset == "sift100M") {
        base_bin_file = pf_path("datasets/sift100m/sift100m_base.bin");
        query_bin_file = pf_path("datasets/sift100m/sift100m_query.bin");
        attr_bin_file = pf_path("workloads/mar30_new_mixed_attributes/step0_attributes/SIFT100M-mixed_attrs.bin");

        index_path_prefix = home + "data/index_files/mar30_mixed_attrs/sift-100M-" + "6way/";

        acorn_index_path = "";

        M = 32;
        ef_con = 200;
        data_type = "uint8";
    } else if (dataset == "sift3") {
        base_bin_file = home + "data/sift/sift_base.bin";
        query_bin_file = home + "data/sift/sift_query.bin";
        attr_bin_file = pf_path("workloads/mar30_new_mixed_attributes/step0_attributes/SIFT-mixed_attrs.bin");

        if (fanout == -1) {
            index_path_prefix = home + "data/index_files/mar30_mixed_attrs/sift-3attr-" + "6way/";
        } else {
            index_path_prefix = home + "data/index_files/mar30_mixed_attrs_fanout/sift-3attr-" + to_string(fanout) + "way/";
        }

        string acorn_postfix = to_string(M_ACORN) + "-" + to_string(Mb_ACORN) + "-" + to_string(gamma_ACORN);
        acorn_index_path += "sift-" + acorn_postfix + ".txt";

        M = 32;
        ef_con = 200;
    } else if (dataset == "gist3") {
        base_bin_file = home + "data/gist/gist_base.bin";
        query_bin_file = home + "data/gist/gist_query.bin";
        attr_bin_file = pf_path("workloads/mar30_new_mixed_attributes/step0_attributes/GIST-mixed_attrs.bin");

        index_path_prefix = home + "data/index_files/mar30_mixed_attrs/gist-3attr-" + "6way/";

        M_ACORN = 64;
        string acorn_postfix = to_string(M_ACORN) + "-" + to_string(Mb_ACORN) + "-" + to_string(gamma_ACORN);
        acorn_index_path += "gist-" + acorn_postfix + ".txt";

        M = 64;
        ef_con = 200;
    } else if (dataset == "arxiv") {
        base_bin_file = home + "data/arxiv/arxiv-embeddings.bin";
        query_bin_file = home + "data/arxiv/arxiv-query.bin";
        attr_bin_file = home + "data/arxiv/arxiv-attributes-time-class-cite.bin";

        index_path_prefix = home + "data/index_files/dec27_HNSWroot_with_NSW/arxiv-" + "6way/";

        M_ACORN = 64;
        string acorn_postfix = to_string(M_ACORN) + "-" + to_string(Mb_ACORN) + "-" + to_string(gamma_ACORN);
        acorn_index_path += "arxiv-" + acorn_postfix + ".txt";

        M = 64;
        ef_con = 200;
    } else if (dataset == "redcaps") {
        base_bin_file = home_redcaps + "redcaps-embeddings.bin";
        query_bin_file = home_redcaps + "redcaps-query.bin";
        attr_bin_file = home_redcaps + "redcaps-attributes-subreddit-time-upvote.bin";

        index_path_prefix = home + "data/index_files/dec27_HNSWroot_with_NSW/redcaps-" + "6way/";

        M_ACORN = 64;
        string acorn_postfix = to_string(M_ACORN) + "-" + to_string(Mb_ACORN) + "-" + to_string(gamma_ACORN);
        acorn_index_path += "redcaps-" + acorn_postfix + ".txt";

        M = 64;
        ef_con = 200;
    } else if (dataset == "wit") {
        base_bin_file = pf_path("datasets/wit/embeddings_normalized.bin");
        query_bin_file = pf_path("datasets/wit/embeddings_query_1K_normalized.bin");
        attr_bin_file = pf_path("datasets/wit/df_blob_float32.bin");

        index_path_prefix = home + "data/index_files/dec27_HNSWroot_with_NSW/WIT-" + "6way/";

        M_ACORN = 64;
        string acorn_postfix = to_string(M_ACORN) + "-" + to_string(Mb_ACORN) + "-" + to_string(gamma_ACORN);
        acorn_index_path += "wit-" + acorn_postfix + ".txt";

        M = 64;
        ef_con = 200;
    } else {
        cerr << "Unknown dataset: " << dataset << endl;
        return -1;
    }
    if (!base_override.empty()) base_bin_file = base_override;
    if (!query_override.empty()) query_bin_file = query_override;
    if (!attr_override.empty()) attr_bin_file = attr_override;
    if (!index_override.empty()) index_path_prefix = index_override + "/";
    if (!acorn_override.empty()) acorn_index_path = acorn_override;
    filesystem::create_directories(index_path_prefix);
    string gt_path_prefix = index_path_prefix + "gt/";
    filesystem::create_directories(gt_path_prefix);

    vector<string> idx_path;
    string gt_path = "./groundtruth.bin";

#ifdef SOMEINDEX
    // vector<int> ef_searches = {10,50,100,200,300,400,500,1000,1500,2000,2500,3000,4000,5000,6000,7000,10000};
    vector<int> ef_searches = {10,50,100,200,300,400,500,1000,1500,2000,2500,3000};
#else
    vector<int> ef_searches = {10,50,100,200,300,400,500,1000,1500,2000,2500,3000};
#endif
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

    auto [preds, schema] = read_disjunctive_queries_from_tsv(query_tsv_file, na);
    if (preds.size() > nq) {
        preds.resize(nq);   // for GIST, preds is 10K but nq is 1K
    }
    schema.describe();
    my_assert(na == schema.names.size());

    distance_handler = ANNS::get_distance_handler(data_type, dist_fn);

    AttrTreeNode *vamanagraph = new AttrTreeNode(0, 100, 0, nb);
    if (icde_options.algorithm.empty() && (flag2 == -1 || flag2 == 2)) {
        string vamana_path = index_path_prefix + "vamana.txt";
        if (filesystem::is_regular_file(vamana_path)) {
            cout << "Loading Vamana Graph at " << vamana_path << "..." << endl;
            vamanagraph->g = make_shared<Graph>(nb);
            vamanagraph->g->load(vamana_path);
            cout << "Load done" << endl;
        } else {
            cout << "Building Vamana Graph using " << num_threads << " threads" << endl;
            Hnsw h(false);
            h.fixed_alpha = 1.2;
            auto g1 = make_shared<Graph>(nb);
            h.build_NSW(base_storage, distance_handler, g1, M, ef_con, num_threads);
            g1->save(vamana_path);

            vamanagraph->g = g1;
        }

        cout << "vamanagraph->map_global_id_directly_to_edge_table();" << endl;
        vamanagraph->map_global_id_directly_to_edge_table();
        cout << "done" << endl;
    }

    if (flag6 >= 0) {
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

        vector<DisjunctiveQueryType> new_preds;
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

    if (!(flag5 != -1) && !(flag6 != -1)) {
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

    // if (flag2 == 1) {
    //     // I think after this point there's no need for the vectors inside base_storage
    //     // NOTE: This explodes when:
    //     //  1. trying to build an index...
    //     //  2. running "prefilter"
    //     //  3. when tbstorage b tree is disabled
    //     cout << "base_storage->clean();" << endl;
    //     base_storage->clean();
    //     cout << "done" << endl;
    // }

    auto truncate_input_query_vecs_and_predicates_and_nq_after_ground_truth = [&](vector<IdxType> &ids_subset) {
        query_storage->reorder_data(ids_subset);

        vector<DisjunctiveQueryType> new_preds;
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
        if (nq > 1000) {
            mt19937 g(42);
            shuffle(sample_1K.begin(), sample_1K.end(), g);
            sample_1K.resize(1000);
        }
        sort(sample_1K.begin(), sample_1K.end());
        truncate_input_query_vecs_and_predicates_and_nq_after_ground_truth(sample_1K);
        filesystem::create_directories("./raw_data");
        ofstream fout("./raw_data/selected_query_ids.txt");
        for (auto i : sample_1K) {
            fout << i << "\n";
        }
        fout.close();
    }

    if (!icde_options.algorithm.empty()) {
        test_search_icde(tbstorage, query_storage, preds, index_path_prefix, K, ef_searches, gt, icde_options);
        delete[] gt;
        return 0;
    }

    shared_ptr<Graph> acorn_graph(nullptr);

    if (acorn_index_path.size() > 0 && !filesystem::is_regular_file(acorn_index_path)) {

        cout << "ACORN graph not found at " << acorn_index_path << "; skipping ACORN." << endl;

        acorn_index_path = "";

    }
    if (flag0 != 108 && acorn_index_path.size() > 0) {
        cout << "Loading ACORN graph..." << endl;
        acorn_graph = make_shared<Graph>(nb);
        acorn_graph->load(acorn_index_path);
        cout << "ACORN graph loaded." << endl;
        size_t n_edges = acorn_graph->get_index_size() / 4;
        cout << "ACORN graph size: " << n_edges << " edges" << endl;
        cout << "Average degree: " << (double)n_edges / nb << endl;
    }

    string idx_and_estimator_prefix = index_path_prefix + "attr" + "_equidepth";
    vector<shared_ptr<irg_attr>> indices;
    for (int i = 0; i < na; i++) {
        idx_path.push_back(idx_and_estimator_prefix + to_string(i));
        indices.emplace_back();
    }
    omp_set_nested(1);

if (flag2 != 2) {   // flag2 == 1 means we load PathFinder; flag2 == 2 means we only load Vamana graph
    #pragma omp parallel for
    for (int i = 0; i < na; i++) {
        if (flag5 != -1 && i != flag5)
            continue;
        cout << "Building graph for attribute #" << i << " using " << num_threads << " threads" << endl;
        if (schema.types[i] == QueryType::PredType::RANGE) {
            // max depth = 9 meaning total of 10 layers
            int maxdepth = 8;
            indices[i] = irg_attr::build_or_load_graph_for_numerical_attr(attr, i, maxdepth, base_storage, idx_path[i], distance_handler, M, ef_con, num_threads);
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
            rootnode->g = make_shared<Graph>(nb);
            rootnode->g->load(root_path);
            rootnode->optional_HNSW_upper = make_shared<HNSWUpperLayer>();
            rootnode->optional_HNSW_upper->load(root_upper_path);
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

        rootnode->vertex_table.clear();
    }
}

    if (flag5 != -1) {
        return 0;
    }

    AttrTreeNode *rootgraph = nullptr;
if (flag2 != 2) {   // flag2 == 1 means we load PathFinder; flag2 == 2 means we only load Vamana graph
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
}

    if (flag6 >= 0)
        return 0;

    for (int i = 0; i < na; i++) {
        if (flag5 != -1 && i != flag5)
            continue;
        if (indices[i] == nullptr)
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

    auto work_baseline = [&](const vector<DisjunctiveQueryType> &preds, vector<int> irg_id) {
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
            const vector<DisjunctiveQueryType> &preds) {
        test_search_using_plan_separate_post(
            tbstorage, query_storage,
            planner,
            attr,
            preds, K, ef_searches, n_entry, gt
        );
    };

    auto work_acorn = [&](
            const vector<DisjunctiveQueryType> &preds) {
        test_search_acorn(
            tbstorage, query_storage,
            acorn_graph, M_ACORN, Mb_ACORN, gamma_ACORN,
            attr,
            preds, K, ef_searches_acorn, n_entry, gt
        );
    };

    auto work_navix = [&](
            const vector<DisjunctiveQueryType> &preds) {

        auto predicate = [&](int q, int pid) {
            const float *attr_ptr = tbstorage.get_attr(tbstorage.lookup(pid));
            return preds[q].check(attr_ptr);
        };

        // int n_entry = 5;    // so as to match the in-filtering configuration

        // test_search_navix_global(tbstorage, query_storage, rootgraph_ref, M, attr, predicate, K, ef_searches, n_entry, gt, 0);   // global-adaptive
        test_search_navix(tbstorage, query_storage, rootgraph_ref, M, attr, predicate, K, ef_searches, n_entry, gt, 0);   // local-adaptive
    };

    auto work_prefilter = [&](
            const vector<DisjunctiveQueryType> &preds) {
        vector<int> query_ids(preds.size());
        iota(query_ids.begin(), query_ids.end(), 0);
        auto result = prefilter_scan(tbstorage, *query_storage, preds, query_ids, *distance_handler, K);
        double recall = calculate_recall(gt, result.hits.data(), preds.size(), K, false);
        cout << "[" << double(result.comparisons) / preds.size() << ",],["
             << preds.size() / result.seconds << ",],[" << recall << ",]," << endl;
    };

    vector<int> all_attrs;
    for (int i = 0; i < na; i++) all_attrs.push_back(i);

    omp_set_num_threads(num_threads);

    try {
        // work_split(density_planner_split, preds);  /////////////////
        // cerr << "^ split" << endl;

        cout << "\n================================\nWarm-up done.\n================================\n" << endl;

        UseSingleUnitGraphOOR OOR_HNSW(rootgraph);
        UseSingleUnitGraphOOR OOR_Vamana(vamanagraph);

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

        if (flag2 != -1) {
            // revert the ef_search list
            reverse(ef_searches.begin(), ef_searches.end());
        }
        if (flag2 == 1) {
            // just PathFinder
            work_split(OOR_HNSW, preds);
            work_navix(preds); cerr << "^ NaviX" << endl;

            Aug31Planner density_planner_split(indices);
            density_planner_split.populate_flattened_lists();
            density_planner_split.alpha = 1.0;
            density_planner_split.alpha_categorical = (dataset == "redcaps" || dataset == "arxiv") ? 0.4 : 1.0;
            work_split(density_planner_split, preds);   // Our method (unified index selection)
            return 0;
        } else if (flag2 == 2) {
            // just Vamana and prefilter
            work_split(OOR_Vamana, preds);
            work_prefilter(preds);
            return 0;
        }

        work_split(OOR_HNSW, preds);
        work_split(OOR_Vamana, preds);
        if (acorn_graph != nullptr) { work_acorn(preds); cerr << "^ ACORN" << endl; }
        work_navix(preds); cerr << "^ NaviX" << endl;
        work_prefilter(preds);

        Aug31Planner density_planner_split(indices);
        density_planner_split.populate_flattened_lists();
        density_planner_split.alpha = 1.0;
        density_planner_split.alpha_categorical = (dataset == "redcaps" || dataset == "arxiv") ? 0.4 : 1.0;
        work_split(density_planner_split, preds);   // Our method (unified index selection)
    } catch (std::exception const &e) {
        cout << "Caught error!" << endl;
        cout << e.what() << endl;
        cerr << "Caught error!" << endl;
        cerr << e.what() << endl;
        return -1;
    }

    return 0;
}
