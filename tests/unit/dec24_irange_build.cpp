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

int fanout = 2;

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

int main(int argc, char** argv) {

    // common auguments
    std::string data_type("float"), dist_fn("L2"), base_bin_file, query_bin_file, index_path_prefix;
    std::string attr_bin_file;
    std::string query_tsv_file;
    string dataset; // "sift", "gist"

    // parameters for HNSW
    int n_entry = 15;

    int K = 10;
    int attr_id = 0;

    try {
        po::options_description desc{"Arguments"};

        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("num_threads", po::value<uint32_t>(&num_threads)->default_value(64),
                           "Number of threads to use");
        desc.add_options()("dataset", po::value<string>(&dataset)->default_value("sift"),
                           "Dataset to use, either 'sift' or 'gist'");
        desc.add_options()("attr_id", po::value<int>(&attr_id)->default_value(0),
                           "Which attribute to build the irangegraph");

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
    string index_dir_prefix = home + "data/index_files/mar30_mixed_attrs_irange/";
    if (dataset == "sift3") {
        base_bin_file = home + "data/sift/sift_base.bin";
        query_bin_file = home + "data/sift/sift_query.bin";
        attr_bin_file = pf_path("workloads/mar30_new_mixed_attributes/step0_attributes/SIFT-mixed_attrs.bin");
        query_tsv_file = "./queries.tsv";

        index_path_prefix = index_dir_prefix + "sift-3attr/";

        M = 32;
        ef_con = 200;
    } else if (dataset == "arxiv") {
        if (attr_id == 1) {
            cerr << "This is not a Numeric attribute!" << endl;
            abort();
        }

        base_bin_file = home + "data/arxiv/arxiv-embeddings.bin";
        query_bin_file = home + "data/arxiv/arxiv-query.bin";
        attr_bin_file = home + "data/arxiv/arxiv-attributes-time-class-cite.bin";
        query_tsv_file = "./queries.tsv";

        index_path_prefix = index_dir_prefix + "arxiv/";

        M = 64;
        ef_con = 200;
    } else if (dataset == "redcaps") {
        if (attr_id == 0) {
            cerr << "This is not a Numeric attribute!" << endl;
            abort();
        }

        base_bin_file = home_redcaps + "redcaps-embeddings.bin";
        query_bin_file = home_redcaps + "redcaps-query.bin";
        attr_bin_file = home_redcaps + "redcaps-attributes-subreddit-time-upvote.bin";
        query_tsv_file = "./queries.tsv";

        index_path_prefix = index_dir_prefix + "redcaps/";

        M = 64;
        ef_con = 200;
    } else {
        cerr << "Unknown dataset: " << dataset << endl;
        return -1;
    }
    filesystem::create_directories(index_path_prefix);
    string gt_path_prefix = index_path_prefix + "gt/";
    filesystem::create_directories(gt_path_prefix);

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

    distance_handler = ANNS::get_distance_handler(data_type, dist_fn);

    string idx_and_estimator_prefix = index_path_prefix + "irangegraph_";
    shared_ptr<irg_attr> index;
    string idx_path = idx_and_estimator_prefix + to_string(attr_id);
    omp_set_nested(1);

    cout << "Building graph for attribute #" << attr_id << " using " << num_threads << " threads" << endl;
    int maxdepth = 1000;
    auto link_start = std::chrono::high_resolution_clock::now();
    index = irg_attr::build_or_load_graph_for_numerical_attr(attr, attr_id, maxdepth, base_storage, idx_path, distance_handler, M, ef_con, num_threads, fanout);
    auto link_end = std::chrono::high_resolution_clock::now();
    auto link_time = std::chrono::duration_cast<std::chrono::milliseconds>(link_end - link_start).count();
    std::cout << "@@@ Index #" << attr_id << " (everything besides root graph) build time = " << link_time / 1000.00 << " seconds." << std::endl;
    size_t edge_count = 0;
    int n_graphs = 0;
    for (auto g : index->tree->treenodes) {
        if (g && g->g) {
            ++n_graphs;
            size_t byte_count = g->g->get_index_size();
            edge_count += byte_count / 4;
        }
    }
    std::cout << "@@@ Index #" << attr_id << " (everything besides root graph) unit graph count = " << n_graphs << std::endl;
    std::cout << "@@@ Index #" << attr_id << " (everything besides root graph) edge count = " << edge_count << std::endl;

    index->_attr_id = attr_id;
    index->_attr = &attr;
    index->_na = na;

    {
        string root_path = idx_and_estimator_prefix + "_root.txt";
        AttrTreeNode *rootnode = index->tree->root;
        if (filesystem::is_regular_file(root_path)) {
            cout << "Loading root graph" << endl;
            rootnode->g = make_shared<Graph>(nb);
            rootnode->g->load(root_path);
            rootnode->vertex_table.clear();
        } else {
            cout << "Building root graph using " << num_threads << " threads" << endl;
            Hnsw h(false);
            auto g1 = make_shared<Graph>(nb);
            auto link_start = std::chrono::high_resolution_clock::now();
            h.build_NSW(base_storage, distance_handler, g1, M, ef_con, num_threads);
            auto link_end = std::chrono::high_resolution_clock::now();
            auto link_time = std::chrono::duration_cast<std::chrono::milliseconds>(link_end - link_start).count();
            std::cout << "@@@ NSW Root build time = " << link_time / 1000.00 << " seconds." << std::endl;
            g1->save(root_path);
            rootnode->g = g1;

            size_t total_byte_count = g1->get_index_size();
            cout << "@@@ Edge count (Root) = " << total_byte_count / 4 << endl;
        }
    }

    return 0;
}
