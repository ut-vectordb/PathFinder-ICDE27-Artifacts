#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <functional>
#include <map>
#include <unordered_set>

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
int flag5 = -1;
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

vector<float> vals;

vector<QueryType> generate_queries(const vector<float> &attr) {
    my_assert(attr.size() == (size_t)nb * na);

    for (IdxType i = 0; i < nb; i++) { vals.push_back(attr[i * na + 1]); }
    sort(vals.begin(), vals.end());

    std::mt19937 rng{42};

    vector<QueryType> ret;
    float inf = std::numeric_limits<float>::infinity();

    for (int i = 0; i < nq; i++) {
        int width;
        // std::uniform_int_distribution kind(0, 2);
        // int selectivity_class = kind(rng);
        // if (selectivity_class == 0) {
        //     width = 0.2 * nb;
        // } else if (selectivity_class == 1) {
        //     width = 0.02 * nb;
        // } else if (selectivity_class == 2) {
        //     width = 0.002 * nb;
        // }
        width = nb * 0.01;
        std::uniform_int_distribution dist(0, nb - width - 1);
        int ql, qr; ql = dist(rng); qr = ql + width;

        QueryType q(na);
        q.set_range(0, -inf, inf);
        q.set_range(1, vals[ql], vals[qr]);
        q.set_range(2, -inf, inf);
        ret.push_back(q);
    }

    return ret;
}

class UseSingleUnitGraph : public IQueryPlanner {
public:
    UseSingleUnitGraph(AttrTreeNode *g_ptr) { g.ag_ptr = g_ptr; g.filter_strategy = unit_graph_ref::NOCHECK; }
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

    double desired_correlation = 0.1;

    int K = 10;

    try {
        po::options_description desc{"Arguments"};

        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("num_threads", po::value<uint32_t>(&num_threads)->default_value(16),
                           "Number of threads to use");
        desc.add_options()("dataset", po::value<string>(&dataset)->default_value("sift"),
                           "Dataset to use, either 'sift' or 'gist'");
        desc.add_options()("flag0", po::value<int>(&flag0)->default_value(102));
        desc.add_options()("flag7", po::value<int>(&flag7)->default_value( 5)); // for tree height cutoff
        desc.add_options()("desired_correlation", po::value<double>(&desired_correlation)->default_value(0.1),
                           "Large value = more noise, a.k.a. weaker correlation");

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
    int M_ACORN = 32, Mb_ACORN = 64, gamma_ACORN = 80;
    string acorn_index_path(pf_path("data/index_files/acorn/txt/"));
    string acorn_postfix = to_string(M_ACORN) + "-" + to_string(Mb_ACORN) + "-" + to_string(gamma_ACORN);

    attr_bin_file = "Synthetic";
    query_tsv_file = "./queries_disjunctive.tsv";

    double input_k = 0.5;

    if (dataset == "sift3") {
        base_bin_file = home + "data/sift/sift_base.bin";
        query_bin_file = home + "data/sift/sift_query.bin";
        attr_bin_file = pf_path("workloads/mar30_new_mixed_attributes/step0_attributes/SIFT-mixed_attrs.bin");
        query_tsv_file = "./queries.tsv";

        index_path_prefix = home + "data/index_files/mar30_mixed_attrs/sift-3attr-" + "6way/"; input_k = 0.5;

        M = 32;
        ef_con = 200;
    } else if (dataset == "gist3") {
        base_bin_file = home + "data/gist/gist_base.bin";
        query_bin_file = home + "data/gist/gist_query.bin";
        attr_bin_file = pf_path("workloads/mar30_new_mixed_attributes/step0_attributes/GIST-mixed_attrs.bin");
        query_tsv_file = "./queries.tsv";

        index_path_prefix = home + "data/index_files/mar30_mixed_attrs/gist-3attr-" + "6way/"; input_k = 0.1;

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
    // vector<int> ef_searches = {10,50,100,200,400,1000,3000};
    // vector<int> ef_searches = {10,50,100};

    vector<int> ef_searches_acorn = {10,50,100,200,300,400,500,1000,1500,2000};
    // vector<int> ef_searches_acorn = {10,50,100,200,400,1000};

    distance_handler = ANNS::get_distance_handler(data_type, dist_fn);

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
    query_storage->truncate(1000);
    nq = query_storage->get_num_points();
    dim = base_storage->get_dim();
    na = attr.size() / nb;

    {
        // cancel the variance
        for (int i = 0; i < nb; i++) {
            float a = attr[i*3], b = attr[i*3+1];
            attr[i*3] = a;
            attr[i*3+1] = (b-a) / input_k;
        }

        // re-introduce the variance, but make sure attr[0] retains the same order
        for (int i = 0; i < nb; i++) {
            float a = attr[i*3], b = attr[i*3+1];
            attr[i*3] = a;
            attr[i*3+1] = a + b * desired_correlation;
        }
    }

    TableStorage tbstorage;
    TableStorage::load_table_storage_helper(base_storage, attr, tbstorage);
    tbstorage.describe();




    // generate queries instead of reading them
    vector<QueryType> preds = generate_queries(attr);

    auto predicate_direct_access = [&](int pid, int q) -> bool { return preds[q].check(attr.data() + pid * na); };

    cout << "\nLoading ground truth for " << nq << " queries...\n" << endl;
    auto gt = new pair<IdxType, float>[nq * K];

    if (false && load_gt_file(gt_path, gt, nq, K)) {
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

    string idx_and_estimator_prefix = index_path_prefix + "attr" + "_equidepth";
    vector<shared_ptr<irg_attr>> indices;
    for (int i = 0; i < na; i++) {
        idx_path.push_back(idx_and_estimator_prefix + to_string(i));
        indices.emplace_back();
    }

    #pragma omp parallel for
    for (int i = 0; i < na; i++) {
        cout << "Building graph for attribute #" << i << " using " << num_threads << " threads" << endl;
        int maxdepth = 2;
        indices[i] = irg_attr::build_or_load_graph_for_numerical_attr(attr, i, maxdepth, base_storage, idx_path[i], distance_handler, M, ef_con, num_threads, 6);

        indices[i]->_attr_id = i;
        indices[i]->_attr = &attr;
        indices[i]->_na = na;
    }

    // load root: if we run normally ("-1") or request to build attr #0 ("0")
    {
        string root_path = idx_and_estimator_prefix + "_root.txt";
        string root_upper_path = idx_and_estimator_prefix + "_root_upper.txt";
        AttrTreeNode *rootnode = indices[0]->tree->root;        // NOTE: we are putting the root at the 2nd position
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

    for (int i = 0; i < na; i++) {
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
        for (auto p : indices[i]->tree->treenodes) {
            if (p) {
                unit_graph_count++;
                total_edge_count += p->size_e;
            }
        }
        maxdepth = indices[i]->tree->max_depth;
        cout << "Index #" << i << " has " << unit_graph_count << " unit graphs; layer count (=maxdepth+1) = " << maxdepth + 1 << "; total_edges = " << total_edge_count << endl;
    }

    AttrTreeNode *rootgraph = nullptr;
    rootgraph = indices[0]->tree->root;
    indices[2]->tree->root->delegate_root = rootgraph;
    indices[1].reset();
    indices[1] = nullptr;

    BorrowPlanner density_planner_split(indices, attr);

    cout << "Root graph bounds: " << endl;
    cout << "Attr 0 min = " << indices[0]->tree->root->value_bounds[0].first << endl;
    cout << "Attr 0 max = " << indices[0]->tree->root->value_bounds[0].second << endl;
    cout << "Attr 1 min = " << indices[0]->tree->root->value_bounds[1].first << endl;
    cout << "Attr 1 max = " << indices[0]->tree->root->value_bounds[1].second << endl;
    cout << "Attr 2 min = " << indices[0]->tree->root->value_bounds[2].first << endl;
    cout << "Attr 2 max = " << indices[0]->tree->root->value_bounds[2].second << endl;
    cout << endl;

    cout << "Leaf graph bounds: " << endl;
    int wid = 15;
    cout << setw(wid) << "Attr 0 min" << setw(wid) << "max" << setw(wid) << "Attr 1 min" << setw(wid) << "max" << setw(wid) << "Attr 2 min" << setw(wid) << "max" << endl;
    for (auto leaf : density_planner_split.leaf_list[0]) {
        cout
        << setw(wid) << leaf->value_bounds[0].first
        << setw(wid) << leaf->value_bounds[0].second
        << setw(wid) << leaf->value_bounds[1].first
        << setw(wid) << leaf->value_bounds[1].second
        << setw(wid) << leaf->value_bounds[2].first
        << setw(wid) << leaf->value_bounds[2].second << endl;
    }
    // return 0;

    // get some statistics!
    std::vector<int> ks;
    ks.reserve(preds.size());
    std::vector<float> borrowed_vals;
    borrowed_vals.reserve(nb);
    for (int i = 0; i < nb; i++) {
        borrowed_vals.push_back(attr[i * na + 0]);
    }
    sort(borrowed_vals.begin(), borrowed_vals.end());

    for (int i = 0; i < (int)preds.size(); i++) {
        const auto &[qL, qR] = density_planner_split.derive_bound(preds[i], 0);
        // float qL, qR;
        // my_assert(preds[i].get_range_bounds(1, qL, qR));
        auto left = std::lower_bound(borrowed_vals.begin(), borrowed_vals.end(), qL);
        auto right = std::upper_bound(borrowed_vals.begin(), borrowed_vals.end(), qR);
        ks.push_back(right - left);
    }

    // compute min, max, mean
    sort(ks.begin(), ks.end());
    int min_k = *std::min_element(ks.begin(), ks.end());
    int max_k = *std::max_element(ks.begin(), ks.end());
    double mean_k = std::accumulate(ks.begin(), ks.end(), 0.0) / ks.size();
    int median_k = ks[ks.size() / 2];

    // compute stddev
    double var = 0.0;
    for (int k : ks) var += (k - mean_k) * (k - mean_k);
    var /= ks.size();
    double stddev = std::sqrt(var);

    std::cout << "Running tests on " << ks.size() << " queries in total\n";
    std::cout << "Total: " << nb << " tuples\n";
    std::cout << "\"Derived Range\" size statistics:\n";
    std::cout << "  min    = " << min_k    << " tuples\n";
    std::cout << "  max    = " << max_k    << " tuples\n";
    std::cout << "  mean   = " << mean_k   << " tuples\n";
    std::cout << "  median = " << median_k << " tuples\n";
    std::cout << "  stddev = " << stddev   << " tuples\n";

    // optional: simple histogram
    const int bins = 10;
    std::vector<int> hist(bins, 0);
    for (int k : ks) {
        int idx = std::min((int)((k - min_k) * bins / (double)(max_k - min_k + 1)), bins - 1);
        hist[idx]++;
    }

    std::cout << "Histogram (approx):\n";
    for (int i = 0; i < bins; i++) {
        std::cout << "  " << i << ": " << std::string(hist[i] / (ks.size() / 50 + 1), '*') << " (" << hist[i] << ")\n";
    }

    {
        int root_only_plans = 0;
        int borrowed_plans = 0;
        int plans_with_nonroot_graph = 0;
        int empty_plans = 0;
        int total_graph_refs = 0;
        int total_nonroot_graph_refs = 0;
        std::map<int, int> plan_size_hist;
        std::map<int, int> max_depth_hist;
        long long borrow_logic_ns = 0;
        long long selection_ns = 0;
        long long borrowed_query_borrow_logic_ns = 0;
        long long borrowed_query_execution_ns = 0;
        long long borrowed_query_search_ns = 0;
        long long all_query_execution_ns = 0;
        long long all_query_search_ns = 0;
        int borrowed_query_execution_count = 0;
        const int overhead_ef_search = 100;
        ANNS::SearchCacheList overhead_search_cache_list(num_threads + 5, nb, overhead_ef_search);

        for (int i = 0; i < (int)preds.size(); i++) {
            auto t_query_start = std::chrono::high_resolution_clock::now();
            auto t_borrow_start = std::chrono::high_resolution_clock::now();

            int index_id = -1;
            float max_sim = -1;
            for (int j = 0; j < (int)density_planner_split.corr.size(); j++) {
                if (density_planner_split.corr[density_planner_split.missing_id][j] >= max_sim) {
                    max_sim = density_planner_split.corr[density_planner_split.missing_id][j];
                    index_id = j;
                }
            }

            float inf = std::numeric_limits<float>::infinity();
            const auto &[A_min, A_max] = density_planner_split.derive_bound(preds[i], index_id);
            QueryType derived_pred(na);
            for (int j = 0; j < na; j++) {
                derived_pred.set_range(j, -inf, inf);
            }
            derived_pred.set_range(index_id, A_min, A_max);

            auto t_borrow_end = std::chrono::high_resolution_clock::now();

            vector<double> dummy_metrics;
            auto t_selection_start = std::chrono::high_resolution_clock::now();
            auto plan = density_planner_split.conjunctive_index_selection(derived_pred, dummy_metrics);
            auto t_selection_end = std::chrono::high_resolution_clock::now();

            borrow_logic_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t_borrow_end - t_borrow_start).count();
            selection_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t_selection_end - t_selection_start).count();
            long long this_borrow_logic_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_borrow_end - t_borrow_start).count();

            if (plan.empty()) {
                empty_plans++;
                plan_size_hist[0]++;
                max_depth_hist[-1]++;
                borrowed_plans++;
                auto t_query_end = std::chrono::high_resolution_clock::now();
                long long this_query_execution_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_query_end - t_query_start).count();
                all_query_execution_ns += this_query_execution_ns;
                borrowed_query_borrow_logic_ns += this_borrow_logic_ns;
                borrowed_query_execution_ns += this_query_execution_ns;
                borrowed_query_execution_count++;
                continue;
            }

            bool is_single_root = plan.size() == 1 && plan[0].ag_ptr != nullptr && plan[0].ag_ptr->depth == 0;
            bool has_nonroot = false;
            int max_depth = 0;
            for (const auto &gref : plan) {
                if (gref.ag_ptr == nullptr) {
                    continue;
                }
                max_depth = std::max(max_depth, gref.ag_ptr->depth);
                if (gref.ag_ptr->depth > 0) {
                    has_nonroot = true;
                    total_nonroot_graph_refs++;
                }
            }

            int n_cmps = 0;
            int one_predicate_k = 0;
            auto predicate_one = [&](int pid) {
                const float *attr_ptr = tbstorage.get_attr(tbstorage.lookup(pid));
                return preds[i].check_one(one_predicate_k, attr_ptr);
            };
            auto predicate = [&](int pid) {
                const float *attr_ptr = tbstorage.get_attr(tbstorage.lookup(pid));
                return preds[i].check(attr_ptr);
            };
            auto query = query_storage->get_vector(i);
            vector<pair<IdxType, float>> all;
            unordered_set<IdxType> seen;

            auto t_search_start = std::chrono::high_resolution_clock::now();
            for (const auto &graphref : plan) {
                one_predicate_k = graphref.attribute_graph_id;
                search_graph_reference_OOR(tbstorage, query, predicate, predicate_one, K, n_entry, graphref, overhead_search_cache_list, n_cmps, all, seen);
            }
            std::sort(all.begin(), all.end(), [](auto& a, auto& b) { return a.second < b.second; });
            if ((int)all.size() > K) {
                all.resize(K);
            }
            auto t_query_end = std::chrono::high_resolution_clock::now();

            long long this_search_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_query_end - t_search_start).count();
            long long this_query_execution_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_query_end - t_query_start).count();
            all_query_search_ns += this_search_ns;
            all_query_execution_ns += this_query_execution_ns;

            if (is_single_root) {
                root_only_plans++;
            } else {
                borrowed_plans++;
                borrowed_query_borrow_logic_ns += this_borrow_logic_ns;
                borrowed_query_search_ns += this_search_ns;
                borrowed_query_execution_ns += this_query_execution_ns;
                borrowed_query_execution_count++;
            }
            if (has_nonroot) {
                plans_with_nonroot_graph++;
            }
            total_graph_refs += plan.size();
            plan_size_hist[(int)plan.size()]++;
            max_depth_hist[max_depth]++;
        }

        double borrow_logic_ms = borrow_logic_ns / 1e6;
        double selection_ms = selection_ns / 1e6;
        double planner_ms = borrow_logic_ms + selection_ms;
        double borrowed_query_borrow_logic_ms = borrowed_query_borrow_logic_ns / 1e6;
        double borrowed_query_search_ms = borrowed_query_search_ns / 1e6;
        double borrowed_query_execution_ms = borrowed_query_execution_ns / 1e6;
        double all_query_search_ms = all_query_search_ns / 1e6;
        double all_query_execution_ms = all_query_execution_ns / 1e6;
        double total_queries = preds.empty() ? 1.0 : (double)preds.size();
        double root_only_pct = 100.0 * root_only_plans / total_queries;
        double borrowed_pct = 100.0 * borrowed_plans / total_queries;
        double nonroot_pct = 100.0 * plans_with_nonroot_graph / total_queries;
        double avg_plan_size = total_graph_refs / total_queries;
        double avg_nonroot_refs = total_nonroot_graph_refs / total_queries;
        double borrow_logic_pct = planner_ms == 0.0 ? 0.0 : 100.0 * borrow_logic_ms / planner_ms;
        double borrow_logic_pct_of_borrowed_execution = borrowed_query_execution_ms == 0.0 ? 0.0 : 100.0 * borrowed_query_borrow_logic_ms / borrowed_query_execution_ms;
        double borrow_logic_pct_of_all_execution = all_query_execution_ms == 0.0 ? 0.0 : 100.0 * borrow_logic_ms / all_query_execution_ms;
        double borrowed_execution_queries = borrowed_query_execution_count == 0 ? 1.0 : (double)borrowed_query_execution_count;

        cout << "R3W2 borrowing plan statistics:\n";
        cout << "  total_queries = " << preds.size() << "\n";
        cout << "  root_only_plans = " << root_only_plans << " (" << root_only_pct << "%)\n";
        cout << "  borrowed_plans = " << borrowed_plans << " (" << borrowed_pct << "%)\n";
        cout << "  plans_with_nonroot_graph = " << plans_with_nonroot_graph << " (" << nonroot_pct << "%)\n";
        cout << "  empty_plans = " << empty_plans << "\n";
        cout << "  avg_plan_size = " << avg_plan_size << "\n";
        cout << "  avg_nonroot_graph_refs = " << avg_nonroot_refs << "\n";
        cout << "  planner_time_ms = " << planner_ms << "\n";
        cout << "  planner_time_us_per_query = " << (planner_ms * 1000.0 / total_queries) << "\n";
        cout << "  borrow_logic_time_ms = " << borrow_logic_ms << "\n";
        cout << "  borrow_logic_us_per_query = " << (borrow_logic_ms * 1000.0 / total_queries) << "\n";
        cout << "  selection_time_ms = " << selection_ms << "\n";
        cout << "  selection_us_per_query = " << (selection_ms * 1000.0 / total_queries) << "\n";
        cout << "  borrow_logic_pct_of_planner = " << borrow_logic_pct << "\n";
        cout << "  overhead_ef_search = " << overhead_ef_search << "\n";
        cout << "  borrowed_query_execution_count = " << borrowed_query_execution_count << "\n";
        cout << "  borrowed_query_borrow_logic_time_ms = " << borrowed_query_borrow_logic_ms << "\n";
        cout << "  borrowed_query_borrow_logic_us_per_query = " << (borrowed_query_borrow_logic_ms * 1000.0 / borrowed_execution_queries) << "\n";
        cout << "  borrowed_query_search_time_ms = " << borrowed_query_search_ms << "\n";
        cout << "  borrowed_query_search_us_per_query = " << (borrowed_query_search_ms * 1000.0 / borrowed_execution_queries) << "\n";
        cout << "  borrowed_query_execution_time_ms = " << borrowed_query_execution_ms << "\n";
        cout << "  borrowed_query_execution_us_per_query = " << (borrowed_query_execution_ms * 1000.0 / borrowed_execution_queries) << "\n";
        cout << "  borrow_logic_pct_of_borrowed_execution = " << borrow_logic_pct_of_borrowed_execution << "\n";
        cout << "  all_query_search_time_ms = " << all_query_search_ms << "\n";
        cout << "  all_query_search_us_per_query = " << (all_query_search_ms * 1000.0 / total_queries) << "\n";
        cout << "  all_query_execution_time_ms = " << all_query_execution_ms << "\n";
        cout << "  all_query_execution_us_per_query = " << (all_query_execution_ms * 1000.0 / total_queries) << "\n";
        cout << "  borrow_logic_pct_of_all_execution = " << borrow_logic_pct_of_all_execution << "\n";
        cout << "  plan_size_histogram:";
        for (auto &[size, count] : plan_size_hist) {
            cout << " " << size << ":" << count;
        }
        cout << "\n";
        cout << "  max_depth_histogram:";
        for (auto &[depth, count] : max_depth_hist) {
            cout << " " << depth << ":" << count;
        }
        cout << "\n";

        if (flag0 == 110) {
            return 0;
        }
    }

    // return 0;





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

    omp_set_num_threads(num_threads);

    try {
        AttrTreeNode *rootgraph = nullptr;
        rootgraph = indices[0]->tree->root;
        UseSingleUnitGraph simple_out_of_range_search(rootgraph);
        work_split(simple_out_of_range_search, preds);

        work_split(density_planner_split, preds);   // Our method (unified index selection)
        cerr << "^ split (alpha = " << density_planner_split.alpha << ")" << endl;
    } catch (std::exception const &e) {
        cout << "Caught error!" << endl;
        cout << e.what() << endl;
        cerr << "Caught error!" << endl;
        cerr << e.what() << endl;
        return -1;
    }

    return 0;
}
