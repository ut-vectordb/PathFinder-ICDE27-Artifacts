/**
 * build_partial_pathfinder.cpp
 *
 * Builds PathFinder (irg_attr) trees on a random 80% subset of base vectors.
 * Reads a TSV query file header to determine each attribute's schema
 * (NUMERICAL for *_Range, CATEGORICAL for *_Categorical).
 *
 * Saves:
 *   - attr{i}_tree.bin  : per numerical attribute (segment tree)
 *   - attr{i}_hash.bin  : per categorical attribute (hash collection)
 *   - initial_ids.bin   : binary mapping packed_id → global_id
 *   - schema.bin        : per-attribute schema (0=NUMERICAL, 1=CATEGORICAL)
 *
 * The remaining 20% are left for online insertion testing.
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <filesystem>

#include <boost/program_options.hpp>
#include <omp.h>

#include "hnsw/hnsw.h"
#include "irangegraph/attribute_tree.h"
#include "tablestorage/tablestorage_utils.h"
#include "querytype/querytype.h"
#include "run_search_online.h"
#include "utils.h"
#include "artifact_paths.h"

namespace po = boost::program_options;
using namespace std;
using namespace ANNS;

int main(int argc, char **argv) {
    uint32_t num_threads = 64;
    double initial_fraction = 0.8;
    int seed = 42;
    string dataset = "sift3";
    int maxdepth = 3;
    int fanout = 6;

    try {
        po::options_description desc{"Build Partial PathFinder"};
        desc.add_options()
            ("help,h", "Print help")
            ("dataset", po::value<string>(&dataset)->default_value("sift3"),
             "Dataset: sift3 or gist3")
            ("num_threads", po::value<uint32_t>(&num_threads)->default_value(64))
            ("fraction", po::value<double>(&initial_fraction)->default_value(0.8),
             "Fraction of base vectors for the initial index")
            ("seed", po::value<int>(&seed)->default_value(42))
            ("maxdepth", po::value<int>(&maxdepth)->default_value(3))
            ("fanout", po::value<int>(&fanout)->default_value(6))
            ;

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) { cout << desc; return 0; }
        po::notify(vm);
    } catch (const exception &ex) {
        cerr << ex.what() << endl;
        return 1;
    }

    omp_set_num_threads(num_threads);

    // ---- dataset paths ----
    string data_root = pf_path("data/");
    string attr_root = pf_path("workloads/mar30_new_mixed_attributes/step0_attributes/");
    string index_root = pf_path("data/index_files/apr1_online_update/");

    string base_bin_file, attr_bin_file, index_prefix, query_tsv_file;
    IdxType M;
    int ef_con = 200;

    string tsv_root = pf_path("workloads/mar30_new_mixed_attributes/step2/");

    if (dataset == "sift3") {
        base_bin_file  = data_root + "sift/sift_base.bin";
        attr_bin_file  = attr_root + "SIFT-mixed_attrs.bin";
        query_tsv_file = tsv_root + "pred_sift_a012_mid.tsv";
        index_prefix   = index_root + "sift3-pathfinder/";
        M = 32;
    } else if (dataset == "gist3") {
        base_bin_file  = data_root + "gist/gist_base.bin";
        attr_bin_file  = attr_root + "GIST-mixed_attrs.bin";
        // TODO: keep builder/test query TSV selection consistent once the
        // intended GIST evaluation split is finalized.
        query_tsv_file = tsv_root + "pred_gist_a012_mid.tsv";
        index_prefix   = index_root + "gist3-pathfinder/";
        M = 64;
    } else if (dataset == "redcaps") {
        string home_redcaps = pf_path("datasets/redcaps/");
        base_bin_file  = home_redcaps + "redcaps-embeddings.bin";
        attr_bin_file  = home_redcaps + "redcaps-attributes-subreddit-time-upvote.bin";
        query_tsv_file = pf_path("workloads/oct20_tsv_two_three_mixed/redcaps_two_three_mid.tsv");
        index_prefix   = index_root + "redcaps-pathfinder/";
        M = 64;
    } else {
        cerr << "Unknown dataset: " << dataset << endl;
        return 1;
    }

    // ---- determine per-attribute schema from TSV header ----
    auto [queries_unused, header] = read_queries_from_tsv(query_tsv_file);
    header.describe();
    vector<bool> is_numerical(header.types.size());
    for (size_t i = 0; i < header.types.size(); i++)
        is_numerical[i] = (header.types[i] == QueryType::PredType::RANGE);

    // ---- load full base vectors ----
    cout << "Loading base vectors from " << base_bin_file << "..." << endl;
    auto base_storage = create_storage("float");
    base_storage->load_from_file(base_bin_file, "");
    int nb = base_storage->get_num_points();
    int dim = base_storage->get_dim();
    cout << "  nb=" << nb << " dim=" << dim << endl;

    // ---- load full attributes ----
    cout << "Loading attributes from " << attr_bin_file << "..." << endl;
    vector<float> full_attr = read_binary_float32(attr_bin_file);
    int na = full_attr.size() / nb;
    cout << "  na=" << na << " (total floats=" << full_attr.size() << ")" << endl;
    if ((size_t)na * nb != full_attr.size()) {
        cerr << "Attribute array size mismatch!" << endl;
        return 1;
    }

    // ---- select random 80% subset ----
    int n_initial = (int)(nb * initial_fraction);
    cout << "Selecting " << n_initial << " / " << nb << " points ("
         << 100.0 * initial_fraction << "%) with seed=" << seed << endl;

    vector<IdxType> all_ids(nb);
    iota(all_ids.begin(), all_ids.end(), 0);
    mt19937 rng(seed);
    shuffle(all_ids.begin(), all_ids.end(), rng);

    vector<IdxType> initial_ids(all_ids.begin(), all_ids.begin() + n_initial);
    sort(initial_ids.begin(), initial_ids.end());

    // ---- reorder storage to subset ----
    cout << "Reordering storage to subset..." << endl;
    base_storage->reorder_data(initial_ids);
    cout << "  Subset storage: " << base_storage->get_num_points() << " points" << endl;

    // ---- build subset attribute array ----
    // After reorder, storage[packed_id] = vector[initial_ids[packed_id]].
    // Attribute array must match: subset_attr[packed_id * na + a] = full_attr[global_id * na + a]
    vector<float> subset_attr(n_initial * na);
    for (int i = 0; i < n_initial; i++)
        for (int a = 0; a < na; a++)
            subset_attr[(size_t)i * na + a] = full_attr[(size_t)initial_ids[i] * na + a];

    shared_ptr<DistanceHandler> distance_handler = get_distance_handler("float", "L2");

    // ---- create output directory ----
    filesystem::create_directories(index_prefix);

    // ---- build one tree per attribute ----
    for (int attr_id = 0; attr_id < na; attr_id++) {
        bool numerical = is_numerical[attr_id];
        string kind = numerical ? "NUMERICAL" : "CATEGORICAL";

        cout << "\n=== Building " << kind << " index for attribute " << attr_id
             << " (M=" << M << ", ef_con=" << ef_con;
        if (numerical) cout << ", maxdepth=" << maxdepth << ", fanout=" << fanout;
        cout << ") ===" << endl;

        auto t0 = chrono::high_resolution_clock::now();

        auto index = make_shared<irg_attr>();
        if (numerical) {
            index->schema = irg_attr::NUMERICAL;
            index->build(base_storage, subset_attr, attr_id, maxdepth,
                         distance_handler, M, ef_con, num_threads, fanout);
        } else {
            index->schema = irg_attr::CATEGORICAL;
            // depth_max=100 is used by build_or_load_graph_for_hash_attr
            index->build(base_storage, subset_attr, attr_id, 100,
                         distance_handler, M, ef_con, num_threads);
        }

        auto t1 = chrono::high_resolution_clock::now();
        double build_s = chrono::duration<double>(t1 - t0).count();

        cout << "  Build time: " << build_s << " s" << endl;

        if (numerical) {
            cout << "  Tree nodes: " << index->tree->treenodes.size() << endl;
            cout << "  Max depth: " << index->tree->max_depth << endl;
            string tree_path = index_prefix + "attr" + to_string(attr_id) + "_tree.bin";
            cout << "  Saving to " << tree_path << "..." << endl;
            index->save_tree(tree_path);
        } else {
            cout << "  Hash buckets: " << index->hash->hash_table.size() << endl;
            string hash_path = index_prefix + "attr" + to_string(attr_id) + "_hash.bin";
            cout << "  Saving to " << hash_path << "..." << endl;
            index->save_hash(hash_path);
        }
    }

    // ---- build HNSW root graph on subset ----
    {
        string root_path = index_prefix + "root.txt";
        string root_upper_path = index_prefix + "root_upper.txt";
        cout << "\n=== Building HNSW root graph (M=" << M
             << ", ef_con=" << ef_con << ") ===" << endl;
        auto t0 = chrono::high_resolution_clock::now();

        Hnsw h;
        auto g1 = make_shared<Graph>(n_initial);
        h.build_HNSWLib(base_storage, distance_handler, g1, M, ef_con,
                        num_threads, /*randomize=*/false, /*merge_upper_layers=*/false);

        auto t1 = chrono::high_resolution_clock::now();
        cout << "  Build time: " << chrono::duration<double>(t1 - t0).count() << " s" << endl;

        cout << "  Saving root graph to " << root_path << "..." << endl;
        g1->save(root_path);
        cout << "  Saving root upper layer to " << root_upper_path << "..." << endl;
        h._upper_layer->save(root_upper_path);
    }

    // ---- build Vamana root graph on subset ----
    {
        string vamana_root_path = index_prefix + "root_vamana.txt";
        cout << "\n=== Building Vamana root graph (M=" << M
             << ", ef_con=" << ef_con << ") ===" << endl;
        auto t0 = chrono::high_resolution_clock::now();

        Hnsw h(true);
        h.fixed_alpha = 1.2f;
        auto g1 = make_shared<Graph>(n_initial);
        h.build_NSW(base_storage, distance_handler, g1, M, ef_con, num_threads);

        auto t1 = chrono::high_resolution_clock::now();
        cout << "  Build time: " << chrono::duration<double>(t1 - t0).count() << " s" << endl;

        cout << "  Saving Vamana root graph to " << vamana_root_path << "..." << endl;
        g1->save(vamana_root_path);
    }

    // ---- save ID mappings ----
    string mapping_path = index_prefix + "initial_ids.bin";
    string hnsw_mapping_path = index_prefix + "initial_ids_hnsw.bin";
    string vamana_mapping_path = index_prefix + "initial_ids_vamana.bin";
    cout << "\nSaving ID mapping (" << n_initial << " IDs) to " << mapping_path << "..." << endl;
    save_id_mapping(mapping_path, initial_ids);
    cout << "Saving HNSW ID mapping to " << hnsw_mapping_path << "..." << endl;
    save_id_mapping(hnsw_mapping_path, initial_ids);
    cout << "Saving Vamana ID mapping to " << vamana_mapping_path << "..." << endl;
    save_id_mapping(vamana_mapping_path, initial_ids);

    cout << "\n=== Summary ===" << endl;
    cout << "  Dataset: " << dataset << endl;
    cout << "  Initial fraction: " << initial_fraction << endl;
    cout << "  Points in index: " << n_initial << " / " << nb << endl;
    cout << "  Attributes: " << na << endl;
    cout << "  Fanout: " << fanout << ", Max depth: " << maxdepth << endl;
    cout << "  Index prefix: " << index_prefix << endl;
    cout << "\nDone." << endl;
    return 0;
}
