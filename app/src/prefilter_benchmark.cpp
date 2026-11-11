#include "prefilter_scan.h"
#include "tablestorage/tablestorage_utils.h"
#include "querytype.h"
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>

using namespace ANNS;

template<class Predicate>
void run(TableStorage& table, IStorage& queries, DistanceHandler& distance,
        std::vector<Predicate> predicates, const std::string& key,
        const std::string& gt_path, const std::string& ids_path) {
    const int k = 10;
    std::vector<int> query_ids;
    if (ids_path != "-") {
        std::ifstream f(ids_path);
        int id;
        while (f >> id) query_ids.push_back(id);
        if (query_ids.size() != predicates.size()) throw std::runtime_error("query ID count mismatch");
    } else {
        predicates.resize(std::min(predicates.size(), size_t(queries.get_num_points())));
        query_ids.resize(predicates.size());
        std::iota(query_ids.begin(), query_ids.end(), 0);
    }
    for (int id : query_ids) if (id < 0 || id >= int(queries.get_num_points())) throw std::runtime_error("invalid query ID");
    std::vector<std::pair<IdxType, float>> gt(predicates.size()*k);
    if (!load_gt_file(gt_path, gt.data(), predicates.size(), k)) throw std::runtime_error("missing ground truth: " + gt_path);
    std::vector<int> selected(predicates.size());
    std::iota(selected.begin(), selected.end(), 0);
    if (selected.size() > 1000) {
        std::mt19937 rng(42);
        std::shuffle(selected.begin(), selected.end(), rng);
        selected.resize(1000);
        std::sort(selected.begin(), selected.end());
    }
    std::vector<Predicate> sampled;
    std::vector<int> sampled_ids;
    std::vector<std::pair<IdxType, float>> sampled_gt;
    for (int i : selected) {
        sampled.push_back(predicates[i]);
        sampled_ids.push_back(query_ids[i]);
        sampled_gt.insert(sampled_gt.end(), gt.begin()+i*k, gt.begin()+(i+1)*k);
    }
    auto result = prefilter_scan(table, queries, sampled, sampled_ids, distance, k);
    double recall = calculate_recall(sampled_gt.data(), result.hits.data(), sampled.size(), k, false);
    size_t valid = std::count_if(sampled_gt.begin(), sampled_gt.end(), [](const auto& hit) { return hit.first != -1; });
    double attainable = 100.0 * valid / sampled_gt.size();
    if (recall + 0.001 < attainable) throw std::runtime_error("Exact scan does not match saved ground truth: " + key);
    std::cout << std::setprecision(12) << "RESULT " << key << " " << omp_get_max_threads()
              << " " << sampled.size() << " " << sampled.size()/result.seconds
              << " " << recall << " " << double(result.comparisons)/sampled.size()
              << " " << result.seconds << std::endl;
}

int main(int argc, char** argv) {
#ifdef BYPASS_B_TREE
    throw std::runtime_error("prefilter_benchmark requires a B-tree build");
#endif
    if (argc != 8) throw std::runtime_error("usage: prefilter_benchmark datatype base query attrs threads workload-list dataset");
    omp_set_num_threads(std::stoi(argv[5]));
    auto base = create_storage(argv[1]); base->load_from_file(argv[2], "");
    auto queries = create_storage(argv[1]); queries->load_from_file(argv[3], "");
    auto attrs = read_binary_float32(argv[4]);
    TableStorage table;
    TableStorage::load_table_storage_helper(base, attrs, table);
    table.describe();
    base->clean(); attrs.clear(); attrs.shrink_to_fit();
    auto distance = get_distance_handler(argv[1], "L2");
    std::ifstream work(argv[6]);
    if (!work) throw std::runtime_error("cannot open workload list");
    std::string key, kind, tsv, gt, ids;
    while (work >> key >> kind >> tsv >> gt >> ids) {
        std::cout << "START " << argv[7] << " " << key << std::endl;
        if (kind == "conj") {
            auto [preds, schema] = read_queries_from_tsv(tsv);
            if (schema.names.size() != table.na) throw std::runtime_error("attribute count mismatch");
            run(table, *queries, *distance, preds, key, gt, ids);
        } else if (kind == "disj") {
            auto [preds, schema] = read_disjunctive_queries_from_tsv(tsv, table.na);
            run(table, *queries, *distance, preds, key, gt, ids);
        } else throw std::runtime_error("unknown workload kind");
    }
}
