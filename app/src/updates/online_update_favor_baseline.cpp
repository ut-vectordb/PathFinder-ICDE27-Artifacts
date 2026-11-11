#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

#include <boost/program_options.hpp>

#include "online_update_baseline_common.h"

namespace po = boost::program_options;
using namespace std;
using namespace ANNS;

int main(int argc, char **argv) {
    BaselineRunConfig cfg;
    cfg.method_name = "FAVOR";
    cfg.artifact_method = "hnsw";
    cfg.search_mode = BaselineSearchMode::FAVOR;
    cfg.ef_descending = true;   // as for the other FAVOR runs: largest ef first

    try {
        po::options_description desc{"Online Update FAVOR Baseline (stock-HNSW updates + upstream delta_d growth)"};
        desc.add_options()
            ("help,h", "Print help")
            ("stress_seconds", po::value<int>(&cfg.stress_seconds)->default_value(0), "Concurrent 1:1:1 query/insert/delete test duration")
            ("stress_ef", po::value<int>(&cfg.stress_ef)->default_value(50))
            ("stress_delta", po::value<double>(&cfg.stress_delta)->default_value(-1), "Previously calibrated initial delta_d")
            ("mixed_ops", po::value<int>(&cfg.mixed_ops)->default_value(0))
            ("update_percent", po::value<int>(&cfg.update_percent)->default_value(1))
            ("mixed_ef", po::value<int>(&cfg.mixed_ef)->default_value(50))
            ("calibrate_only", po::bool_switch(&cfg.calibrate_only), "Calibrate initial mixed-query recall; do not update")
            ("calibration_max_ef", po::value<int>(&cfg.calibration_max_ef)->default_value(3000))
            ("calibration_target", po::value<int>(&cfg.calibration_target)->default_value(0), "Strict recall target; 0 evaluates 90 and 95")
            ("dataset", po::value<string>(&cfg.dataset)->default_value("sift3"))
            ("num_threads", po::value<uint32_t>(&cfg.num_threads)->default_value(16))
            ("ef_con", po::value<int>(&cfg.ef_con)->default_value(200))
            ("n_batches", po::value<int>(&cfg.n_batches)->default_value(100))
            ("favor_initial", po::value<string>(&cfg.favor_initial)->default_value("capacity"), "capacity | full (see FavorUpdateState)")
            ("ef_descending", po::value<bool>(&cfg.ef_descending)->default_value(true))
            ("update_threads", po::value<uint32_t>(&cfg.update_threads)->default_value(0), "threads for update batches only (0 = num_threads)");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const exception &ex) {
        cerr << ex.what() << endl;
        return 1;
    }

    return run_baseline_experiment(cfg);
}
