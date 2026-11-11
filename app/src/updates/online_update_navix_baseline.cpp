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
    cfg.method_name = "NaviX";
    cfg.artifact_method = "hnsw";
    cfg.search_mode = BaselineSearchMode::NaviX;

    try {
        po::options_description desc{"Online Update NaviX Baseline"};
        desc.add_options()
            ("help,h", "Print help")
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
            ("n_entry", po::value<int>(&cfg.navix_n_entry)->default_value(15));

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
