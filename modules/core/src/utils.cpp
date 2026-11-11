#include <set>
#include "utils.h"
#include <algorithm>


namespace ANNS {

    void write_kv_file(const std::string& filename, const std::map<std::string, std::string>& kv_map) {
        std::ofstream out(filename);
        for (auto& kv : kv_map) {
            out << kv.first << "=" << kv.second << std::endl;
        }
        out.close();
    }


    std::map<std::string, std::string> parse_kv_file(const std::string& filename) {
        std::map<std::string, std::string> kv_map;
        std::ifstream in(filename);
        if (!in.is_open()) {
            std::cerr << "Error: cannot open file " << filename << std::endl;
            exit(-1);
        }
        std::string line;
        while (std::getline(in, line)) {
            size_t pos = line.find("=");
            if (pos == std::string::npos)
                continue;
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            kv_map[key] = value;
        }
        in.close();
        return kv_map;
    }


    void write_gt_file(const std::string& filename, const std::pair<IdxType, float>* gt, uint32_t num_queries, uint32_t K) {
        std::ofstream fout(filename, std::ios::binary);
        fout.write(reinterpret_cast<const char*>(gt), num_queries * K * sizeof(std::pair<IdxType, float>));
        std::cout << "Ground truth written to " << filename << std::endl;
    }


    bool load_gt_file(const std::string& filename, std::pair<IdxType, float>* gt, uint32_t num_queries, uint32_t K) {
        std::ifstream fin(filename, std::ios::binary);
        if (!fin) {
            // std::cerr << "Ground truth file does not exist: " << filename << std::endl;
            // abort();
            return false;
        }
        std::size_t expected_size = num_queries * K * sizeof(std::pair<IdxType, float>);
        fin.read(reinterpret_cast<char*>(gt), expected_size);

        // check how many bytes were actually read
        if (static_cast<std::size_t>(fin.gcount()) != expected_size) {
            return false; // file too short
        }

        fin.close();
        // fin.read(reinterpret_cast<char*>(gt), num_queries * K * sizeof(std::pair<IdxType, float>));
        // fin.close();
        std::cout << "Ground truth loaded from " << filename << std::endl;

        // scan for -1 and output the total count as warning
        uint32_t invalid = 0;
        for (uint32_t i = 0; i < num_queries * K; ++i) {
            if (gt[i].first == -1) {
                invalid++;
            }
        }
        if (invalid > 0) {
            double max_recall = 100.0 * (K*num_queries - invalid) / (K*num_queries);
            std::cout << "Warning: Ground truth contains " << invalid << " invalid entries (-1), "
                      << "maximum recall will be " << max_recall << "%." << std::endl;
        }

        return true;
    }


    void clean_results(std::vector<std::pair<IdxType, float>>& input, std::vector<std::pair<IdxType, float>>& cleaned, uint32_t K) {
        // Step 1: Sort by IdxType
        std::sort(input.begin(), input.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        // Step 2: Remove duplicates (keep first occurrence)
        std::vector<std::pair<IdxType, float>> unique;
        unique.reserve(input.size());
        for (size_t j = 0; j < input.size(); ++j) {
            if (j == 0 || input[j].first != input[j - 1].first) {
                unique.push_back(input[j]);
            }
        }

        // Step 3: Sort by float value (ascending)
        std::sort(unique.begin(), unique.end(), [](const auto& a, const auto& b) {
            return a.second < b.second;
        });

        cleaned = std::move(unique);
    }


// #define ALTERNATIVE_DEFINITION
    float calculate_recall(const std::pair<IdxType, float>* gt, const std::pair<IdxType, float>* results, uint32_t num_queries, uint32_t K, bool verbose) {
        float total_correct = 0;
        float recall = 0;
        int last_warn = 0;
        for (uint32_t i = 0; i < num_queries; i++) {
            // prepare ground truth set, offset records the last valid gt index
            std::set<IdxType> gt_set;
            int32_t offset = -1;
            for (uint32_t j = 0; j < K; j++) {
                if (gt[i * K + j].first != -1) {
                    offset = j;
                    gt_set.insert(gt[i * K + j].first);     
                }
                // else {
                //     invalid++;
                // }
            }
            
            // count the correct
            float this_correct = 0;
            float this_valid = offset + 1;
            std::vector<std::pair<IdxType, float>> res, cleaned;
            for (uint32_t j = 0; j < K; j++) {
                res.push_back(results[i * K + j]);
            }
            clean_results(res, cleaned, K);
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
            if (verbose && this_correct <= 0.7 * K) {
                std::cerr << "Warning: query #" << i << " has low recall of " << this_correct / K << std::endl;
                last_warn = i;
            }
            total_correct += this_correct;
            if (this_valid == 0)
                recall += 1;
            else
                recall += this_correct / this_valid;
        }
        if (verbose) {
            std::cerr << "#" << last_warn << ": ";
            for (int k = 0; k < K; ++k) {
                std::cerr << results[last_warn * K + k].first << " ";
            }
            std::cerr << std::endl;
            std::cerr << "#" << last_warn << ": ";
            for (int k = 0; k < K; ++k) {
                std::cerr << gt[last_warn * K + k].first << " ";
            }
            std::cerr << std::endl;

            for (int k = 0; k < K; ++k) {
                std::cerr << results[last_warn * K + k].second << " ";
            }
            std::cerr << std::endl;
            for (int k = 0; k < K; ++k) {
                std::cerr << gt[last_warn * K + k].second << " ";
            }
            std::cerr << std::endl;
        }
#ifdef ALTERNATIVE_DEFINITION
        return 100.0 * recall / num_queries;
#else
        return 100.0 * total_correct / (num_queries * K);
#endif
    }
}