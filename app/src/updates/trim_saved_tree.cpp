// Load/save through the repository's native tree format. No graph rebuilding.
// Usage: trim_saved_tree --max-depth 2 --expected-depth 3 --output-dir DIR tree1 tree2
// Explicit --replace is also supported.
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include "irangegraph/attribute_tree.h"

namespace fs = std::filesystem;
using namespace ANNS;

static void require(bool ok, const std::string &message) {
    if (!ok) throw std::runtime_error(message);
}

// irg_attr does not own its raw tree/node pointers through a destructor.
struct OwnedTree {
    irg_attr index;
    OwnedTree() { index.tree = nullptr; index.schema = irg_attr::NUMERICAL; }
    ~OwnedTree() {
        if (index.tree) {
            for (auto *node : index.tree->treenodes) delete node;
            delete index.tree;
        }
    }
    void load(const fs::path &path) {
        require(fs::is_regular_file(path), "Missing tree: " + path.string());
        // load_tree only deserializes here; vector storage and distance are unused.
        index.load_tree(path.string(), nullptr, nullptr, 64, 200);
        require(index.tree && index.tree->root, "Could not load tree");
    }
};

static void sync_file(const fs::path &path) {
    int fd = open(path.c_str(), O_RDONLY);
    require(fd >= 0, "Cannot open for fsync: " + path.string());
    int result = fsync(fd);
    close(fd);
    require(result == 0, "fsync failed: " + path.string());
}

int main(int argc, char **argv) {
    std::vector<std::pair<fs::path, fs::path>> staged;
    try {
        require(argc >= 7 && std::string(argv[1]) == "--max-depth" &&
                std::string(argv[3]) == "--expected-depth" &&
                (std::string(argv[5]) == "--replace" ||
                 (std::string(argv[5]) == "--output-dir" && argc >= 8)),
                "Usage: trim_saved_tree --max-depth N --expected-depth OLD (--replace | --output-dir DIR) TREE...");
        bool replace = std::string(argv[5]) == "--replace";
        fs::path output_dir;
        if (!replace) {
            output_dir = fs::absolute(argv[6]);
            fs::create_directories(output_dir);
        }
        int depth = std::stoi(argv[2]), expected = std::stoi(argv[4]);
        require(depth >= 0 && expected > depth, "Require 0 <= target depth < expected depth");
        std::vector<fs::path> inputs;
        for (int i = replace ? 6 : 7; i < argc; ++i) {
            fs::path path = fs::canonical(argv[i]);
            require(std::find(inputs.begin(), inputs.end(), path) == inputs.end(), "Duplicate tree path");
            if (!replace) require(!fs::exists(output_dir / path.filename()), "Output already exists");
            inputs.push_back(path);
            // Renumbering tree nodes would also require remapping HNSW sidecars.
            for (const auto &entry : fs::directory_iterator(path.parent_path()))
                require(entry.path().filename().string().find(path.filename().string() + "_hnsw_upper_node") != 0,
                        "HNSW sidecars are not supported by this NSW-tree utility");
        }
        for (const auto &path : inputs) {
            OwnedTree source;
            source.load(path);
            auto &tree = *source.index.tree;
            require(tree.max_depth == expected, "Unexpected source depth: " + path.string());
            size_t before = tree.treenodes.size();
            source.index.cutoff_too_deep(depth);
            // cutoff leaves null slots; save_tree expects a compact preorder list.
            auto &nodes = tree.treenodes;
            nodes.erase(std::remove(nodes.begin(), nodes.end(), nullptr), nodes.end());
            for (size_t i = 0; i < nodes.size(); ++i) nodes[i]->node_id = int(i);
            fs::path destination = replace ? path : output_dir / path.filename();
            fs::path tmp = destination.string() + ".trim-" + std::to_string(getpid()) + ".tmp";
            require(!fs::exists(tmp), "Temporary output already exists");
            staged.emplace_back(destination, tmp);
            source.index.save_tree(tmp.string());
            {
                OwnedTree output;
                output.load(tmp);
                const auto &saved = output.index.tree->treenodes;
                require(output.index.tree->max_depth == depth && saved.size() == nodes.size(),
                        "Exported tree shape mismatch");
                require(output.index.tree->size_threshold == tree.size_threshold, "Threshold mismatch");
                for (size_t i = 0; i < nodes.size(); ++i) {
                    const auto *a = nodes[i], *b = saved[i];
                    require(a->lbound == b->lbound && a->rbound_open == b->rbound_open &&
                            a->depth == b->depth && a->size == b->size &&
                            a->childs.size() == b->childs.size() && a->vertex_table == b->vertex_table,
                            "Retained node/mapping mismatch");
                    for (int row = 0; row < a->size; ++row)
                        require(a->g->neighbors[row] == b->g->neighbors[row], "Retained adjacency mismatch");
                    for (size_t j = 0; j < a->childs.size(); ++j)
                        require(a->childs[j]->node_id == b->childs[j]->node_id, "Child link mismatch");
                }
            }
            fs::permissions(tmp, fs::status(path).permissions());
            sync_file(tmp);
            std::cout << "Verified " << path << ": depth " << expected << " -> " << depth
                      << ", nodes " << before << " -> " << nodes.size()
                      << "; all retained mappings and adjacency lists unchanged" << std::endl;
        }
        // Prepare and verify every output before replacing any input. Each rename
        // is atomic; multiple files are not a single filesystem transaction.
        for (const auto &[path, tmp] : staged) {
            fs::rename(tmp, path);
            sync_file(path.parent_path());
            std::cout << (replace ? "Replaced " : "Exported ") << path << std::endl;
        }
    } catch (const std::exception &e) {
        for (const auto &[path, tmp] : staged) {
            std::error_code ec;
            fs::remove(tmp, ec);
        }
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
