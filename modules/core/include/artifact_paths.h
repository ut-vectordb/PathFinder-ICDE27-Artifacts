#pragma once
// Datasets, query workloads and saved indexes are resolved under one root
// directory, given by the PATHFINDER_DATA_ROOT environment variable (default:
// the current directory). See the "Data layout" section of README.md.
#include <cstdlib>
#include <string>

inline std::string pf_path(const std::string &relative) {
    const char *root = std::getenv("PATHFINDER_DATA_ROOT");
    std::string base = (root && *root) ? root : ".";
    if (base.back() != '/') base += '/';
    return base + relative;
}
