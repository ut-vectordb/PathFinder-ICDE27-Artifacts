#ifndef TABLESTORAGE_UTILS_H

#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <filesystem>

#include <sys/stat.h>

namespace fs = std::filesystem;

#define my_assert(expr) \
    ((expr) ? (void)0 : my_assert_fail(#expr, __FILE__, __LINE__))

inline void my_assert_fail(const char* expr_str, const char* file, int line) {
    std::cerr << "Assertion failed: (" << expr_str << "), file " << file << ", line " << line << std::endl;
    std::abort();
}

std::vector<float> fvecs_read(const char* fname, size_t& d_out, size_t& n_out);
size_t get_file_size(const std::string& filename);
std::vector<float> read_binary_float32(const std::string &filename);
std::vector<float> csv_read(const char* fname, size_t& d_out, size_t& n_out, char sep = ',', bool has_header = false);
bool needs_regeneration(const std::string& input_file, const std::string& output_file);

#define TABLESTORAGE_UTILS_H
#endif // TABLESTORAGE_UTILS_H