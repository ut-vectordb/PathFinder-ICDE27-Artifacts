#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <mutex>
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <charconv>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include "config.h"


namespace ANNS {

    class Graph {
            
        public:
            IdxType _num_points;

            std::vector<IdxType>* neighbors;
            std::mutex* neighbor_locks;

            Graph() = default;

            Graph(IdxType num_points) {
                _num_points = num_points;
                neighbors = new std::vector<IdxType>[num_points];
                neighbor_locks = new std::mutex[num_points];
            };

            Graph(std::shared_ptr<Graph> graph, IdxType start, IdxType end) {
                neighbors = graph->neighbors + start;
                neighbor_locks = graph->neighbor_locks + start;
                _num_points = end - start;
            };

            virtual void save(const std::string& filename) {
                std::filesystem::path filepath(filename);
                if (filepath.has_parent_path()) {
                    std::filesystem::create_directories(filepath.parent_path());
                }
                std::ofstream out(filename);
                if (!out) {
                    throw std::runtime_error("Failed to open graph file for writing: " + filename);
                }
                for (IdxType i = 0; i < _num_points; i++) {
                    out << i << " ";
                    for (auto& neighbor : neighbors[i])
                        out << neighbor << " ";
                    out << std::endl;
                }
                out.close();
                save_binary_cache(filename + "_mmap_cache");
            }

            virtual void load(const std::string& filename) {
                std::string cache_path = filename + "_mmap_cache";
                if (std::filesystem::exists(cache_path)) {
                    try {
                        load_binary_cache(cache_path);
                        return;
                    } catch (...) {
                        // cache is corrupt or incompatible, fall through to text load
                    }
                }
                load_text(filename);
                try {
                    save_binary_cache(cache_path);
                } catch (...) {
                    // non-fatal: cache write failure shouldn't break load
                }
            }

            virtual size_t get_index_size() {
                size_t index_size = 0;
                for (IdxType i = 0; i < _num_points; i++)
                    index_size += neighbors[i].size() * sizeof(IdxType);
                return index_size;
            }

            virtual void clean() {
                if (neighbors != nullptr)
                    delete[] neighbors;
                if (neighbor_locks != nullptr)
                    delete[] neighbor_locks;
                neighbors = nullptr;
                neighbor_locks = nullptr;
            }

            ~Graph() {
                clean();
            }

        private:
            static constexpr uint32_t CACHE_MAGIC = 0x47524148; // "GRAH"
            static constexpr uint32_t CACHE_VERSION = 1;

            void save_binary_cache(const std::string& cache_path) {
                std::string tmp_path = cache_path + ".tmp";
                std::filesystem::path filepath(tmp_path);
                if (filepath.has_parent_path()) {
                    std::filesystem::create_directories(filepath.parent_path());
                }

                int fd = open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
                if (fd < 0) {
                    throw std::runtime_error("Failed to open binary cache for writing: " + tmp_path);
                }

                auto checked_write = [&](const void* buf, size_t len) {
                    if (write(fd, buf, len) != static_cast<ssize_t>(len))
                        throw std::runtime_error("write failed in save_binary_cache");
                };

                // header: magic, version, num_points
                uint32_t magic = CACHE_MAGIC;
                uint32_t version = CACHE_VERSION;
                uint32_t np = _num_points;
                checked_write(&magic, sizeof(magic));
                checked_write(&version, sizeof(version));
                checked_write(&np, sizeof(np));

                // for each node: degree (uint32), then neighbor ids
                for (IdxType i = 0; i < _num_points; i++) {
                    uint32_t deg = neighbors[i].size();
                    checked_write(&deg, sizeof(deg));
                    if (deg > 0) {
                        checked_write(neighbors[i].data(), deg * sizeof(IdxType));
                    }
                }
                close(fd);

                // atomic rename
                if (rename(tmp_path.c_str(), cache_path.c_str()) != 0) {
                    unlink(tmp_path.c_str());
                    throw std::runtime_error("Failed to rename binary cache: " + tmp_path);
                }
            }

            void load_binary_cache(const std::string& cache_path) {
                int fd = open(cache_path.c_str(), O_RDONLY);
                if (fd < 0) {
                    throw std::runtime_error("Failed to open binary cache: " + cache_path);
                }
                struct stat st;
                if (fstat(fd, &st) < 0) {
                    close(fd);
                    throw std::runtime_error("Failed to stat binary cache: " + cache_path);
                }
                size_t file_size = st.st_size;
                if (file_size < 3 * sizeof(uint32_t)) {
                    close(fd);
                    throw std::runtime_error("Binary cache too small: " + cache_path);
                }

                const char* data = static_cast<const char*>(
                    mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
                if (data == MAP_FAILED) {
                    close(fd);
                    throw std::runtime_error("Failed to mmap binary cache: " + cache_path);
                }
                close(fd);

                const char* p = data;
                const char* end = data + file_size;

                // validate header
                uint32_t magic, version, np;
                memcpy(&magic, p, sizeof(magic)); p += sizeof(magic);
                memcpy(&version, p, sizeof(version)); p += sizeof(version);
                memcpy(&np, p, sizeof(np)); p += sizeof(np);

                if (magic != CACHE_MAGIC || version != CACHE_VERSION || np != _num_points) {
                    munmap(const_cast<char*>(data), file_size);
                    throw std::runtime_error("Binary cache header mismatch: " + cache_path);
                }

                for (IdxType i = 0; i < _num_points; i++) {
                    if (p + sizeof(uint32_t) > end) {
                        munmap(const_cast<char*>(data), file_size);
                        throw std::runtime_error("Binary cache truncated at node " + std::to_string(i));
                    }
                    uint32_t deg;
                    memcpy(&deg, p, sizeof(deg)); p += sizeof(deg);
                    if (p + deg * sizeof(IdxType) > end) {
                        munmap(const_cast<char*>(data), file_size);
                        throw std::runtime_error("Binary cache truncated at node " + std::to_string(i));
                    }
                    neighbors[i].resize(deg);
                    memcpy(neighbors[i].data(), p, deg * sizeof(IdxType));
                    p += deg * sizeof(IdxType);
                }

                munmap(const_cast<char*>(data), file_size);
            }

            void load_text(const std::string& filename) {
                int fd = open(filename.c_str(), O_RDONLY);
                if (fd < 0) {
                    throw std::runtime_error("Failed to open graph file: " + filename);
                }
                struct stat st;
                if (fstat(fd, &st) < 0) {
                    close(fd);
                    throw std::runtime_error("Failed to stat graph file: " + filename);
                }
                size_t file_size = st.st_size;
                if (file_size == 0) { close(fd); return; }

                const char* data = static_cast<const char*>(
                    mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
                if (data == MAP_FAILED) {
                    close(fd);
                    throw std::runtime_error("Failed to mmap graph file: " + filename);
                }
                close(fd);

                const char* p = data;
                const char* end = data + file_size;
                std::vector<IdxType> buf; // reusable scratch buffer

                while (p < end) {
                    // skip blank lines
                    while (p < end && (*p == '\n' || *p == '\r')) p++;
                    if (p >= end) break;

                    // parse node id
                    IdxType id;
                    auto [ptr, ec] = std::from_chars(p, end, id);
                    if (ec != std::errc()) {
                        munmap(const_cast<char*>(data), file_size);
                        throw std::runtime_error("Failed to parse node id in graph file: " + filename);
                    }
                    if (id >= _num_points) {
                        munmap(const_cast<char*>(data), file_size);
                        throw std::out_of_range("Node id " + std::to_string(id) +
                            " exceeds graph size " + std::to_string(_num_points));
                    }
                    p = ptr;

                    // count neighbors by scanning spaces on this line
                    const char* line_end = p;
                    size_t count = 0;
                    while (line_end < end && *line_end != '\n') {
                        if (*line_end == ' ') {
                            const char* next = line_end + 1;
                            if (next < end && *next >= '0' && *next <= '9')
                                count++;
                        }
                        line_end++;
                    }

                    // parse neighbors with pre-reserved buffer
                    buf.clear();
                    buf.reserve(count);
                    while (p < line_end) {
                        while (p < line_end && *p == ' ') p++;
                        if (p >= line_end) break;
                        IdxType nb;
                        auto [ptr2, ec2] = std::from_chars(p, line_end, nb);
                        if (ec2 != std::errc()) break;
                        buf.push_back(nb);
                        p = ptr2;
                    }

                    neighbors[id] = std::move(buf);
                    p = line_end;
                }

                munmap(const_cast<char*>(data), file_size);
            }
    };

    // we can also support graphs with arbitrary attributes on each edge.
    // each attribute should look like std::tuple<...>.
    template <typename T>
    class EdgeAttrGraph : public Graph {
        public:
            std::vector<T>* edge_attrs;
            std::vector<IdxType>* rev_neighbors;    // used in SeRF 2D, "reverse_nns"

            EdgeAttrGraph() = default;

            EdgeAttrGraph(IdxType num_points) {
                _num_points = num_points;
                neighbors = new std::vector<IdxType>[num_points];
                neighbor_locks = new std::mutex[num_points];
                edge_attrs = new std::vector<T>[num_points];
                rev_neighbors = nullptr;
            };

            EdgeAttrGraph(std::shared_ptr<EdgeAttrGraph<T>> graph, IdxType start, IdxType end) {
                neighbors = graph->neighbors + start;
                neighbor_locks = graph->neighbor_locks + start;
                edge_attrs = graph->edge_attrs + start;
                _num_points = end - start;
            };

            void save(const std::string& filename) override {
                std::ofstream out(filename);
                for (IdxType i = 0; i < _num_points; i++) {
                    out << i << " ";
                    for (IdxType j = 0; j < neighbors[i].size(); j++) {
                        out << neighbors[i][j] << " ";
                        PrintTuple(out, edge_attrs[i][j]);
                    }
                    out << std::endl;

                    if (rev_neighbors != nullptr) {
                        for (IdxType j = 0; j < rev_neighbors[i].size(); j++)
                            out << rev_neighbors[i][j] << " ";
                        out << std::endl;
                    }
                }
                out.close();
            }

            void load(const std::string& filename) override {
                int totn = 0, tot1 = 0, tot2 = 0;
                std::ifstream in(filename);
                std::string line;
                IdxType id, neighbor;
                T attr;
                while (std::getline(in, line)) {
                    ++totn;
                    std::istringstream iss(line);
                    iss >> id;
                    neighbors[id].clear();
                    edge_attrs[id].clear();
                    while (iss >> neighbor) {
                        ++tot1;
                        neighbors[id].push_back(neighbor);
                        ReadTuple(iss, attr);
                        edge_attrs[id].push_back(attr);
                    }

                    if (rev_neighbors != nullptr) {
                        std::getline(in, line);
                        std::istringstream iss(line);
                        rev_neighbors[id].clear();
                        while (iss >> neighbor) {
                            ++tot2;
                            rev_neighbors[id].push_back(neighbor);
                        }
                    }
                }
                in.close();
                std::cout << "total nodes: " << totn << ", total edges: " << tot1 << ", total rev edges: " << tot2 << std::endl;
            }

            size_t get_index_size() override {
                size_t index_size = 0;
                for (IdxType i = 0; i < _num_points; i++)
                    index_size += neighbors[i].size() * (sizeof(IdxType) + edge_attrs[i].size() * sizeof(T));
                return index_size;
            }

            void clean() override {
                delete[] neighbors;
                delete[] neighbor_locks;
                delete[] edge_attrs;
                if (rev_neighbors != nullptr)
                    delete[] rev_neighbors;
            }

            ~EdgeAttrGraph() = default;

        private:
        
            template <typename... Args>
            void PrintTuple(std::ostream &os, const std::tuple<Args...> &t) {
                std::apply([&os](const Args &...args) { ((os << args << " "), ...); }, t);
            }

            template <typename Tup>
            void ReadTuple(std::istream &is, Tup &t) {
                std::apply([&is](auto &&...args) { ((is >> args), ...); }, t);
            }
    };

    // see: `SegmentNeighbor1D`
    // 1D SeRF graph has attribute like (float, int).
    // (int: neighbor id, float: neighbor dist, int: when pruned)
    // vector<vector<SegmentNeighbor1D<dist_t>>> indexed_arr;

    // 2D SeRF graph has attribute like (u32, int, int).
    // (u32: "batch", int: start=-1, int: end=-2)
    // but the three attributes are shared among all edges starting from the same node.
    // vector<DirectedSegNeighbors> directed_indexed_arr;
    typedef EdgeAttrGraph<std::tuple<float, int>> SeRF1DGraph;
    typedef EdgeAttrGraph<std::tuple<unsigned, int, int>> SeRF2DGraph;
}

#endif // GRAPH_H