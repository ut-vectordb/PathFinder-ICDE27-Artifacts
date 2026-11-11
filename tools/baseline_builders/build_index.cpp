// Build-only input adapter. Index construction calls mirror the upstream
// PathSeer benchmark and CGIF apps/build_ivf_index.cc, using raw L2.
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#ifdef BUILD_PATHSEER
#include <faiss/IndexPathSeer.h>
#else
#include <faiss/IndexIVFFlat.h>
#endif
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  try {
#ifdef BUILD_PATHSEER
    if (argc != 9) throw std::runtime_error("usage: pathseer_build base.bin output.faiss threads M M_exp efConstruction efConstructionExp limit(0=all)");
#else
    if (argc != 6) throw std::runtime_error("usage: cgif_build base.bin output.faiss threads nlist limit(0=all)");
#endif
    const std::string output = argv[2], partial = output + ".partial";
    if (std::filesystem::exists(output) || std::filesystem::exists(partial))
      throw std::runtime_error("Refusing to overwrite an existing output or partial index");
    const int threads = std::stoi(argv[3]);
    if (threads <= 0) throw std::runtime_error("threads must be positive");
    omp_set_num_threads(threads);
    uint32_t total = 0, dim = 0;
    std::ifstream input(argv[1], std::ios::binary);
    input.read(reinterpret_cast<char*>(&total), 4);
    input.read(reinterpret_cast<char*>(&dim), 4);
    if (!input || !total || !dim || std::filesystem::file_size(argv[1]) != 8ULL + uint64_t(total)*dim*4)
      throw std::runtime_error("Invalid vector header/file size");
    const auto limit = std::stoull(argv[argc-1]);
    const uint32_t n = limit ? std::min<uint64_t>(limit, total) : total;
    std::vector<float> vectors(uint64_t(n)*dim);
    input.read(reinterpret_cast<char*>(vectors.data()), vectors.size()*sizeof(float));
    if (!input) throw std::runtime_error("Short vector read");
    for (float x : vectors) if (!std::isfinite(x)) throw std::runtime_error("Non-finite vector value");
    std::cout << "BUILD_START n=" << n << " dim=" << dim << " threads=" << threads << " metric=L2 normalization=none\n" << std::flush;
    const auto start = std::chrono::steady_clock::now();
    std::unique_ptr<faiss::Index> index;
#ifdef BUILD_PATHSEER
    const int m=std::stoi(argv[4]), mexp=std::stoi(argv[5]);
    const int ef=std::stoi(argv[6]), efexp=std::stoi(argv[7]);
    if (m <= 1 || mexp < m || ef < m || efexp < mexp) throw std::runtime_error("Invalid PathSeer build parameters");
    auto p = new faiss::IndexPathSeerFlat(dim, m, mexp, faiss::METRIC_L2);
    index.reset(p);
    p->pathseer.efConstruction=ef;
    p->pathseer.efConstructionExp=efexp;
    p->set_using_only_expanding_idx(false);
    p->set_using_optimized_building(true);
    p->verbose=true;
    std::cout << "M=" << m << " M_exp=" << mexp << " efConstruction=" << ef << " efConstructionExp=" << efexp << std::endl;
    p->add(n, vectors.data());
#else
    const int clusters=std::stoi(argv[4]);
    if (clusters <= 0 || clusters > int(n)) throw std::runtime_error("Invalid cluster count");
    auto quantizer = std::make_unique<faiss::IndexFlatL2>(dim);
    auto p = new faiss::IndexIVFFlat(quantizer.get(), dim, clusters, faiss::METRIC_L2);
    p->own_fields=true; quantizer.release(); index.reset(p);
    p->verbose=true; p->cp.verbose=true;
    std::cout << "nlist=" << clusters << " seed=" << p->cp.seed << " training_iterations=" << p->cp.niter << " max_points_per_centroid=" << p->cp.max_points_per_centroid << std::endl;
    p->train(n, vectors.data());
    p->add(n, vectors.data());
#endif
    if (index->ntotal != n || index->d != int(dim)) throw std::runtime_error("Build shape mismatch");
    std::cout << "BUILD_SECONDS=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count() << std::endl;
    std::vector<float> first(vectors.begin(), vectors.begin()+dim);
    std::vector<float>().swap(vectors);
    faiss::write_index(index.get(), partial.c_str());
    index.reset();
    index.reset(faiss::read_index(partial.c_str()));
    if (index->ntotal != n || index->d != int(dim) || index->metric_type != faiss::METRIC_L2)
      throw std::runtime_error("Reloaded index metadata mismatch");
#ifdef BUILD_PATHSEER
    auto loaded=dynamic_cast<faiss::IndexPathSeerFlat*>(index.get());
    if (!loaded || loaded->pathseer.levels.size()!=n || loaded->pathseer.M!=m || loaded->pathseer.M_exp!=mexp)
      throw std::runtime_error("Reloaded PathSeer graph mismatch");
    std::vector<float> check(dim); loaded->reconstruct(0,check.data());
    if (check!=first) throw std::runtime_error("Vector IDs changed");
#else
    auto loaded=dynamic_cast<faiss::IndexIVFFlat*>(index.get());
    if (!loaded || loaded->nlist!=size_t(clusters)) throw std::runtime_error("Reloaded IVF mismatch");
    std::vector<uint8_t> seen(n,0); uint64_t count=0;
    for (size_t c=0;c<loaded->nlist;++c) {
      const auto ids=loaded->invlists->get_ids(c);
      for(size_t j=0;j<loaded->invlists->list_size(c);++j) {
        if(ids[j]<0 || ids[j]>=n || seen[ids[j]]++) throw std::runtime_error("Cluster ID coverage invalid");
        ++count;
      }
      loaded->invlists->release_ids(c,ids);
    }
    if(count!=n) throw std::runtime_error("Missing vectors in clusters");
#endif
    std::filesystem::rename(partial,output);
    std::cout << "BUILD_COMPLETE verified_reload=true output=" << output << std::endl;
    return 0;
  } catch(const std::exception& e) {
    std::cerr << "BUILD_FAILED: " << e.what() << std::endl;
    return 1;
  }
}
