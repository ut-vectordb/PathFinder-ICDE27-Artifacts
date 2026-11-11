// Format conversion only: retain graph order, layers and boundaries / IVF ID order.
#include "baselines/index_io.h"
#include <faiss/index_io.h>
#ifdef BUILD_PATHSEER
#include <faiss/IndexPathSeer.h>
#else
#include <faiss/IndexIVFFlat.h>
#endif
#include <filesystem>
#include <memory>
#include <iostream>
int main(int argc,char** argv) {try {
    if(argc!=3)throw std::runtime_error("usage: export_search input.faiss output.search");
    std::string path=argv[2];
    if(std::filesystem::exists(path)||std::filesystem::exists(path+".partial"))throw std::runtime_error("output exists");
    std::unique_ptr<faiss::Index> index(faiss::read_index(argv[1]));
    std::ofstream out(path+".partial",std::ios::binary);
#ifdef BUILD_PATHSEER
    auto* typed=dynamic_cast<faiss::IndexPathSeer*>(index.get());
    if(!typed || typed->metric_type!=faiss::METRIC_L2)throw std::runtime_error("expected L2 PathSeer");
    auto& p=typed->pathseer;icde::Graph g;
    g.entry=p.entry_point;g.max_level=p.max_level;g.M=p.M;g.maxM0=p.M_exp*2;
    g.edges.reserve(p.neighbors.size());
    for(size_t id=0;id<p.levels.size();++id) {
        for(int l=0;l<p.levels[id];++l) {
            size_t b,e;p.neighbor_range(id,l,&b,&e);
            for(size_t j=b;j<e;++j)g.edges.push_back(p.neighbors[j]);
            g.offset.push_back(g.edges.size());g.boundary.push_back(p.get_neighbor_dist_boundary(id,l));
        }
        g.node.push_back(g.boundary.size());
    }
    icde::validate(g);icde::save_graph(out,g);
#else
    auto* p=dynamic_cast<faiss::IndexIVFFlat*>(index.get());
    if(!p || p->metric_type!=faiss::METRIC_L2)throw std::runtime_error("expected L2 IVF Flat");
    icde::Clusters c;c.assignment.resize(p->ntotal,-1);
    for(size_t list=0;list<p->nlist;++list) {
        auto ids=p->invlists->get_ids(list);
        for(size_t j=0;j<p->invlists->list_size(list);++j) {
            auto id=ids[j];
            if(id<0 || id>=p->ntotal || c.assignment[id]!=-1)throw std::runtime_error("invalid IVF ID");
            c.ids.push_back(id);c.assignment[id]=list;
        }
        p->invlists->release_ids(list,ids);c.offset.push_back(c.ids.size());
    }
    if(c.ids.size()!=size_t(p->ntotal))throw std::runtime_error("missing IVF IDs");
    icde::save_clusters(out,c);
#endif
    out.close();if(!out)throw std::runtime_error("write failed");
    std::filesystem::rename(path+".partial",path);std::cout<<"Exported "<<path<<"\n";
} catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
