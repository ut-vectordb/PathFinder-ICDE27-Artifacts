#include <faiss/IndexPathSeer.h>
#include <faiss/index_io.h>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
int main(int argc,char**argv){
 try {
  if(argc!=2)throw std::runtime_error("usage: count_pathseer_edges index.faiss");
  std::unique_ptr<faiss::Index> base(faiss::read_index(argv[1]));
  auto index=dynamic_cast<faiss::IndexPathSeer*>(base.get());
  if(!index)throw std::runtime_error("Not a PathSeer index");
  const auto&g=index->pathseer;
  uint64_t bottom=0,upper=0,dist=0,expansion=0;
  for(size_t id=0;id<g.levels.size();++id){
   for(int level=0;level<g.levels[id];++level){
    size_t begin,end;g.neighbor_range(id,level,&begin,&end);
    int boundary=g.get_neighbor_dist_boundary(id,level);
    for(size_t pos=begin;pos<end;++pos){
     auto neighbor=g.neighbors[pos];
     if(neighbor<0)continue;
     if(neighbor>=index->ntotal)throw std::runtime_error("Neighbor out of bounds");
     if(level==0)++bottom;else ++upper;
     if(pos-begin<size_t(boundary))++dist;else ++expansion;
    }
   }
  }
  std::cout<<"{\n  \"nodes\": "<<index->ntotal<<",\n  \"bottom_edges\": "<<bottom<<",\n  \"upper_edges\": "<<upper<<",\n  \"distance_oriented_edges\": "<<dist<<",\n  \"expansion_edges\": "<<expansion<<",\n  \"total_edges\": "<<bottom+upper<<",\n  \"allocated_slots\": "<<g.neighbors.size()<<"\n}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
