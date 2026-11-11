#pragma once
#include "search.h"
#include <fstream>
#include <string>
namespace icde {
template<class T> void write(std::ostream& out,const T& v) {
    out.write(reinterpret_cast<const char*>(&v),sizeof(v));
    if(!out)throw std::runtime_error("index write failed");
}
template<class T> void write_array(std::ostream& out,const std::vector<T>& v) {
    write(out,uint64_t(v.size()));out.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));
    if(!out)throw std::runtime_error("index write failed");
}
template<class T> void read(std::istream& in,T& v) {
    in.read(reinterpret_cast<char*>(&v),sizeof(v));if(!in)throw std::runtime_error("truncated index");
}
template<class T> void read_array(std::istream& in,std::vector<T>& v) {
    uint64_t n;read(in,n);auto pos=in.tellg();in.seekg(0,std::ios::end);auto end=in.tellg();in.seekg(pos);
    if(n>uint64_t(end-pos)/sizeof(T))throw std::runtime_error("invalid index array length");
    v.resize(n);in.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));if(!in)throw std::runtime_error("truncated array");
}
inline void save_graph(std::ostream& out,const Graph& g) {
    write(out,uint64_t(0x3147524545534650ULL)); // PFSEERG1, version 1
    write(out,g.entry);write(out,g.max_level);write(out,g.M);write(out,g.maxM0);
    write_array(out,g.node);write_array(out,g.offset);write_array(out,g.boundary);write_array(out,g.edges);
}
inline void validate(const Graph& g) {
    if(g.node.size()<2 || g.entry<0 || size_t(g.entry)>=g.size() || g.max_level<0 ||
       g.M<=0 || g.maxM0<=0 || g.node.front()!=0 || g.offset.empty() || g.offset.front()!=0 ||
       g.node.back()!=g.boundary.size() || g.offset.size()!=g.boundary.size()+1 ||
       g.offset.back()!=g.edges.size())throw std::runtime_error("invalid graph header");
    for(size_t i=0;i<g.size();++i)if(g.node[i]>=g.node[i+1] || g.node[i+1]>g.boundary.size())throw std::runtime_error("invalid levels");
    if(g.node[g.entry+1]-g.node[g.entry]<=size_t(g.max_level))throw std::runtime_error("invalid entry level");
    for(size_t r=0;r<g.boundary.size();++r)
        if(g.offset[r]>g.offset[r+1] || g.offset[r+1]>g.edges.size() || g.boundary[r]<0 ||
           uint64_t(g.boundary[r])>g.offset[r+1]-g.offset[r])throw std::runtime_error("invalid adjacency boundary");
    for(size_t i=0;i<g.size();++i)for(size_t r=g.node[i];r<g.node[i+1];++r)
        for(uint64_t j=g.offset[r];j<g.offset[r+1];++j) {
            Id id=g.edges[j];
            if(id<-1 || (id>=0 && (size_t(id)>=g.size() || g.node[id+1]-g.node[id]<=r-g.node[i])))throw std::runtime_error("invalid edge/level");
        }
}
inline Graph load_graph(const std::string& path) {
    std::ifstream in(path,std::ios::binary);uint64_t magic;read(in,magic);
    if(magic!=0x3147524545534650ULL)throw std::runtime_error("not a PathSeer search export");
    Graph g;read(in,g.entry);read(in,g.max_level);read(in,g.M);read(in,g.maxM0);
    read_array(in,g.node);read_array(in,g.offset);read_array(in,g.boundary);read_array(in,g.edges);
    if(in.peek()!=EOF)throw std::runtime_error("trailing index data");validate(g);return g;
}
inline void save_clusters(std::ostream& out,const Clusters& c) {
    write(out,uint64_t(0x3152464947434650ULL));write_array(out,c.offset);write_array(out,c.ids);write_array(out,c.assignment);
}
inline Clusters load_clusters(const std::string& path,size_t n) {
    std::ifstream in(path,std::ios::binary);uint64_t magic;read(in,magic);
    if(magic!=0x3152464947434650ULL)throw std::runtime_error("not a CGIF cluster export");
    Clusters c;read_array(in,c.offset);read_array(in,c.ids);read_array(in,c.assignment);
    if(c.offset.size()<2 || c.offset.front()!=0 || c.offset.back()!=n || c.ids.size()!=n || c.assignment.size()!=n || in.peek()!=EOF)
        throw std::runtime_error("cluster shape mismatch");
    std::vector<bool> seen(n,false);
    for(size_t j=0;j+1<c.offset.size();++j) {
        if(c.offset[j]>c.offset[j+1] || c.offset[j+1]>n)throw std::runtime_error("invalid cluster offsets");
        for(size_t p=c.offset[j];p<c.offset[j+1];++p) {
            Id id=c.ids[p];
            if(id<0 || size_t(id)>=n || seen[id] || c.assignment[id]!=int(j))throw std::runtime_error("invalid cluster membership");
            seen[id]=true;
        }
    }
    return c;
}
}
