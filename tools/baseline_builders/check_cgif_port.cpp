#include "baselines/search.h"
#include <ivf_connect.h>
#include <array>
#include <random>
#include <iostream>
int main(){try {
    constexpr int n=512,dim=4,k=10;
    std::mt19937 rng(1338);std::vector<std::array<float,dim>> vectors(n);
    std::vector<std::vector<uint32_t>> lists(16);
    for(int i=0;i<n;++i)lists[i%16].push_back(i);
    auto ivf=std::make_shared<IVFIndex>(lists,n);
    hnswlib::L2Space space(dim);IVFConnectHNSW<float> reference(ivf,&space,n,8,32);
    for(int i=0;i<n;++i) {for(auto& x:vectors[i])x=rng()%100;reference.addPoint(vectors[i].data(),i);}
    icde::Graph g;g.entry=reference.enterpoint_node_;g.max_level=reference.maxlevel_;g.M=8;g.maxM0=16;
    for(int id=0;id<n;++id) {
        for(int l=0;l<=reference.element_levels_[id];++l) {
            auto* ptr=l?reference.get_linklist(id,l):reference.get_linklist0(id);
            int count=reference.getListCount(ptr);auto* ids=reinterpret_cast<hnswlib::tableint*>(ptr+1);
            for(int j=0;j<count;++j)g.edges.push_back(ids[j]);
            g.offset.push_back(g.edges.size());g.boundary.push_back(count);
        }
        g.node.push_back(g.boundary.size());
    }
    icde::Clusters clusters;clusters.assignment.resize(n);
    for(size_t c=0;c<lists.size();++c) {
        for(int id:lists[c]){clusters.ids.push_back(id);clusters.assignment[id]=c;}
        clusters.offset.push_back(clusters.ids.size());
    }
    icde::Visited visited(n);
    for(int trial=0;trial<100;++trial) {
        int threshold=100+trial*3;
        std::array<float,dim> query;for(auto& x:query)x=rng()%100;
        int ef=10+trial%3*10;reference.setEf(ef);
        std::vector<uint8_t> bits(n/8,0);
        for(int i=threshold;i<n;++i)bits[i>>3]|=1<<(i&7);
        BitsetView mask(bits.data(),n);
        auto expected=reference.searchKnn(query.data(),k,mask);
        icde::Access a;
        a.distance=[&](int id){float d=0;for(int j=0;j<dim;++j){float x=query[j]-vectors[id][j];d+=x*x;}return d;};
        a.predicate=[&](int id){return mask.is_member(id);};
        auto actual=icde::cgif(g,clusters,a,visited,k,ef);
        std::vector<float> expected_d,actual_d;
        while(!expected.empty()){expected_d.push_back(expected.top().first);expected.pop();}
        for(auto x:actual){if(!a.predicate(x.second))throw std::runtime_error("invalid result");actual_d.push_back(x.first);}
        std::sort(expected_d.begin(),expected_d.end());std::sort(actual_d.begin(),actual_d.end());
        if(expected_d!=actual_d)throw std::runtime_error("CGIF differential mismatch at "+std::to_string(trial));
    }
    std::cout<<"100 CGIF upstream differential searches passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
