#include "baselines/search.h"
#include <favor.h>
#include <iostream>
#include <random>
#include <array>
int main(){try {
    constexpr int n=256,dim=4,k=10;
    hnswlib::L2Space space(dim);favor::FAVOR<float> reference(&space,n,8,32,1);
    std::mt19937 rng(1337);std::vector<std::array<float,dim>> vectors(n);
    for(int i=0;i<n;++i) {
        for(auto& x:vectors[i])x=rng()%100;
        float attr=i;reference.addPoint(vectors[i].data(),i,&attr);
    }
    reference.delta_d=0.5;
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
    icde::Visited visited(n);int scans=0;
    for(int trial=0;trial<100;++trial) {
        int threshold=trial%10==0?0:trial%10==1?1:30+trial*2;
        std::array<float,dim> query;for(auto& x:query)x=rng()%100;
        int ef=10+trial%4*20;reference.setEf(ef);
        std::vector<FilterConditionWithId> conditions{{0,"<",{float(threshold)}}};
        auto expected=reference.searchKnn(query.data(),k,conditions);
        icde::Access a;
        a.distance=[&](int id){float d=0;for(int j=0;j<dim;++j){float x=query[j]-vectors[id][j];d+=x*x;}return d;};
        a.predicate=[&](int id){return id<threshold;};
        a.scan=[&](const auto& visit){++scans;for(int id=0;id<n;++id)if(a.predicate(id)){std::function<float()> d=[&]{return a.distance(id);};visit(id,d);}};
        auto actual=icde::favor(g,a,visited,k,ef,0.5f);
        std::vector<float> expected_d,actual_d;
        while(!expected.empty()){auto x=expected.top();expected.pop();if(x.second!=size_t(-1))expected_d.push_back(x.first);}
        for(auto x:actual){if(!a.predicate(x.second))throw std::runtime_error("invalid result");actual_d.push_back(x.first);}
        std::sort(expected_d.begin(),expected_d.end());std::sort(actual_d.begin(),actual_d.end());
        if(expected_d!=actual_d)throw std::runtime_error("FAVOR differential mismatch at "+std::to_string(trial));
    }
    if(scans!=20)throw std::runtime_error("FAVOR scan branch not exercised as expected");
    std::cout<<"100 FAVOR upstream differential searches passed (20 low-selectivity scans)\n";
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
