#include "baselines/search.h"
#include <faiss/impl/PathSeer.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <iostream>
#include <random>
#include <numeric>
struct Distance:faiss::DistanceComputer {
    std::vector<float> d;std::vector<int> trace;
    void set_query(const float*)override{}
    float operator()(faiss::idx_t id)override{trace.push_back(id);return d.at(id);}
    float symmetric_dis(faiss::idx_t,faiss::idx_t)override{return 0;}
};
struct Selector:faiss::IDSelector {
    std::vector<bool> mask;
    bool is_member(faiss::idx_t i)const override{return mask.at(i);}
    bool is_member(faiss::idx_t i,int)const override{return mask.at(i);}
};
int main(){try {
    constexpr int n=256,k=10;
    std::mt19937 rng(12);
    faiss::PathSeer p(8,16);p.entry_point=0;p.max_level=1;
    p.levels.assign(n,2);p.offsets.clear();p.offset_neighbor.clear();
    icde::Graph g;g.entry=0;g.max_level=1;g.M=8;g.maxM0=32;
    for(int id=0;id<n;++id) {
        p.offsets.push_back(p.neighbors.size());p.offset_neighbor.push_back(p.neighbor_boundaries.size());
        for(int l=0;l<2;++l) {
            std::vector<int> ids(n);std::iota(ids.begin(),ids.end(),0);std::shuffle(ids.begin(),ids.end(),rng);
            int cap=p.nb_neighbors(l), split=l?8:16;
            for(int j=0;j<cap;++j){p.neighbors.push_back(ids[j]);g.edges.push_back(ids[j]);}
            p.neighbor_boundaries.push_back(split);g.boundary.push_back(split);g.offset.push_back(g.edges.size());
        }
        g.node.push_back(g.boundary.size());
    }
    p.offsets.push_back(p.neighbors.size());
    Distance distance;distance.d.resize(n);Selector selector;selector.mask.resize(n);
    icde::Visited seen(n);
    int comparisons=0;
    for(int test=0;test<120;++test) {
        for(int i=0;i<n;++i) {distance.d[i]=float(rng()%1000000)*0.001f+i*0.0000001f;selector.mask[i]=test%6==0?false:test%6==1?true:(rng()%100<20);}
        icde::Costs costs{int64_t(10+test%9*100),int64_t(5+test%7*200)};
        p.profilier->dist_comp_cost_ns=costs.distance_ns;p.profilier->filter_cost_ns=costs.predicate_ns;
        int ef=10+test%5*16;
        faiss::SearchParametersPathSeer params;params.efSearch=ef;params.M_exp_search=0;params.sel=&selector;
        faiss::VisitedTable vt(n);
        std::vector<float> d(k,std::numeric_limits<float>::infinity());std::vector<faiss::idx_t> ids(k,-1);
        distance.trace.clear();p.search(distance,k,ids.data(),d.data(),vt,0,&params);
        auto trace=distance.trace;distance.trace.clear();
        icde::Access a;a.distance=[&](int id){return distance(id);};a.predicate=[&](int id){return selector.is_member(id);};
        auto actual=icde::pathseer(g,a,seen,k,ef,costs);
        std::vector<icde::Hit> expected;
        for(int i=0;i<k;++i)if(ids[i]>=0)expected.push_back({d[i],ids[i]});
        std::sort(expected.begin(),expected.end());std::sort(actual.begin(),actual.end());
        if(actual!=expected || trace!=distance.trace) {
            std::cerr<<"PathSeer differential mismatch test="<<test<<" results="<<(actual==expected)<<" trace="<<(trace==distance.trace)<<"\n";return 1;
        }
        ++comparisons;
    }
    // Compare bounded-frontier operations independently, including equal distances.
    for(int round=0;round<20;++round) {
        faiss::PathSeer::MinimaxHeap reference(20);icde::Minimax local(20);
        for(int i=0;i<200;++i) {
            if(reference.size()>0 && rng()%3==0) {
                float d;int id=reference.pop_min(&d);auto actual=local.pop();
                if(actual!=icde::Hit(d,id))throw std::runtime_error("minimax pop differs");
            } else {float d=rng()%20;reference.push(i,d);local.push(i,d);}
            if(reference.size()!=local.size() || reference.count_below(10)!=local.below(10))throw std::runtime_error("minimax bookkeeping differs");
        }
    }
    std::cout<<comparisons<<" PathSeer upstream differential searches passed; bounded-queue differential passed\n";
} catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
