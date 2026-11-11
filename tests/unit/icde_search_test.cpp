#include "baselines/index_io.h"
#include <iostream>
#include <set>
int main(){try {
    constexpr int n=32,k=5;
    icde::Graph graph;graph.entry=0;graph.M=16;graph.maxM0=32;
    icde::Clusters clusters;clusters.assignment.assign(n,0);
    for(int i=0;i<n;++i) {
        for(int j=0;j<n;++j)if(j!=i)graph.edges.push_back(j);
        graph.node.push_back(i+1);graph.offset.push_back(graph.edges.size());graph.boundary.push_back(16);
        clusters.ids.push_back(i);
    }
    clusters.offset.push_back(n);icde::validate(graph);
    icde::Visited seen(n);
    for(int mode=0;mode<4;++mode) {
        icde::Access access;
        access.distance=[](int id){return float((31-id)*(31-id));};
        access.predicate=[&](int id){return mode==0?false:mode==1?true:mode==2?id%2==0:id==31;};
        access.scan=[&](const auto& visit){for(int id=0;id<n;++id)if(access.predicate(id)){std::function<float()> d=[&]{return access.distance(id);};visit(id,d);}};
        std::vector<icde::Hit> truth;
        for(int id=0;id<n;++id)if(access.predicate(id))truth.push_back({access.distance(id),id});
        std::sort(truth.begin(),truth.end());if(truth.size()>k)truth.resize(k);
        auto cgif=icde::cgif(graph,clusters,access,seen,k,n);
        auto pathseer=icde::pathseer(graph,access,seen,k,n,{100,1});
        if(cgif!=truth || pathseer!=truth)throw std::runtime_error("complete-graph filtered truth mismatch");
        auto favor=icde::favor(graph,access,seen,k,n,1);
        auto [step,denominator]=icde::favor_sampling(n);
        std::vector<uint8_t> cached;
        for(size_t id=0;id<n;id+=step)cached.push_back(access.predicate(id));
        int direct_checks=0,cached_checks=0;
        auto original=access.predicate;
        access.predicate=[&](int id){++direct_checks;return original(id);};
        auto direct=icde::favor(graph,access,seen,k,n,1);
        access.predicate=[&](int id){++cached_checks;return original(id);};
        auto ablation=icde::favor(graph,access,seen,k,n,1,&cached);
        access.predicate=original;
        if(direct!=ablation || direct_checks-cached_checks!=int(cached.size()))
            throw std::runtime_error("FAVOR cache changed results or nonsampling predicate checks");
        std::set<int> ids;
        for(auto x:favor)if(!access.predicate(x.second)||!ids.insert(x.second).second)throw std::runtime_error("FAVOR invalid/duplicate result");
        if(mode==0 && !favor.empty())throw std::runtime_error("FAVOR empty predicate returned results");
    }
    bool rejected=false;icde::Access access;
    try {icde::favor(graph,access,seen,k,10,std::numeric_limits<float>::quiet_NaN());}catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)throw std::runtime_error("FAVOR accepted missing calibration");
    std::cout<<"Empty/all/sparse filters, fewer-than-K results and required calibration passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
