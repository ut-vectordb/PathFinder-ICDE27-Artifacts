#include "icde_search.h"
#include "baselines/index_io.h"
#include <algorithm>
#include <chrono>
#include <sstream>
#include <numeric>
#include <random>
#include <memory>
#include <atomic>
#include <iomanip>

namespace {
icde::Graph stock_hnsw(const std::string& prefix,size_t n) {
    ANNS::Graph bottom(n);bottom.load(prefix+"attr_equidepth_root.txt");
    ANNS::HNSWUpperLayer upper;upper.load(prefix+"attr_equidepth_root_upper.txt");
    if(upper.element_levels_.size()!=n)throw std::runtime_error("HNSW/table size mismatch");
    icde::Graph g;g.entry=upper.enterpoint_node_;g.max_level=upper.maxlevel_;
    g.M=upper.n_links_per_element_-1;g.maxM0=2*g.M;
    for(size_t id=0;id<n;++id) {
        const auto& ns=bottom.neighbors[id];
        g.edges.insert(g.edges.end(),ns.begin(),ns.end());
        g.boundary.push_back(ns.size());g.offset.push_back(g.edges.size());
        for(int l=1;l<=upper.element_levels_[id];++l) {
            const auto& data=upper.edgelist.at(id);
            size_t start=(l-1)*upper.n_links_per_element_;int count=data.at(start);
            if(count<0 || count>g.M || start+1+count>data.size())throw std::runtime_error("invalid HNSW upper adjacency");
            g.edges.insert(g.edges.end(),data.begin()+start+1,data.begin()+start+1+count);
            g.boundary.push_back(count);g.offset.push_back(g.edges.size());
        }
        g.node.push_back(g.boundary.size());
    }
    icde::validate(g);return g;
}
// Approximates insertion-time calibration: replay unfiltered efConstruction=200 searches
// on the completed HNSW, excluding the sampled vector from the result pool.
void calibrate_favor(TableStorage& table,const icde::Graph& graph) {
    const int breadth=200;
    if(table.nb<=size_t(breadth))throw std::runtime_error("FAVOR calibration needs >200 vectors");
    std::vector<int> ids(table.nb);std::iota(ids.begin(),ids.end(),0);
    std::mt19937 rng(20261102);std::shuffle(ids.begin(),ids.end(),rng);
    ids.resize(std::min<size_t>(10000,ids.size()));
    std::vector<std::unique_ptr<icde::Visited>> visits;
    for(unsigned t=0;t<num_threads;++t)visits.emplace_back(new icde::Visited(table.nb));
    std::vector<double> terms(ids.size());
    int full=0;
    #pragma omp parallel for schedule(dynamic,1) reduction(+:full)
    for(size_t s=0;s<ids.size();++s) {
        int sample=ids[s];auto* query=table.get_vector(table.lookup(sample));
        icde::Access a;a.distance=[&](int id){return distance_handler->compute(query,table.get_vector(table.lookup(id)),table.n_dim);};
        auto entry=icde::descend(graph,a);icde::Heap queue,top;
        auto& seen=*visits[omp_get_thread_num()];seen.clear();seen.set(entry.second);
        queue.push({-entry.first,entry.second});if(entry.second!=sample)top.push(entry);
        while(!queue.empty()) {
            auto cur=queue.top();queue.pop();
            if(top.size()==breadth && -cur.first>top.top().first)break;
            for(int id:graph.neighbors(cur.second)) {
                if(seen.has(id))continue;seen.set(id);float d=a.distance(id);
                if(top.size()<breadth || d<top.top().first) {
                    queue.push({-d,id});
                    if(id!=sample) {top.push({d,id});if(top.size()>breadth)top.pop();}
                }
            }
        }
        if(top.size()==breadth) {
            auto hits=icde::finish(top,breadth);
            terms[s]=5.0*(hits.back().first-hits[9].first)/(breadth-10);++full;
        }
    }
    double mean=std::accumulate(terms.begin(),terms.end(),0.0)/terms.size();
    double variance=0;for(double t:terms)variance+=(t-mean)*(t-mean);variance/=terms.size();
    std::cout<<std::setprecision(10)<<"FAVOR calibration approximation: samples="<<terms.size()<<" complete_pools="<<full
             <<" efConstruction="<<breadth<<" seed=20261102 mean="<<mean<<" standard_error="<<std::sqrt(variance/terms.size())<<"\n";
    std::cout<<"FAVOR_DELTA_D="<<float(mean)<<"\n";
}

using Clock=std::chrono::steady_clock;
template<class Pred>
icde::Costs profile(TableStorage& table,const Pred& predicate) {
    if(table.nb<2)throw std::runtime_error("PathSeer profiling needs at least two vectors");
    // Same 1% shuffled-ID microbenchmark as upstream, with valid IDs, RAII and
    // an observable predicate sink. Profile once per workload, outside QPS.
    std::vector<int> ids(table.nb-1);std::iota(ids.begin(),ids.end(),1);
    std::mt19937 rng(20261102);std::shuffle(ids.begin(),ids.end(),rng);
    size_t count=std::max<size_t>(1,table.nb/100);count=std::min(count,ids.size());
    auto* query=table.get_vector(table.lookup(0));
    auto start=Clock::now();
    for(size_t i=0;i<count;++i) {
        volatile float sink=distance_handler->compute(query,table.get_vector(table.lookup(ids[i])),table.n_dim);(void)sink;
    }
    auto dist_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count()/count;
    ids.resize(table.nb);std::iota(ids.begin(),ids.end(),0);std::shuffle(ids.begin(),ids.end(),rng);
    start=Clock::now();size_t matches=0;
    for(size_t i=0;i<count;++i)matches+=predicate.check(table.get_attr(table.lookup(ids[i])));
    auto filter_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count()/count;
    std::cerr<<"PathSeer profile: samples="<<count<<" distance_ns="<<dist_ns<<" predicate_ns="<<filter_ns<<" matches="<<matches<<" seed=20261102\n";
    return {std::max<int64_t>(1,dist_ns),std::max<int64_t>(1,filter_ns)};
}
}
namespace {
template<class Pred>
void run_icde_one(TableStorage& table,std::shared_ptr<ANNS::IStorage> queries,
                  const std::vector<Pred>& predicates,int k,const std::vector<int>& efs,
                  std::pair<ANNS::IdxType,float>* gt,const IcdeSearchOptions& options,
                  const std::string& algorithm,const icde::Graph& graph) {
    if(algorithm=="favor" && (!std::isfinite(options.favor_delta_d)||options.favor_delta_d<0))
        throw std::invalid_argument("FAVOR calibration pending: supply --favor_delta_d explicitly");
    if(graph.size()!=table.nb)throw std::runtime_error("graph/table size mismatch");
    if(algorithm=="favor-calibrate") {calibrate_favor(table,graph);return;}
    icde::Clusters clusters;
    if(algorithm=="cgif")clusters=icde::load_clusters(options.cgif_clusters,table.nb);
    int nq=queries->get_num_points();
    if(nq<=0 || predicates.size()!=size_t(nq))throw std::runtime_error("query/predicate size mismatch");
    std::vector<std::vector<uint8_t>> sampling_cache;
    if(options.favor_cache_sampling) {
        if(algorithm!="favor")throw std::invalid_argument("favor_cache_sampling requires FAVOR");
        auto [step,denominator]=icde::favor_sampling(table.nb);
        size_t count=(table.nb+step-1)/step;
        sampling_cache.resize(nq,std::vector<uint8_t>(count));
        int scan_decisions=0;
        auto start=Clock::now();
        #pragma omp parallel for schedule(dynamic,1) num_threads(num_threads) reduction(+:scan_decisions)
        for(int q=0;q<nq;++q) {
            size_t sample=0,passed=0;
            for(size_t id=0;id<table.nb;id+=step,++sample) {
                auto value=predicates[q].check(table.get_attr(table.lookup(id)));
                sampling_cache[q][sample]=value;passed+=value;
            }
            scan_decisions+=(float(passed)/denominator<=0.01f);
        }
        std::cerr<<"FAVOR sampling cache: queries="<<nq<<" samples_per_query="<<count
                 <<" denominator="<<denominator<<" scan_decisions="<<scan_decisions
                 <<" precompute_seconds="<<std::chrono::duration<double>(Clock::now()-start).count()
                 <<" excluded_from_query_timing=true bytes="<<size_t(nq)*count<<"\n";
    }
    icde::Costs costs;
    if(algorithm=="pathseer")costs=profile(table,predicates[0]);
    std::vector<std::unique_ptr<icde::Visited>> visited;
    for(unsigned t=0;t<num_threads;++t)visited.emplace_back(new icde::Visited(table.nb));
    std::vector<std::pair<ANNS::IdxType,float>> results(size_t(nq)*k);
    std::vector<double> all_cmps,all_qps,all_recall;
    std::cerr<<"ICDE search="<<algorithm
#ifdef BYPASS_B_TREE
             <<" predicate=direct-arrays"
#else
             <<" predicate=TableStorage/B-tree"
#endif
             <<" scan=true cache=per-query\n";
    if(algorithm=="favor")std::cerr<<"favor_delta_d="<<options.favor_delta_d<<" favor_oor_results="<<options.favor_oor_results<<"\n";
    omp_set_num_threads(num_threads);
    // Largest ef first (as jul14_demo does with flag2 != -1): small-ef points measured
    // on a cold cache understate QPS. Output arrays follow this run order.
    std::vector<int> order(efs);std::sort(order.rbegin(),order.rend());
    for(int ef:order) {
        if(ef<=0)throw std::runtime_error("ef must be positive");
        uint64_t comparisons=0,checks=0;
        auto start=Clock::now();
        #pragma omp parallel for schedule(dynamic,1) reduction(+:comparisons,checks)
        for(int q=0;q<nq;++q) {
            auto* query=queries->get_vector(q);
            icde::Access access;
            access.distance=[&](int id){++comparisons;return distance_handler->compute(query,table.get_vector(table.lookup(id)),table.n_dim);};
            access.predicate=[&](int id){++checks;return predicates[q].check(table.get_attr(table.lookup(id)));};
            access.scan=[&](const auto& visit) {
#ifdef BYPASS_B_TREE
                for (size_t id = 0; id < table.nb; ++id) {
                    auto* record = table.lookup(id);
                    ++checks;
                    if (predicates[q].check(table.get_attr(record))) {
                        std::function<float()> dist = [&] {
                            ++comparisons;
                            return distance_handler->compute(query, table.get_vector(record), table.n_dim);
                        };
                        visit(id, dist);
                    }
                }
#else
                // Bounded batches avoid materializing the whole table per query.
                // Every batch traverses leaves; no point lookup for each row.
                using Tree=decltype(table.tree);
                std::vector<typename Tree::KeyValuePair> rows;rows.reserve(4096);
                uint32_t low=0;
                for(;;) {
                    table.tree.scan(TableStorage::KeyWrapper(low),TableStorage::KeyWrapper(UINT32_MAX),true,true,4096,rows);
                    if(rows.empty())break;
                    for(const auto& row:rows) {
                        ++checks;auto* record=row.second.getValue();
                        if(predicates[q].check(table.get_attr(record))) {
                            std::function<float()> dist=[&]{++comparisons;return distance_handler->compute(query,table.get_vector(record),table.n_dim);};
                            visit(row.first.getValue(),dist);
                        }
                    }
                    auto last=rows.back().first.getValue();
                    if(rows.size()<4096 || last==UINT32_MAX)break;
                    low=last+1;
                }
#endif
            };
            auto& seen=*visited[omp_get_thread_num()];std::vector<icde::Hit> found;
            if(algorithm=="favor" && options.favor_oor_results)found=icde::favor<icde::Graph, true>(graph,access,seen,k,ef,options.favor_delta_d,
                options.favor_cache_sampling?&sampling_cache[q]:nullptr);
            else if(algorithm=="favor")found=icde::favor(graph,access,seen,k,ef,options.favor_delta_d,
                options.favor_cache_sampling?&sampling_cache[q]:nullptr);
            else if(algorithm=="cgif")found=icde::cgif(graph,clusters,access,seen,k,ef);
            else found=icde::pathseer(graph,access,seen,k,ef,costs);
            for(int j=0;j<k;++j) results[q*k+j]=j<int(found.size())?std::make_pair(found[j].second,found[j].first):std::make_pair(-1,std::numeric_limits<float>::infinity());
        }
        double seconds=std::chrono::duration<double>(Clock::now()-start).count();
        double recall=calculate_recall(gt,results.data(),nq,k,false);
        all_cmps.push_back(double(comparisons)/nq);all_qps.push_back(nq/seconds);all_recall.push_back(recall);
        std::cerr<<algorithm<<" ef="<<ef<<" QPS="<<all_qps.back()<<" comparisons="<<all_cmps.back()<<" predicates="<<double(checks)/nq<<" recall="<<recall<<"\n";
    }
    for(const auto* values:{&all_cmps,&all_qps,&all_recall}) {std::cout<<"[";for(double x:*values)std::cout<<x<<",";std::cout<<"],";}
    std::cout<<std::endl;
}
}  // namespace

// --icde_baseline accepts a comma-separated list (e.g. favor,cgif,pathseer): one
// execution loads the table/queries once and parses the stock HNSW once for FAVOR
// and CGIF. Each algorithm prints its own [cmps],[QPS],[recall] row, in list order.
template<class Pred>
void test_search_icde(TableStorage& table,std::shared_ptr<ANNS::IStorage> queries,
                     const std::vector<Pred>& predicates,const std::string& prefix,
                     int k,const std::vector<int>& efs,std::pair<ANNS::IdxType,float>* gt,
                     const IcdeSearchOptions& options) {
    std::vector<std::string> algorithms;
    {std::stringstream ss(options.algorithm);std::string a;while(std::getline(ss,a,','))if(!a.empty())algorithms.push_back(a);}
    if(algorithms.empty())throw std::invalid_argument("empty icde_baseline");
    for(const auto& a:algorithms)
        if(a!="favor" && a!="cgif" && a!="pathseer" && a!="favor-calibrate")throw std::invalid_argument("icde_baseline entries must be favor, cgif, pathseer or favor-calibrate");
    std::unique_ptr<icde::Graph> stock,specialized;
    for(const auto& a:algorithms) {
        std::cerr<<"=== ICDE baseline "<<a<<" ===\n";
        if(a=="pathseer") {
            if(!specialized)specialized.reset(new icde::Graph(icde::load_graph(options.pathseer_graph)));
            run_icde_one(table,queries,predicates,k,efs,gt,options,a,*specialized);
        } else {
            if(!stock)stock.reset(new icde::Graph(stock_hnsw(prefix,table.nb)));
            run_icde_one(table,queries,predicates,k,efs,gt,options,a,*stock);
        }
    }
}
template void test_search_icde<QueryType>(TableStorage&,std::shared_ptr<ANNS::IStorage>,const std::vector<QueryType>&,
    const std::string&,int,const std::vector<int>&,std::pair<ANNS::IdxType,float>*,const IcdeSearchOptions&);
template void test_search_icde<DisjunctiveQueryType>(TableStorage&,std::shared_ptr<ANNS::IStorage>,const std::vector<DisjunctiveQueryType>&,
    const std::string&,int,const std::vector<int>&,std::pair<ANNS::IdxType,float>*,const IcdeSearchOptions&);
