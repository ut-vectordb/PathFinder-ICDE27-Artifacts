#pragma once
// Search-only ports. References pinned in baselines/reference-revisions.json.
// FAVOR: include/favor.h; CGIF: src/ivf_connect.h (bitset overload);
// PathSeer: faiss/impl/PathSeer.cpp (adaptive overhead-bounded full search).
// Storage/predicate callbacks are supplied by PathFinder, never upstream arrays.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

namespace icde {
using Id = int32_t;
using Hit = std::pair<float, Id>;
struct DistanceLess { bool operator()(const Hit& a, const Hit& b) const { return a.first < b.first; } };
using Heap = std::priority_queue<Hit, std::vector<Hit>, DistanceLess>;
struct Span {
    const Id* first; const Id* last;
    const Id* begin() const { return first; }
    const Id* end() const { return last; }
    size_t size() const { return last-first; }
};
// Each vertex has consecutive level records; each record has an ordered edge
// span and a distance/expansion boundary. No vectors or attributes live here.
struct Graph {
    Id entry = -1;
    int32_t max_level = 0, M = 0, maxM0 = 0;
    std::vector<uint64_t> node{0}, offset{0};
    std::vector<int32_t> boundary, edges;
    size_t size() const { return node.size()-1; }
    Span neighbors(Id id, int level=0) const {
        auto r = node[id]+level;
        return {edges.data()+offset[r], edges.data()+offset[r+1]};
    }
    int split(Id id, int level=0) const { return boundary[node[id]+level]; }
};
struct Clusters {
    std::vector<uint64_t> offset{0};
    std::vector<Id> ids, assignment;
    Span members(int c) const { return {ids.data()+offset[c],ids.data()+offset[c+1]}; }
};
struct Visited {
    std::vector<uint32_t> tags; uint32_t epoch=0;
    explicit Visited(size_t n): tags(n) {}
    void clear() { if (++epoch==0) { std::fill(tags.begin(),tags.end(),0); ++epoch; } }
    bool has(Id i) const { return tags[i]==epoch; }
    void set(Id i) { tags[i]=epoch; }
};
struct Access {
    std::function<float(Id)> distance;
    std::function<bool(Id)> predicate;
    // True table scan; callback receives matching row's distance lazily.
    std::function<void(const std::function<void(Id, const std::function<float()>&)>&)> scan;
};
inline void keep(Heap& h, Hit v, int k) {
    if (h.size()<size_t(k) || v.first<h.top().first) { h.push(v); if(h.size()>size_t(k)) h.pop(); }
}
inline std::vector<Hit> finish(Heap h, int k) {
    while(h.size()>size_t(k)) h.pop();
    std::vector<Hit> out(h.size());
    for(size_t i=out.size();i>0;--i) {out[i-1]=h.top();h.pop();}
    return out;
}
template<class GraphView>
inline Hit descend(const GraphView& g, const Access& a, bool distance_zone=false) {
    Hit best{a.distance(g.entry),g.entry};
    for(int level=g.max_level;level>0;--level) {
        Id prev;
        do {
            prev=best.second;
            auto ns=g.neighbors(prev,level);
            size_t count=distance_zone?g.split(prev,level):ns.size();
            for(size_t j=0;j<count;++j) {
                Id id=ns.first[j]; if(id<0) break;
                float d=a.distance(id); if(d<best.first) best={d,id};
            }
        } while(prev!=best.second);
    }
    return best;
}
// Upstream sampling positions and denominator (including rounding).
inline std::pair<size_t,size_t> favor_sampling(size_t n) {
    if(n>100000)return {n/10000,10000};
    if(n>10000)return {100,n/100};
    if(n>1000)return {10,n/10};
    return {1,n};
}
template<class GraphView, bool RetainAllMatches = false>
inline std::vector<Hit> favor(const GraphView& g, const Access& a, Visited& seen,
                              int k, int ef, float delta_d,
                              const std::vector<uint8_t>* sampling_cache=nullptr) {
    if(k<=0 || ef<=0 || !std::isfinite(delta_d) || delta_d<0)
        throw std::invalid_argument("FAVOR requires positive k/ef and explicit finite nonnegative delta_d");
    if(!g.size()) return {};
    size_t n=g.size();auto [step,samples]=favor_sampling(n);
    if(sampling_cache && sampling_cache->size()!=(n+step-1)/step)
        throw std::invalid_argument("FAVOR sampling cache size mismatch");
    size_t passed=0, sample=0;
    for(size_t i=0;i<n;i+=step,++sample)
        passed+=sampling_cache?(*sampling_cache)[sample]:a.predicate(i);
    float p=float(passed)/samples;
    Heap top;
    if(p<=0.01f) {
        a.scan([&](Id id,const std::function<float()>& distance){keep(top,{distance(),id},k);});
        return finish(top,k);
    }
    float penalty=(1.0f-p)*(float(ef)-p)*delta_d/(2.0f*p)/float(ef);
    auto entry=descend(g,a);
    // Upstream recomputes the bottom entry distance after upper-layer descent.
    bool valid=a.predicate(entry.second);
    float d=a.distance(entry.second)+(valid?0:penalty);
    // OOR-result ablation: retain matching discoveries independently of the beam.
    Heap discovered;
    if constexpr (RetainAllMatches) { if(valid)keep(discovered,{d,entry.second},k); }
    Heap queue; top.push({d,entry.second}); queue.push({-d,entry.second});
    size_t in_range=valid; float bound=d; int capacity=std::max(k,ef);
    seen.clear(); seen.set(entry.second);
    while(!queue.empty()) {
        auto cur=queue.top();
        if(-cur.first>0.95f*bound && in_range>k*0.5) break;
        queue.pop();
        for(Id id:g.neighbors(cur.second)) {
            if(seen.has(id)) continue;
            seen.set(id); bool pass=a.predicate(id);
            // Upstream FastApproxL2Sqr actually evaluates every dimension.
            float dist=a.distance(id)+(pass?0:penalty);
            if constexpr (RetainAllMatches) { if(pass)keep(discovered,{dist,id},k); }
            if(top.size()<size_t(capacity) || bound>dist) {
                queue.push({-dist,id});top.push({dist,id});in_range+=pass;
                if(top.size()>size_t(capacity)) {
                    auto evicted=top.top();top.pop();
                    in_range-=a.predicate(evicted.second);
                }
                bound=top.top().first;
            }
        }
    }
    if constexpr (RetainAllMatches) return finish(discovered,k);
    Heap valid_top;
    while(!top.empty()) {auto h=top.top();top.pop();if(a.predicate(h.second))keep(valid_top,h,k);}
    return finish(valid_top,k);
}

inline std::vector<Hit> cgif(const Graph& g, const Clusters& clusters, const Access& a,
                             Visited& seen,int k,int ef) {
    ef=std::max(k,ef);
    auto entry=descend(g,a);
    // Match the base-layer initialization, including its separate distance call.
    entry.first=a.distance(entry.second);
    Heap queue, satisfied;
    float bound=std::numeric_limits<float>::max();
    if(a.predicate(entry.second)) {satisfied.push(entry);bound=entry.first;}
    queue.push({-entry.first,entry.second}); seen.clear();seen.set(entry.second);
    std::vector<size_t> cursor(clusters.offset.size()-1,0);
    std::vector<char> complete(cursor.size(),0);
    bool monotonic=true;
    float window_sum=entry.first*3;
    std::vector<float> window(3,entry.first);
    double sum_distance=0; // Upstream does not include the entry in this sum.
    size_t score_limit=g.maxM0*6;
    std::vector<Id> score,explore;score.reserve(score_limit);explore.reserve(g.maxM0);
    while(!queue.empty()) {
        auto cur=queue.top();float current=-cur.first;
        if(current>bound && satisfied.size()==size_t(ef)) {
            break;
        }
        queue.pop();score.clear();explore.clear();
        for(Id id:g.neighbors(cur.second)) if(!seen.has(id)) {
            seen.set(id);
            if(a.predicate(id)) score.push_back(id);
            else if(explore.size()<size_t(g.maxM0)) explore.push_back(id);
        }
        if(!monotonic && !explore.empty()) {
            size_t limit=score.size()+explore.size();
            if(!satisfied.empty() && current<sum_distance/satisfied.size())
                limit=score.size()+explore.size()*size_t(ef*0.1f);
            limit=std::min(limit,score_limit);
            for(Id id:explore) {
                if(score.size()>=limit) break;
                for(Id next:g.neighbors(id)) {
                    if(!seen.has(next)) {seen.set(next);if(a.predicate(next))score.push_back(next);}
                    if(score.size()>=limit) break;
                }
            }
            int c=clusters.assignment[cur.second];
            if(score.size()<limit && !complete[c]) {
                auto members=clusters.members(c);size_t pos=cursor[c];
                for(;pos<members.size() && score.size()<limit;++pos) {
                    Id id=members.first[pos];
                    if(!seen.has(id) && a.predicate(id)) {
                        seen.set(id);score.push_back(id);
                        if(score.size()>=limit) break;
                    }
                }
                cursor[c]=pos;
                if(pos+1>=members.size()) complete[c]=true;
            }
        }
        for(Id id:score) {
            float d=a.distance(id);
            if(d<bound || satisfied.size()<size_t(ef)) {
                queue.push({-d,id});satisfied.push({d,id});sum_distance+=d;
                if(satisfied.size()>size_t(ef)) {sum_distance-=satisfied.top().first;satisfied.pop();}
                bound=satisfied.top().first;
            }
        }
        if(monotonic) for(Id id:explore) {
            float d=a.distance(id);
            if(d<bound || satisfied.size()<size_t(ef)) queue.push({-d,id});
        }
        if(monotonic && !queue.empty()) {
            float last=window_sum;
            window_sum-=window[0]; window.push_back(-queue.top().first);
            // Preserves the upstream formula as written (index 2,
            // not the newly appended item at index 3).
            window_sum+=window[2];window.erase(window.begin());
            if(last<window_sum) monotonic=false;
        }
    }
    return finish(satisfied,k);
}

// PathSeer's bounded max heap retains distances of expanded entries. They still
// participate in count_below and capacity pruning; a normal frontier is NOT equivalent.
class Minimax {
    std::vector<Hit> h; size_t capacity; int valid=0;
public:
    explicit Minimax(size_t n):capacity(n) {h.reserve(n);}
    void push(Id id,float d) {
        if(h.size()==capacity) {
            if(d>=h.front().first)return;
            std::pop_heap(h.begin(),h.end());h.pop_back();--valid;
        }
        h.push_back({d,id});std::push_heap(h.begin(),h.end());++valid;
    }
    int size() const {return valid;}
    Hit pop() {
        size_t at=h.size();float best=std::numeric_limits<float>::infinity();
        for(size_t i=0;i<h.size();++i) if(h[i].second!=-1 && h[i].first<=best) {at=i;best=h[i].first;}
        if(at==h.size())throw std::logic_error("empty minimax frontier");
        Hit out=h[at]; h[at].second=-1;--valid;return out;
    }
    int below(float d) const {int n=0;for(auto x:h)n+=x.first<d;return n;}
};
struct Costs {int64_t distance_ns=0,predicate_ns=0;};

inline std::vector<Hit> pathseer(const Graph& g,const Access& a,Visited& seen,
                                int k,int ef,Costs costs) {
    if(costs.distance_ns<=0 || costs.predicate_ns<=0)throw std::invalid_argument("PathSeer requires profiled positive costs");
    auto entry=descend(g,a,true);Heap results;
    Minimax queue(std::max(k,ef));queue.push(entry.second,entry.first);
    int nfilter=1,nsatisfy=0;
    if(a.predicate(entry.second))keep(results,entry,k);
    seen.clear();seen.set(entry.second);
    int64_t budget=2LL*g.M*(costs.distance_ns+costs.predicate_ns);
    std::vector<Id> batch;batch.reserve(g.maxM0);
    while(queue.size()>0) {
        auto cur=queue.pop();
        if(queue.below(cur.first)>=ef) break;
        auto ns=g.neighbors(cur.second);int split=g.split(cur.second);
        int visited=0, fresh=0, passed=0;
        batch.clear();
        for(int j=0;j<split;++j) {
            Id id=ns.first[j];if(id<0) break;
            if(seen.has(id)){++visited;continue;}
            seen.set(id);batch.push_back(id);
        }
        for(Id id:batch) {
            float d=a.distance(id);queue.push(id,d);
            if(results.size()<size_t(k) || d<results.top().first) {
                ++fresh;++nfilter;
                if(a.predicate(id)){++passed;++nsatisfy;keep(results,{d,id},k);}
            } else ++visited;
        }
        batch.clear();int count=0;
        for(size_t j=split;j<ns.size();++j) {
            if(count%5==0) {
                int64_t estimate=fresh?
                    (fresh*costs.predicate_ns+passed*costs.distance_ns)*(fresh+visited)/fresh:
                    int64_t(visited*(costs.predicate_ns+double(nsatisfy)/nfilter*costs.distance_ns));
                if(estimate>=budget) {
                    break;
                }
            }
            ++count;Id id=ns.first[j];if(id<0)break;
            if(seen.has(id)){++visited;continue;}
            seen.set(id);++nfilter;++fresh;
            if(a.predicate(id)){batch.push_back(id);++passed;++nsatisfy;}
        }
        for(Id id:batch) {float d=a.distance(id);queue.push(id,d);keep(results,{d,id},k);}
    }
    return finish(results,k);
}
} // namespace icde
