#include "tablestorage/tablestorage.h"
#include <iostream>
#include <stdexcept>
#include <omp.h>
int main() {
    TableStorage table;
    // Own records independently; exercise the same B-tree<ValueType> as TableStorage.
    std::vector<float> values(8192);
    for(uint32_t i=0;i<values.size();++i) {
        values[i]=i;
        table.tree.insert(TableStorage::KeyWrapper(i),TableStorage::ValueType(&values[i]));
    }
    using Tree=decltype(table.tree);
    int failures=0;
    #pragma omp parallel for num_threads(4) reduction(+:failures)
    for(int trial=0;trial<24;++trial) {
        std::vector<Tree::KeyValuePair> rows;
        uint32_t low=0;size_t total=0;
        for(;;) {
            table.tree.scan(TableStorage::KeyWrapper(low),TableStorage::KeyWrapper(UINT32_MAX),true,true,4096,rows);
            for(auto& row:rows) {
                auto id=row.first.getValue();
                if(id!=total || *row.second.getValue()!=id)++failures;
                // A point lookup following a scan must not spin or see corruption.
                TableStorage::ValueType found;
                if(!table.tree.lookup(TableStorage::KeyWrapper(id),found)||found.getValue()!=&values[id])++failures;
                ++total;
            }
            if(rows.empty()||rows.size()<4096)break;
            low=rows.back().first.getValue()+1;
        }
        if(total!=values.size())++failures;
    }
    if(failures)throw std::runtime_error("B-tree scan/lookup regression");
    std::cout<<"24 concurrent two-batch B-tree scans plus point lookups passed\n";

    // Deletions leave key gaps behind stale inner separators. A batched scan restarting
    // at last+1 must still visit every live key: when a batch ends on a leaf's last live
    // key and its successor was deleted, last+1 routes to the end of that leaf
    // (regression: scan returned nothing there and callers stopped early).
    // Scattered deletes (as in the online-update experiment) avoid leaf merges.
    {
        TableStorage t2;
        const uint32_t n=200000;
        std::vector<float> v2(n);
        for(uint32_t i=0;i<n;++i){v2[i]=i;t2.tree.insert(TableStorage::KeyWrapper(i),TableStorage::ValueType(&v2[i]));}
        std::vector<char> live(n,1);size_t n_live=n;
        t2.tree.EBREnter();   // removals may retire nodes through EBR
        for(uint32_t i=0;i<n;++i) if((i*2654435761u)%4==0){
            if(!t2.tree.remove(TableStorage::KeyWrapper(i)))throw std::runtime_error("remove failed");
            live[i]=0;--n_live;}
        t2.tree.EBRExit();
        for(uint32_t limit: {1u,7u,4096u}) {
            std::vector<Tree::KeyValuePair> rows;uint32_t low=0;size_t seen=0;int64_t prev=-1;
            for(;;) {
                t2.tree.scan(TableStorage::KeyWrapper(low),TableStorage::KeyWrapper(UINT32_MAX),true,true,limit,rows);
                for(auto& row:rows){auto id=row.first.getValue();
                    if(int64_t(id)<=prev || !live[id] || *row.second.getValue()!=id)throw std::runtime_error("gap scan returned a wrong row");
                    prev=id;++seen;}
                if(rows.size()<limit)break;
                low=rows.back().first.getValue()+1;
            }
            if(seen!=n_live)throw std::runtime_error("gap scan (limit "+std::to_string(limit)+") visited "+std::to_string(seen)+" of "+std::to_string(n_live)+" live keys");
            std::cout<<"batched scan (limit "<<limit<<") after deletions visited all "<<seen<<" live keys\n";
        }
    }
}
