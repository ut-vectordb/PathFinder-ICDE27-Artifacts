#ifndef RUN_SEARCH_JUL14_H_INCLUDED

#include <vector>
#include <queue>
#include <iostream>
#include <algorithm>

#include "omp.h"

#include "hnsw/hnsw.h"
#include "irangegraph/irangegraph.h"
#include "irangegraph/attribute_tree.h"

#include "utils.h"

#include "tablestorage/tablestorage_utils.h"
#include "queryplanning/queryplanning.h"

#include "querytype.h"

using namespace std;

extern std::shared_ptr<ANNS::DistanceHandler> distance_handler;

extern int nb, nq, dim, na;
extern ANNS::IdxType M, ef_con;
extern uint32_t num_threads;

extern int flag0, flag1, flag2, flag3, flag4;
extern int flag13;

template <class T>
bool inrange(T a, T L, T R) { return L <= a && a < R; }

template <class T>
bool inrange(T a, const pair<T, T> &Range) { return Range.first <= a && a < Range.second; }

void test_search_IRG_collection(TableStorage &tbstorage,
                                shared_ptr<ANNS::IStorage> query_storage,
                                const vector<int> &index_ids,
                                const vector<shared_ptr<irg_attr>> &irg_indices,
                                const vector<float> &attrs,
                                const QueryCollection &range,
                                int K, const vector<int> &ef_searches, int n_entry,
                                pair<IdxType, float> *gt,
                                AttrTreeNode *rootgraph = nullptr);

void test_search_using_plan_separate_post(TableStorage &tbstorage,
                            shared_ptr<ANNS::IStorage> query_storage,
                            IQueryPlanner &query_planner,
                            const vector<float> &attrs,
                            const QueryCollection &range,
                            int K, const vector<int> &ef_searches, int n_entry,
                            pair<IdxType, float> *gt);

void test_search_postfiltering_baseline(
        TableStorage &tbstorage,
        shared_ptr<ANNS::IStorage> query_storage,
        IQueryPlanner &query_planner,
        const vector<float> &attrs,
        const QueryCollection &range,
        int K, const vector<int> &ef_searches, int n_entry,
        pair<IdxType, float> *gt);

void test_search_using_plan_separate_post_twophase(TableStorage &tbstorage,
                            shared_ptr<ANNS::IStorage> query_storage,
                            IQueryPlanner &query_planner,
                            const vector<float> &attrs,
                            const QueryCollection &range,
                            int K, const vector<int> &ef_searches, int n_entry,
                            pair<IdxType, float> *gt);

void test_search_acorn(TableStorage &tbstorage,
                       shared_ptr<ANNS::IStorage> query_storage,
                       shared_ptr<Graph> acorn_graph, int M_ACORN, int Mb_ACORN,
                       int gamma_ACORN, const vector<float> &attrs,
                       const QueryCollection &range, int K,
                       const vector<int> &ef_searches, int n_entry,
                       pair<IdxType, float> *gt,
                       vector<int> *bitmap = nullptr);

void test_search_acorn_twophase(TableStorage &tbstorage,
                       shared_ptr<ANNS::IStorage> query_storage,
                       shared_ptr<Graph> acorn_graph, int M_ACORN, int Mb_ACORN,
                       int gamma_ACORN, const vector<float> &attrs,
                       const QueryCollection &range, int K,
                       const vector<int> &ef_searches, int n_entry,
                       pair<IdxType, float> *gt,
                       vector<int> *bitmap = nullptr);

void test_search_navix(
        TableStorage &tbstorage,
        shared_ptr<ANNS::IStorage> query_storage,
        const unit_graph_ref &graphref,
        int M,
        const vector<float> &attrs,                 // for predicate = [&]()
        std::function<bool(int, int)> predicate,
        // const QueryCollection &range,
        int K, const vector<int> &ef_searches, int n_entry,
        pair<IdxType, float> *gt,
        int mode);

void test_search_navix_global(
        TableStorage &tbstorage,
        shared_ptr<ANNS::IStorage> query_storage,
        const unit_graph_ref &graphref,
        int M,
        const vector<float> &attrs,                 // for predicate = [&]()
        std::function<bool(int, int)> predicate,
        // const QueryCollection &range,
        int K, const vector<int> &ef_searches, int n_entry,
        pair<IdxType, float> *gt,
        int mode);

void search_graph_reference_doubling_postfiltering(
        TableStorage &tbstorage,
        char *query,    // query_vec
        std::function<bool(int)> predicate,
        int K,

        const unit_graph_ref &graphref,
        SearchCacheList &search_cache_list,
        vector<pair<IdxType, float>> &all,
        unordered_set<IdxType> &seen);

void search_graph_reference_simple_postfiltering(
        TableStorage &tbstorage,
        char *query,    // query_vec
        std::function<bool(int)> predicate,
        int K,

        const unit_graph_ref &graphref,
        SearchCacheList &search_cache_list,
        int &n_cmps,
        vector<pair<IdxType, float>> &all,
        unordered_set<IdxType> &seen);

void search_graph_reference_OOR(
        TableStorage &tbstorage,
        char *query,    // query_vec
        std::function<bool(int)> predicate,
        std::function<bool(int)> predicate_one, // We don't have functool in C++... Should we use a functor instead?
        int K, int n_entry,

        const unit_graph_ref &graphref,
        SearchCacheList &search_cache_list,
        int &n_cmps,
        vector<pair<IdxType, float>> &all,
        unordered_set<IdxType> &seen);

void search_HNSW_reference_OOR(
        TableStorage &tbstorage,
        char *query,    // query_vec
        std::function<bool(int)> predicate,
        std::function<bool(int)> predicate_one,
        int K,

        const unit_graph_ref &graphref,     // this function expects non-null gref->optional_HNSW_upper
        SearchCacheList &search_cache_list,
        int &n_cmps,
        vector<pair<IdxType, float>> &all,
        unordered_set<IdxType> &seen);

#define RUN_SEARCH_JUL14_H_INCLUDED
#endif