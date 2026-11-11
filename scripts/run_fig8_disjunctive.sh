#!/bin/bash
# Fig. 8 (and Fig. 9b for SIFT100M): filters mixing disjunctions and conjunctions at low, medium, and high
# selectivity. One run per workload evaluates PathFinder, OOR HNSW, OOR Vamana,
# NaviX, pre-filtering, and (if its graph is present) ACORN.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
LOGDIR=$(new_logdir fig8-disjunctive)

for d in "${DATASETS[@]}"; do
    for s in "${SELECTIVITIES[@]}"; do
        args=(--flag0 102 --dataset "$d" --num_threads "$N_THREADS" --flag7 2)
        if [[ $d == sift100M ]]; then
            # SIFT100M runs the baselines (flag2=2) and PathFinder (flag2=1) separately
            # so that PathFinder can release the raw base vectors after loading.
            run_case "$LOGDIR/work/$d-$s" "$(disj_tsv "$d" "$s")" queries_disjunctive.tsv "$LOGDIR/$d-$s-baselines.log" \
                "$BIN/app/jul14_demo_disjunctive" "${args[@]}" --flag2 2
            run_case "$LOGDIR/work/$d-$s" "$(disj_tsv "$d" "$s")" queries_disjunctive.tsv "$LOGDIR/$d-$s-pathfinder.log" \
                "$BIN/app/jul14_demo_disjunctive" "${args[@]}" --flag2 1
        else
            run_case "$LOGDIR/work/$d-$s" "$(disj_tsv "$d" "$s")" queries_disjunctive.tsv "$LOGDIR/$d-$s.log" \
                "$BIN/app/jul14_demo_disjunctive" "${args[@]}"
        fi
    done
done
