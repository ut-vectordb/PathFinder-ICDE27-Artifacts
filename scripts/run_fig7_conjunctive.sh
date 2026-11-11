#!/bin/bash
# Fig. 7 (and Fig. 9a for SIFT100M): conjunctive filters at low, medium, and high
# selectivity. One run per workload evaluates PathFinder, OOR HNSW, OOR Vamana,
# NaviX, pre-filtering, and (if its graph is present) ACORN.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
LOGDIR=$(new_logdir fig7-conjunctive)

for d in "${DATASETS[@]}"; do
    for s in "${SELECTIVITIES[@]}"; do
        args=(--flag0 102 --dataset "$d" --num_threads "$N_THREADS" --flag7 2)
        if [[ $d == sift100M ]]; then
            # SIFT100M runs the baselines (flag2=2) and PathFinder (flag2=1) separately
            # so that PathFinder can release the raw base vectors after loading.
            run_case "$LOGDIR/work/$d-$s" "$(conj_tsv "$d" "$s")" queries.tsv "$LOGDIR/$d-$s-baselines.log" \
                "$BIN/app/jul14_demo" "${args[@]}" --flag2 2
            run_case "$LOGDIR/work/$d-$s" "$(conj_tsv "$d" "$s")" queries.tsv "$LOGDIR/$d-$s-pathfinder.log" \
                "$BIN/app/jul14_demo" "${args[@]}" --flag2 1
        else
            run_case "$LOGDIR/work/$d-$s" "$(conj_tsv "$d" "$s")" queries.tsv "$LOGDIR/$d-$s.log" \
                "$BIN/app/jul14_demo" "${args[@]}"
        fi
    done
done
