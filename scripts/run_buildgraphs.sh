#!/bin/bash
# Builds PathFinder's per-attribute indexes (one tree or hash index per attribute,
# plus the HNSW root graph) for every dataset in config.sh. Searches build any
# missing Vamana graph for the OOR Vamana baseline on their first run.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
LOGDIR=$(new_logdir buildgraphs)

for d in "${DATASETS[@]}"; do
    for a in 0 1 2; do
        # The driver reads a query file only for the attribute schema in its header.
        run_case "$LOGDIR/work/$d" "$(conj_tsv "$d" low)" queries.tsv "$LOGDIR/$d-attr$a.log" \
            "$BIN/app/jul14_demo" --flag0 102 --dataset "$d" \
            --num_threads "$BUILD_THREADS" --flag5 "$a"
    done
done
