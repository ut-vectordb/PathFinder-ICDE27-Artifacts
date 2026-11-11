#!/bin/bash
# The FAVOR, CGIF, and PathSeer search ports (see baselines/README.md).
#   1. FAVOR on every dataset, conjunctive and disjunctive (Figs. 7-9). Its
#      distance statistic delta_d is first calibrated offline per dataset.
#   2. Comparison with non-updatable indexes on RedCaps (Fig. 12):
#      FAVOR, CGIF, and PathSeer in one run per selectivity. Needs the CGIF and
#      PathSeer search exports configured in config.sh.
# Uses the stock HNSW root graph built by run_buildgraphs.sh.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
LOGDIR=$(new_logdir icde-baselines)

declare -A DELTA_D
for d in "${DATASETS[@]}"; do
    common=(--flag0 102 --dataset "$d" --num_threads "$N_THREADS")

    # Calibration ignores the queries, but the driver still parses a query file.
    run_case "$LOGDIR/work/$d-calibrate" "$(conj_tsv "$d" low)" queries.tsv "$LOGDIR/$d-favor-calibrate.log" \
        "$BIN/app/jul14_demo" "${common[@]}" --icde_baseline favor-calibrate
    DELTA_D[$d]=$(sed -n 's/^FAVOR_DELTA_D=//p' "$LOGDIR/$d-favor-calibrate.log" | tail -1)
    echo "FAVOR delta_d for $d: ${DELTA_D[$d]}"

    for s in "${SELECTIVITIES[@]}"; do
        run_case "$LOGDIR/work/$d-conj-$s" "$(conj_tsv "$d" "$s")" queries.tsv "$LOGDIR/$d-conj-$s-favor.log" \
            "$BIN/app/jul14_demo" "${common[@]}" \
            --icde_baseline favor --favor_delta_d "${DELTA_D[$d]}"
        run_case "$LOGDIR/work/$d-disj-$s" "$(disj_tsv "$d" "$s")" queries_disjunctive.tsv "$LOGDIR/$d-disj-$s-favor.log" \
            "$BIN/app/jul14_demo_disjunctive" "${common[@]}" \
            --icde_baseline favor --favor_delta_d "${DELTA_D[$d]}" \
            --query_tsv "$LOGDIR/work/$d-disj-$s/queries_disjunctive.tsv"
    done
done

if [[ -n "${DELTA_D[redcaps]:-}" ]]; then
    for s in "${SELECTIVITIES[@]}"; do
        run_case "$LOGDIR/work/redcaps-nonupdatable-$s" "$(conj_tsv redcaps "$s")" queries.tsv \
            "$LOGDIR/redcaps-nonupdatable-$s.log" \
            "$BIN/app/jul14_demo" --flag0 102 --dataset redcaps --num_threads "$N_THREADS" \
            --icde_baseline favor,cgif,pathseer --favor_delta_d "${DELTA_D[redcaps]}" \
            --cgif_clusters "$CGIF_CLUSTERS" --pathseer_graph "$PATHSEER_GRAPH"
    done
fi
