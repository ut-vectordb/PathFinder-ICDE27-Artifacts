#!/bin/bash
# Fig. 10 on RedCaps, starting from the indexes built by run_updates_build.sh.
#   (a, b) Insertion and deletion throughput: 100 batches of inserts and deletes.
#   (c)    Mixed workloads of 1M operations in which updates make up 1% or 10%
#          of operations. Each method searches with the ef that reaches 80% recall.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
LOGDIR=$(new_logdir updates-bench)

TRIMMED="$PATHFINDER_DATA_ROOT/data/index_files/online_update_trimmed/redcaps"
UPD="$BIN/app/updates"
common=(--dataset redcaps --num_threads "$N_THREADS" --ef_con 200)
# FAVOR's initial delta_d for the mixed runs. The throughput run below calibrates it
# on the starting graph; this is the value the paper's mixed runs used.
FAVOR_STRESS_DELTA=0.001541205682
declare -A EF=([hnsw]=1800 [vamana]=1260 [favor]=510 [navix]=380 [pathfinder]=30)

run() { local tag=$1; shift; echo "=== $tag"; stdbuf -oL "$@" 2>&1 | tee "$LOGDIR/$tag.log"; }

# Insertion and deletion throughput.
run pathfinder "$UPD/online_update_pathfinder_test" "${common[@]}" --index_dir "$TRIMMED" --n_batches 100
run hnsw       "$UPD/online_update_hnsw_baseline"   "${common[@]}"
run vamana     "$UPD/online_update_vamana_baseline" "${common[@]}"
run navix      "$UPD/online_update_navix_baseline"  "${common[@]}" --n_entry 15
run favor      "$UPD/online_update_favor_baseline"  "${common[@]}"

# Mixed query/update workloads.
for pct in 1 10; do
    mixed=(--mixed_ops 1000000 --update_percent "$pct")
    run mixed-pathfinder-$pct "$UPD/online_update_pathfinder_test" "${common[@]}" "${mixed[@]}" \
        --mixed_ef "${EF[pathfinder]}" --index_dir "$TRIMMED"
    run mixed-hnsw-$pct   "$UPD/online_update_hnsw_baseline"   "${common[@]}" "${mixed[@]}" --mixed_ef "${EF[hnsw]}"
    run mixed-vamana-$pct "$UPD/online_update_vamana_baseline" "${common[@]}" "${mixed[@]}" --mixed_ef "${EF[vamana]}"
    run mixed-navix-$pct  "$UPD/online_update_navix_baseline"  "${common[@]}" "${mixed[@]}" --mixed_ef "${EF[navix]}"
    run mixed-favor-$pct  "$UPD/online_update_favor_baseline"  "${common[@]}" "${mixed[@]}" --mixed_ef "${EF[favor]}" \
        --stress_delta "$FAVOR_STRESS_DELTA"
done
