#!/bin/bash
# Builds the starting indexes for the update benchmarks on RedCaps: PathFinder,
# HNSW, and Vamana over a random 80% of the base vectors (the remaining vectors
# are inserted during the benchmark). PathFinder's trees are then cut from three
# to two layers below the root, matching the search experiments.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
LOGDIR=$(new_logdir updates-build)

FULL="$PATHFINDER_DATA_ROOT/data/index_files/apr1_online_update/redcaps-pathfinder"
TRIMMED="$PATHFINDER_DATA_ROOT/data/index_files/online_update_trimmed/redcaps"

stdbuf -oL "$BIN/app/updates/build_partial_pathfinder" --dataset redcaps \
    --num_threads "$BUILD_THREADS" --fanout 6 --maxdepth 3 2>&1 | tee "$LOGDIR/build_partial_pathfinder.log"

mkdir -p "$TRIMMED"
"$BIN/app/updates/trim_saved_tree" --max-depth 2 --expected-depth 3 --output-dir "$TRIMMED" \
    "$FULL/attr1_tree.bin" "$FULL/attr2_tree.bin" 2>&1 | tee "$LOGDIR/trim_saved_tree.log"
for f in "$FULL"/*; do
    [[ -e "$TRIMMED/$(basename "$f")" ]] || cp -a "$f" "$TRIMMED/"
done
echo "Trimmed PathFinder index: $TRIMMED"
