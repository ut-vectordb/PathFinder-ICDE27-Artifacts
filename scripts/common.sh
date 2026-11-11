# Shared setup for the run scripts; sourced, not executed.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$PROJECT_ROOT/scripts/config.sh" ]]; then
    echo "Missing scripts/config.sh: copy scripts/config_example.sh and edit it." >&2
    exit 1
fi
source "$PROJECT_ROOT/scripts/config.sh"
BIN="$PROJECT_ROOT/build"
FILTERS="$PROJECT_ROOT/query_filters"

cmake -S "$PROJECT_ROOT" -B "$BIN" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$BIN" -j

# A fresh, timestamped results folder per invocation of a run script.
new_logdir() {
    local dir="$RESULTS_DIR/$1/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$dir"
    echo "$dir"
}

# run_case WORKDIR TSV TSV_NAME LOG EXE ARGS...
# Runs EXE inside WORKDIR with the query filters copied to TSV_NAME (the drivers
# read queries.tsv / queries_disjunctive.tsv from the working directory). Ground
# truth is computed on the first run and cached as WORKDIR/groundtruth.bin.
run_case() {
    local workdir=$1 tsv=$2 tsv_name=$3 log=$4
    shift 4
    mkdir -p "$workdir"
    cp "$tsv" "$workdir/$tsv_name"
    echo "=== $(basename "$log" .log)"
    (cd "$workdir" && stdbuf -oL "$@" 2>&1 | tee "$log")
}

# Query filter files bundled under query_filters/.
conj_tsv() { echo "$FILTERS/conjunctives/$1_two_three_$2.tsv"; }
disj_tsv() { echo "$FILTERS/disjunctives/$1_disjunctive_$2.tsv"; }

