#!/bin/bash
# Runs every experiment in order. Each script can also be run on its own.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
bash run_buildgraphs.sh
bash run_fig7_conjunctive.sh
bash run_fig8_disjunctive.sh
bash run_icde_baselines.sh
bash run_updates_build.sh
bash run_updates_bench.sh
