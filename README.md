# How to build

Dependencies:

- CMake >= 3.15
- C++17 compiler
- Boost library
    - program_options component
    - filesystem component
- Intel TBB library (for concurrent hashmap)
- OpenMP support
- AVX2 instruction set support

Do the following to compile the project:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
```

# Preparing the dataset

There are six datasets (see separate repo for dataset overview and preparation scripts):

1. SIFT (1M vectors)
2. GIST (1M vectors)
3. ArXiv (132K vectors)
4. RedCaps (6.9M vectors)
5. WIT (6.5M vectors)
6. SIFT-100M (100M vectors)

Each dataset contains (A) base vectors, (B) query vectors, (C) attributes associated with each base vector, and (D) query filters associated with the query vectors. A copy of the query filters are provided at `query_filters/`.

All datasets and indexes are placed under one root directory. After you have the datasets, copy `scripts/config_example.sh` to `scripts/config.sh`, then edit `config.sh` to set `PATHFINDER_DATA_ROOT` to that directory. The file paths expected under the root are listed in the dataset block near the top of each program under `app/src/` (e.g. `app/src/jul14_demo.cpp`).

# How to run

You can run the experiments using:

```
bash ./scripts/run_everything.sh
```

Or equivalently, execute the following scripts sequentially:

- `scripts/run_buildgraphs.sh`: For each dataset, this script builds one graph per attribute.
- `scripts/run_fig7_conjunctive.sh`: This executes PathFinder and the baselines on conjunctive query filters. (`query_filters/conjunctives`)
- `scripts/run_fig8_disjunctive.sh`: This executes PathFinder and the baselines on disjunctive query filters. (`query_filters/disjunctives`)
- `scripts/run_icde_baselines.sh`: This executes FAVOR on both of the above, and FAVOR, CGIF and PathSeer on RedCaps.
- `scripts/run_updates_build.sh`: Builds a partial PathFinder index over a fraction of the base vectors (the remainder is inserted online during the benchmark).
- `scripts/run_updates_bench.sh`: Runs the online-update benchmarks (PathFinder + HNSW / Vamana / NaviX / FAVOR baselines): insertion and deletion throughput, then mixed query/update workloads.

`scripts/config.sh` selects the datasets, selectivity levels, and thread counts for all of the scripts.

A few notes:

1. FAVOR, CGIF and PathSeer are implemented in `modules/baselines/`. CGIF and PathSeer search indexes built with their original code (versions and conversion tools in `tools/baseline_builders/`); set their paths in `config.sh`.
2. ACORN is evaluated when its graphs are present under `data/index_files/acorn/txt/`. (These are exported from ACORN's codebase.)
3. The code for the remaining experiments is under `app/src/` and `tests/unit/`.
