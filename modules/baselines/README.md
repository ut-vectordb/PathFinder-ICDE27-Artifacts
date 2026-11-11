# FAVOR, CGIF, and PathSeer

`include/baselines/search.h` implements the search procedures of FAVOR, CGIF, and
PathSeer, following their original code. `app/src/icde_search.cpp` runs them from
`jul14_demo` and `jul14_demo_disjunctive` (`--icde_baseline favor,cgif,pathseer`),
with the same recall/QPS output as the other systems.

All three evaluate predicates through PathFinder's B-tree attribute table, inside
query timing, like PathFinder and the other baselines.

- **FAVOR** searches PathFinder's HNSW root graph. Its distance statistic `delta_d`
  is calibrated once per dataset (`--icde_baseline favor-calibrate`) on 10,000
  base vectors and passed to each search with `--favor_delta_d`.
- **CGIF** searches the HNSW root graph with its IVF clustering (`--cgif_clusters`).
- **PathSeer** searches its own graph (`--pathseer_graph`) with its adaptive search.

CGIF's clustering and PathSeer's graph are built with their original code and
converted by the tools in `tools/baseline_builders/`. Conversion keeps IDs,
adjacency and membership order, levels, and boundaries, and drops the vectors.

The PathSeer port is derived from PathSeer's Faiss-based code, under the MIT
license in `PathSeer-LICENSE`.
