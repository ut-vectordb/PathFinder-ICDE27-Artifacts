# Copy to scripts/config.sh and edit before running any script.

# Root directory holding datasets, query workloads, and indexes (see README, "Data layout").
export PATHFINDER_DATA_ROOT=/path/to/pathfinder-data

N_THREADS=16                     # search threads; the paper uses 16
BUILD_THREADS=96                 # threads for index construction
RESULTS_DIR="$PROJECT_ROOT/results"

# Datasets to run. sift100M is large (100M vectors); remove it for a quicker run.
DATASETS=(sift3 gist3 arxiv redcaps wit sift100M)
SELECTIVITIES=(low mid high)

# CGIF and PathSeer search exports for RedCaps (see README, "ICDE baselines").
CGIF_CLUSTERS="$PATHFINDER_DATA_ROOT/index_files/CGIF/redcaps/redcaps-l2-nlist1024.search"
PATHSEER_GRAPH="$PATHFINDER_DATA_ROOT/index_files/PathSeer/redcaps/redcaps-l2-M64-Mexp128-ef128-efexp256.search"
