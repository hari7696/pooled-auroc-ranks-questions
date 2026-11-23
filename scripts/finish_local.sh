#!/usr/bin/env bash
# Local stages for runs whose GPU stages ran on the cluster without clustering/probes (.gpu_done).
set -u
cd "$(dirname "$0")/.."   # run inside the project environment (see README)
export HF_HUB_OFFLINE=1 WQ_OUT=${WQ_OUT:-outputs_h100}
for d in $WQ_OUT/*__*; do
  r=$(basename $d)
  [ -f $d/labels.npz ] && [ -s $d/hidden.npz ] || continue
  [ -f $d/items_L1_alias.parquet ] && continue
  python src/cluster.py --device cuda --runs $r
  python src/build_items.py --runs $r
  python src/build_items.py --runs $r --layer_idx 0
  python src/build_items.py --runs $r --layer_idx 2
  python src/build_items.py --runs $r --label alias
  echo "local done $r"
done
