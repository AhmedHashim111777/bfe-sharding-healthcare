#!/usr/bin/env bash
# Builds results/raw/sharding_all, the union view of the base runs (results/raw/sharding) and the
# revision runs (results/raw/sharding_rev) that analyse.py, revision_posthoc.py and figures_rev.py read.
set -euo pipefail
cd "$(dirname "$0")/../raw"
mkdir -p sharding_all
for base in sharding sharding_rev; do
  [ -d "$base" ] || continue
  for prob in "$base"/*/; do
    p="$(basename "$prob")"; mkdir -p "sharding_all/$p"
    for alg in "$prob"*/; do
      a="$(basename "$alg")"
      [ -e "sharding_all/$p/$a" ] || ln -s "../../$base/$p/$a" "sharding_all/$p/$a"
    done
  done
done
echo "results/raw/sharding_all ready"
