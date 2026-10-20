#!/usr/bin/env bash
# SB and LB training (100 epochs each) for F1, C1 and C2, then the interpolation of Figure 3.
# Runs whose weights exist are skipped; interrupted runs resume from their checkpoint.
set -euo pipefail
cd "$(dirname "$0")"

run() {
  if [ ! -f "results/${1}_${2}_s${3}.pt" ]; then
    python3 train.py --net "$1" --regime "$2" --seed "$3"
  fi
}

interp() {
  local output="results/${1}_interp_s${2}.json"
  if [ ! -f "$output" ] ||
     [ "results/${1}_SB_s${2}.pt" -nt "$output" ] ||
     [ "results/${1}_LB_s${2}.pt" -nt "$output" ]; then
    python3 interpolate.py --net "$1" --seed "$2"
  fi
}

for net in F1 C1 C2; do
  run "$net" SB 0
  run "$net" LB 0
  interp "$net" 0
done

# two more seeds for F1, and C1 as I first read it (stride-2 convolutions)
for seed in 1 2; do
  run F1 SB "$seed"
  run F1 LB "$seed"
  interp F1 "$seed"
done
run C1s2 SB 0
run C1s2 LB 0
interp C1s2 0

python3 plot_fig3.py --bn recompute
python3 plot_fig3.py --bn recompute_norm
python3 summarize.py
