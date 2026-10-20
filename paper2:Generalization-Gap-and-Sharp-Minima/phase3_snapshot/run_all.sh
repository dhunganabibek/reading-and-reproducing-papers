#!/usr/bin/env bash
# F1 (MNIST), C1 and C2 (CIFAR-10): SB and LB training, then the alpha-interpolation of Figure 3.
set -e
cd "$(dirname "$0")"
for net in F1 C1 C2; do
  for regime in SB LB; do
    [ -f results/${net}_${regime}_s0.pt ] || python3 train.py --net $net --regime $regime
  done
  [ -f results/${net}_interp_s0.json ] || python3 interpolate.py --net $net
done
python3 plot_fig3.py --bn interp
python3 plot_fig3.py --bn recompute
