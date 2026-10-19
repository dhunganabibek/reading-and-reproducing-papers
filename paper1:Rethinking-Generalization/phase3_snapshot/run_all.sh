#!/usr/bin/env bash
# Full sweep for Figure 1. Two queues run concurrently (Inception is the expensive one).
# Runs whose log already has a final entry are skipped, so the script can be resumed.
# Set CIFAR10_NPZ to a cached npz or let data.py download CIFAR-10 with torchvision.
set -e
cd "$(dirname "$0")"
OUT=${OUT:-results}
EP=${EP:-100}
run() {  # run <tag> <train.py args...>
  local tag=$1; shift
  if [ -f "$OUT/$tag.json" ] && grep -q final_test_err "$OUT/$tag.json"; then echo "skip $tag"; return; fi
  python3 train.py "$@" --epochs $EP --out $OUT
}
inception_queue() {
  run inception_true_s0            --arch inception --mode true
  run inception_corrupt_p1.00_s0   --arch inception --mode corrupt --p 1.0
  run inception_shuffled_pixels_s0 --arch inception --mode shuffled_pixels
  run inception_random_pixels_s0   --arch inception --mode random_pixels
  run inception_gaussian_s0        --arch inception --mode gaussian
  for p in 0.2 0.4 0.6 0.8; do
    run inception_corrupt_p${p}0_s0 --arch inception --mode corrupt --p $p
  done
}
small_queue() {
  for a in mlp1 alexnet; do
    run ${a}_true_s0 --arch $a --mode true
    for p in 0.2 0.4 0.6 0.8 1.0; do
      run ${a}_corrupt_p${p}0_s0 --arch $a --mode corrupt --p $p
    done
  done
}
QUEUE=${QUEUE:-all}      # all | inception | small
[ "$QUEUE" != small ] && inception_queue >> logs_inception.txt 2>&1 &
[ "$QUEUE" != inception ] && small_queue >> logs_small.txt 2>&1 &
wait
python3 plot_fig1.py --results $OUT --out figures/fig1_reproduced.png
