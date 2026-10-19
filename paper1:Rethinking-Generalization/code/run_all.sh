#!/usr/bin/env bash
# Every run behind Figure 1, one at a time (about 3.5 h on an Apple M4 Pro).
# Finished runs are skipped; interrupted runs resume from their checkpoint.
set -e
cd "$(dirname "$0")"

run() {  # run <out dir> <tag> <train.py args>
  local out=$1 tag=$2; shift 2
  if grep -qs final_test_err "$out/$tag.json"; then echo "skip $tag"; return; fi
  python3 train.py "$@" --out "$out" --tag "$tag"
}

# panel (a): Inception on the randomization tests
run results inception_true_s0            --arch inception --mode true
run results inception_corrupt_p1.00_s0   --arch inception --mode corrupt --p 1.0
run results inception_shuffled_pixels_s0 --arch inception --mode shuffled_pixels
run results inception_random_values_s0   --arch inception --mode random_values
run results inception_random_pixels_s0   --arch inception --mode random_pixels
run results inception_gaussian_s0        --arch inception --mode gaussian

# panels (b) and (c): label corruption for the three networks
for a in inception alexnet mlp1; do
  run results ${a}_true_s0 --arch $a --mode true
  for p in 0.2 0.4 0.6 0.8 1.0; do
    run results ${a}_corrupt_p${p}0_s0 --arch $a --mode corrupt --p $p
  done
done

# diagnostic: TensorFlow's default initialization
run results_ablation inception_true_glorot --arch inception --mode true --init glorot
run results_ablation mlp1_true_glorot      --arch mlp1 --mode true --init glorot
for p in 0.2 0.4 0.6 0.8 1.0; do
  run results_ablation mlp1_corrupt_p${p}0_glorot --arch mlp1 --mode corrupt --p $p --init glorot
done

python3 plot_fig1.py
python3 compare_with_paper.py
