#!/usr/bin/env bash
# VAE (beta=1) and beta-VAE (beta=4) for three seeds, plus one beta=16 model, then Figure 7A/7B.
# About 1.3 h on an Apple M4 Pro; finished models are skipped and interrupted runs resume.
set -euo pipefail
cd "$(dirname "$0")"
python3 train.py --betas 1,4,16 --seeds 0 --skip-complete
python3 train.py --betas 1,4 --seeds 1,2 --skip-complete
python3 analyze.py --betas 4,1 --seeds 0,1,2
python3 analyze.py --betas 16 --seeds 0 --out ""
python3 reconstruct.py
