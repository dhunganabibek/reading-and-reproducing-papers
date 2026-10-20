#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 train.py --betas 1,4 --epochs ${EPOCHS:-30} --batch_size ${BS:-256}
python3 analyze.py --betas 4,1
