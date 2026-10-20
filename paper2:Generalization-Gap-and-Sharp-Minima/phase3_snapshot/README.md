# Reproducing Figure 3 (F1, C1, C2) of Keskar et al. (2017), "On large-batch training for deep learning"

Small-batch (256) vs large-batch (10% of the training set) ADAM training of F1 (MNIST, fully connected),
C1 (CIFAR-10, shallow conv) and C2 (CIFAR-10, VGG-like), followed by the loss/accuracy along the line
between the two solutions, alpha in [-1, 2].

```
pip install -r requirements.txt
bash run_all.sh                         # trains SB and LB for F1, C1, C2, interpolates, plots (~3 h on an M4 Pro)
python3 plot_fig3.py --bn interp        # or --bn recompute (batch-norm statistics re-estimated at each alpha)
jupyter notebook reproduce_fig3.ipynb
```

* `data.py` — MNIST / CIFAR-10 (torchvision download or `MNIST_NPZ`, `CIFAR10_NPZ`).
* `models.py` — F1, C1, C2 as in Appendix B of the paper (batch norm + ReLU, dropout on dense layers).
* `train.py` — ADAM (defaults) until the full-training-set loss stops improving (patience 10 epochs);
  SB and LB share the initialisation for a given `--seed`; saves `results/<net>_<SB|LB>_s0.{json,pt}`.
* `interpolate.py` — evaluates train/test cross-entropy and accuracy at 61 points of alpha, with the
  batch-norm running statistics either interpolated or recomputed; writes `results/<net>_interp_s0.json`.
* `plot_fig3.py` — the figure; `figures/` — the reproduced panels.
