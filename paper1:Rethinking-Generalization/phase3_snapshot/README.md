# Reproducing Figure 1 of Zhang et al. (2017), "Understanding deep learning requires rethinking generalization"

PyTorch reproduction of Fig. 1 (a) learning curves of a small Inception net on CIFAR-10 under the
randomization tests, (b) convergence slowdown vs. label corruption and (c) test error vs. label
corruption for Inception, AlexNet and MLP 1x512.

```
pip install -r requirements.txt
bash run_all.sh                         # full sweep (~3 h on an Apple M4 Pro GPU); writes results/ and figures/
python3 plot_fig1.py                    # redraw the figure from results/
jupyter notebook reproduce_fig1.ipynb   # the notebook does the same and shows sanity checks
```

* `data.py` — CIFAR-10 loading (torchvision download, or `CIFAR10_NPZ=/path/cifar10.npz`), centre crop to
  28x28, per-image whitening, and the five randomization tests (labels are randomized once and kept fixed).
* `models.py` — small Inception, small AlexNet, MLP; `python3 models.py` prints parameter counts that match
  Table 1 of the paper exactly (1,649,402 / 1,387,786 / 1,209,866 / 1,735,178).
* `train.py` — one run: SGD momentum 0.9, lr 0.1 (Inception) or 0.01, x0.95 per epoch, batch 128, no
  regularization; logs the loss every 25 steps and train/test accuracy every epoch to `results/<tag>.json`.
* `plot_fig1.py` — draws the three panels; convergence time = steps until training accuracy >= 99%.
* `results/` — logs of the runs used in the report; `figures/fig1_reproduced.png` — the reproduced figure.
