# Figure 3 (F1, C1, C2) of Keskar et al. (2017), "On large-batch training for deep learning"

[Shared LLM conversation](https://chatgpt.com/share/e/6ad7087b-6704-83e8-9168-a7a5151d61f2)

Small-batch (256) against large-batch (10% of the training set) ADAM training of F1 (MNIST, fully connected),
C1 (CIFAR-10, shallow convolutional) and C2 (CIFAR-10, VGG-like), then the loss and accuracy along the straight
line through the two solutions for alpha in [-1, 2].

Run these commands from this `code/` directory in an activated Python environment:

```bash
pip install -r requirements.txt
jupyter notebook reproduce_fig3.ipynb
```

The notebook redraws the required panels from saved JSON measurements. To recreate weights and
interpolation results, use `bash run_all.sh`. To redraw just the figure, use
`python3 plot_fig3.py --bn recompute`; `python3 summarize.py` prints the report measurements.

MNIST and CIFAR-10 are downloaded with torchvision on first use, or set `MNIST_NPZ` / `CIFAR10_NPZ`.

| File | Purpose |
|---|---|
| `data.py` | MNIST (pixels in [0, 1]) and CIFAR-10 (standardized per channel), no augmentation |
| `models.py` | F1, C1, C2 from Appendix B; `C1s2` is my first reading of C1 (stride-2 convolutions) |
| `train.py` | ADAM with default settings, 100 epochs; SB and LB share the starting weights for a given `--seed`; large batches accumulate gradients over chunks; batch norm sees each chunk |
| `interpolate.py` | loss and accuracy at 61 values of alpha, with batch-norm statistics re-measured (`recompute`), interpolated (`interp`), or re-measured after normalizing the layers that feed a batch norm (`recompute_norm`) |
| `plot_fig3.py`, `summarize.py` | the figures and the numbers |

`results/` contains training logs and interpolation measurements in JSON, including distances from
the initial weights. Model weights are excluded from Git; `bash run_all.sh` recreates them and recomputes
interpolation whenever weights change. Plotting, summaries and the default notebook need no weights
or datasets. `logs/` preserves the original training output and failed attempt.

Choices the paper leaves open: 100 epochs for both (as in the paper's Figures 2, 4 and 5), a shared starting
point, how batch-norm statistics are obtained at interpolated weights, the preprocessing, and one pair of runs
per network (three for F1).

`recompute` recalibrates batch norm on training batches with dropout off. `interp` interpolates stored
statistics. `recompute_norm` first normalizes the layers that feed batch norm; scale cancellation is
approximate because batch norm includes epsilon. This diagnostic is discussed in the report.
