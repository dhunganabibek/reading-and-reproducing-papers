# Figure 1 of Zhang et al. (2017), "Understanding deep learning requires rethinking generalization"

PyTorch reproduction of Figure 1 on CIFAR-10: (a) learning curves of the small Inception network on the
randomization tests, (b) convergence slowdown and (c) test error against label corruption, for the small
Inception, the small AlexNet and an MLP with one hidden layer of 512 units.

```
pip install -r requirements.txt
bash run_all.sh                         # all runs, one after the other (about 3.5 h on an Apple M4 Pro)
python3 plot_fig1.py                    # redraw figures/fig1_reproduced.png from results/
python3 compare_with_paper.py           # the paper's plotted values next to my runs
jupyter notebook reproduce_fig1.ipynb   # the figure and every number in the report, no retraining needed
```

CIFAR-10 is downloaded with torchvision on first use, or set `CIFAR10_NPZ=/path/to/cifar10.npz`.

| file | |
|---|---|
| `data.py` | loading, preprocessing (center crop to 28x28, per-image whitening) and the randomization tests |
| `models.py` | small Inception, small AlexNet, MLP; `python3 models.py` checks the parameter counts against Table 1 |
| `train.py` | one run: SGD with momentum 0.9, lr 0.1 (Inception) or 0.01, times 0.95 per epoch, batch 128, no regularization |
| `plot_fig1.py` | the figure; convergence time = steps until training accuracy first reaches 99% (`--threshold`) |
| `compare_with_paper.py` | tables of the report |
| `extract_paper_fig1.py` | reads the values plotted in the paper's Figure 1 from its PDF into `paper_fig1_values.json` |
| `results/`, `results_ablation/`, `logs/` | logs of the runs in the report (`results_ablation/` uses Glorot initialization) |

Choices the paper leaves open: batch size 128, stop at 100% training accuracy or 100 epochs, convergence at 99%
training accuracy, six corruption levels instead of eleven, one seed. "Random pixels" is run both ways: with
all 2352 values of an image permuted (`random_values`, the one in the figure) and with each pixel's three color
values kept together (`random_pixels`, the dashed curve).

Run one training job at a time; several jobs at once on a laptop GPU crashed the machine.
