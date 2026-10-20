# Reproducing Figure 7A/7B of Higgins et al. (2017), "beta-VAE"

A VAE (beta = 1) and a beta-VAE (beta = 4) with the paper's fully connected architecture for the 2D shapes
(dSprites) dataset, trained with Adagrad (lr 1e-2), followed by the latent-response and latent-traversal
analysis of Figure 7.

```
pip install -r requirements.txt
bash run_all.sh                         # trains both models (~10 min on an M4 Pro) and draws the figure
python3 analyze.py --betas 4,1          # redraw from the saved checkpoints
jupyter notebook reproduce_fig7.ipynb
```

* `data.py` — downloads `dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz` from the official repository
  (or use `DSPRITES_NPZ=/path/dsprites.npz`).
* `model.py` — encoder FC 1200-1200 (ReLU) -> 10 Gaussian latents; decoder FC 1200-1200-1200-4096 (tanh),
  Bernoulli likelihood; the per-sample reconstruction and per-dimension KL terms.
* `train.py` — trains all requested betas on the same minibatches in one process; logs the reconstruction
  term, the KL and the KL per latent each epoch to `results/log_beta<b>_s0.json`, checkpoints to `results/vae_beta<b>_s0.pt`.
* `analyze.py` — encodes the whole dataset, orders latents by learnt variance, computes the mean latent
  response per generative factor and the traversals, and draws `figures/fig7ab_reproduced.png`.
