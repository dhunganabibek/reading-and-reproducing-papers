# Figure 7A/7B of Higgins et al. (2017), "beta-VAE"

[Shared LLM conversation](https://chatgpt.com/share/e/6ad7087d-51d0-83e9-8f59-be137ca3d921)

A VAE (beta = 1) and a beta-VAE (beta = 4) with the paper's fully connected model for the 2D shapes (dSprites)
dataset, three seeds each, plus one model with beta = 16. `analyze.py` draws the latent responses and latent
traversals of Figure 7.

Run these commands from this `code/` directory in an activated Python environment:

```bash
pip install -r requirements.txt
jupyter notebook reproduce_fig7.ipynb
```

The notebook displays saved figures and measurements. To train the models and regenerate every
figure, use `bash run_all.sh`. With weights available, `python3 analyze.py --betas 4,1 --seeds 0,1,2`
recomputes the main comparison. Add `--summary-only` to read saved measurements without weights.

dSprites is downloaded from the official repository on first use, or set `DSPRITES_NPZ=/path/to/dsprites.npz`.

| File | Purpose |
|---|---|
| `data.py` | loads the 737,280 images and their factor values |
| `model.py` | `BetaVAE` (encoder 4096-1200-1200-2x10 with ReLU, decoder 10-1200-1200-1200-4096 with tanh, Bernoulli output) and `BatchedBetaVAE`, which runs separately parameterized models in one call |
| `train.py` | trains one model per (beta, seed) side by side on the same batches; Adagrad, lr 0.01, accumulator starting at 0.1 (TensorFlow's default); checkpoints and resumes |
| `analyze.py` | sorts latents by learnt variance, draws the figure, and prints what each used latent encodes and the reconstruction error |
| `run_all.sh` | all runs, one after the other |

Choices the paper leaves open: batch size 256, 30 epochs (86,400 updates), three seeds per beta, and a centered
heart of middle size as the starting image of the traversals.

`results/` contains the seven training logs. Weights are excluded from Git and are recreated by
`bash run_all.sh`. The default notebook reads `figures/fig7_summary.json` and displays the committed
figures without data or weights. Set `RUN_TRAINING = True` to train and regenerate them, or
`RECOMPUTE_ANALYSIS = True` to analyze existing weights. `reconstruct.py` regenerates the reconstruction
comparison used to answer the question about larger beta. `logs/` preserves the original training output.

The three initializations per beta share minibatch order within each training group. These are paired
comparisons, not independent replications of every source of training randomness. The reported
reconstruction measurements use the full training grid, not held-out images.
