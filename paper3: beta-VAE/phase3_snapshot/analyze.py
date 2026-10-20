"""Figure 7A/7B of Higgins et al. (2017) for our trained models.

For each model: encode every image, order latents by their learnt (mean posterior) variance
(ascending = most informative first; equivalently by descending KL); then plot, per latent,
  row 1: mean of mu_i over the 32x32 (posX, posY) grid, averaged over shape / scale / rotation
  row 2: mean of mu_i vs scale, one curve per shape (square red, oval green, heart blue)
  row 3: mean of mu_i vs rotation, one curve per shape
  rows 4-8: decoder output when latent i is set to 3, 1.5, 0, -1.5, -3 and the other latents are
            kept at the posterior means of one dataset image
  last row: the learnt variance.  Five most informative latents + one unused latent are shown.
"""
import argparse, json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_dsprites
from model import BetaVAE
from train import get_device


@torch.no_grad()
def encode_all(model, imgs, dev, bs=8192):
    mus, vars_ = [], []
    for i in range(0, imgs.shape[0], bs):
        x = torch.from_numpy(imgs[i:i + bs]).to(dev).float()
        mu, logvar = model.encode(x)
        mus.append(mu.cpu()); vars_.append(logvar.exp().cpu())
    return torch.cat(mus).numpy(), torch.cat(vars_).numpy()


def group_mean(values, keys, sizes):
    """Mean of `values` (N,) for each combination of integer keys (N,k) with ranges `sizes`."""
    flat = np.ravel_multi_index(keys.T, sizes)
    s = np.bincount(flat, weights=values, minlength=int(np.prod(sizes)))
    c = np.bincount(flat, minlength=int(np.prod(sizes)))
    return (s / np.maximum(c, 1)).reshape(sizes)


def analyze(model, imgs, classes, dev, n_show=5, seed_img=None):
    mu, var = encode_all(model, imgs, dev)
    learnt_var = var.mean(0)                           # the "learnt variance" row of Fig. 7
    kl = 0.5 * (mu ** 2 + var - 1 - np.log(var)).mean(0)
    order = np.argsort(learnt_var)                     # informative first
    show = list(order[:n_show]) + [order[-1]]          # + one unused latent
    shape, scale, rot, px, py = classes[:, 1], classes[:, 2], classes[:, 3], classes[:, 4], classes[:, 5]
    res = []
    if seed_img is None:
        seed_img = np.flatnonzero((shape == 2) & (scale == 3) & (rot == 0) & (px == 16) & (py == 16))[0]
    with torch.no_grad():
        x0 = torch.from_numpy(imgs[seed_img:seed_img + 1]).to(dev).float()
        mu0, _ = model.encode(x0)
    for i in show:
        pos = group_mean(mu[:, i], np.stack([py, px], 1), (32, 32))
        sc = group_mean(mu[:, i], np.stack([shape, scale], 1), (3, 6))
        ro = group_mean(mu[:, i], np.stack([shape, rot], 1), (3, 40))
        trav = []
        with torch.no_grad():
            for v in [3.0, 1.5, 0.0, -1.5, -3.0]:
                z = mu0.clone(); z[0, i] = v
                trav.append(torch.sigmoid(model.decode(z)).view(64, 64).cpu().numpy())
        res.append(dict(latent=int(i), learnt_var=float(learnt_var[i]), kl=float(kl[i]), pos=pos, scale=sc,
                        rot=ro, trav=np.stack(trav)))
    return res, learnt_var, kl


def draw_panel(fig, gs_outer, res, title):
    cols = len(res)
    gs = gs_outer.subgridspec(8, cols, hspace=0.08, wspace=0.08)
    colors = ["red", "green", "blue"]                  # square, oval, heart
    for c, r in enumerate(res):
        ax = fig.add_subplot(gs[0, c]); ax.imshow(r["pos"], cmap="jet", origin="lower"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"$z_{{{r['latent'] + 1}}}$" if c < cols - 1 else "$z_{unused}$", fontsize=9)
        if c == 0: ax.set_ylabel("position", fontsize=7)
        ax = fig.add_subplot(gs[1, c])
        for s in range(3): ax.plot(r["scale"][s], color=colors[s], lw=1)
        ax.set_ylim(-3, 3); ax.set_xticks([]); ax.set_yticks([])
        if c == 0: ax.set_ylabel("scale", fontsize=7)
        ax = fig.add_subplot(gs[2, c])
        for s in range(3): ax.plot(r["rot"][s], color=colors[s], lw=1)
        ax.set_ylim(-3, 3); ax.set_xticks([]); ax.set_yticks([])
        if c == 0: ax.set_ylabel("rotation", fontsize=7)
        for k in range(5):
            ax = fig.add_subplot(gs[3 + k, c]); ax.imshow(r["trav"][k], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0: ax.set_ylabel(["3", "1.5", "0", "-1.5", "-3"][k], fontsize=7)
        ax.set_xlabel(f"{r['learnt_var']:.2f}", fontsize=8)
    fig.text(gs_outer.get_position(fig).x0 + gs_outer.get_position(fig).width / 2,
             gs_outer.get_position(fig).y1 + 0.01, title, ha="center", fontsize=11)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--betas", default="4,1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figures/fig7ab_reproduced.png")
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    dev = get_device(); imgs, classes = load_dsprites(args.data)
    betas = [float(b) for b in args.betas.split(",")]
    fig = plt.figure(figsize=(4.0 * len(betas) + 1, 9))
    outer = fig.add_gridspec(1, len(betas), wspace=0.12, left=0.05, right=0.98, top=0.92, bottom=0.04)
    summary = {}
    for j, b in enumerate(betas):
        m = BetaVAE().to(dev); m.load_state_dict(torch.load(os.path.join(args.results, f"vae_beta{b:g}_s{args.seed}.pt"), map_location=dev)); m.eval()
        res, lv, kl = analyze(m, imgs, classes, dev)
        label = ("A: " if j == 0 else "B: ") + (r"$\beta$-VAE ($\beta$=%g)" % b if b != 1 else "VAE ($\\beta$=1)")
        draw_panel(fig, outer[0, j], res, label)
        summary[f"beta{b:g}"] = dict(learnt_var=lv.tolist(), kl_per_dim=kl.tolist(), shown=[r["latent"] for r in res])
        print(f"beta={b:g}: learnt variance per latent {np.round(lv, 2).tolist()}")
        print(f"beta={b:g}: KL per latent        {np.round(kl, 2).tolist()}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=170); json.dump(summary, open(os.path.splitext(args.out)[0] + "_summary.json", "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
