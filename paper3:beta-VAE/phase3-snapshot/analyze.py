"""Draw Figure 7A/7B and measure latent responses over the full dSprites grid."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data import load_dsprites
from model import BetaVAE
from train import get_device

USED = 0.5   # a latent counts as used if its learnt variance is below this
TRAVERSAL = [3.0, 1.5, 0.0, -1.5, -3.0]
SUMMARY = "figures/fig7_summary.json"


@torch.no_grad()
def encode_all(model, imgs, device, batch_size=8192):
    """Posterior means and variances of every image, and the reconstruction from z = mu
    (-log p(x|z) in nats per image, and wrong pixels per image after thresholding at 0.5)."""
    mus, variances, nll, wrong = [], [], 0.0, 0.0
    for i in range(0, len(imgs), batch_size):
        x = torch.from_numpy(imgs[i:i + batch_size]).to(device).float()
        mu, logvar = model.encode(x)
        logits = model.decode(mu)
        x = x.flatten(1)
        # sum each image first, then the images in float64 (one float32 sum over 33M values is not accurate)
        nll += F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(1).cpu().double().sum().item()
        wrong += ((logits > 0) != (x > 0.5)).sum().item()
        mus.append(mu.cpu())
        variances.append(logvar.exp().cpu())
    mu = torch.cat(mus).numpy().astype(np.float64)
    var = torch.cat(variances).numpy().astype(np.float64)
    return mu, var, nll / len(imgs), wrong / len(imgs)


def group_mean(values, keys, sizes):
    """Mean of values (N,) for every combination of the integer keys (N, k), which range over sizes."""
    flat = np.ravel_multi_index(keys.T, sizes)
    total = np.bincount(flat, weights=values, minlength=int(np.prod(sizes)))
    count = np.bincount(flat, minlength=int(np.prod(sizes)))
    return (total / np.maximum(count, 1)).reshape(sizes)


def explained(values, keys, sizes):
    """Fraction of the variance of values explained by the group means (all groups have the same size)."""
    variance = values.var()
    return group_mean(values, keys, sizes).var() / variance if variance > 0 else 0.0


def analyze(model, imgs, classes, device, n_show=5, seed_img=None):
    mu, var, recon_nll, wrong_pixels = encode_all(model, imgs, device)
    learnt_var = var.mean(0)
    kl = 0.5 * (mu ** 2 + var - 1 - np.log(var)).mean(0)
    order = np.argsort(learnt_var)
    show = list(order[:n_show]) + [order[-1]]   # the most informative latents and one unused one
    shape, scale, rot, px, py = classes[:, 1], classes[:, 2], classes[:, 3], classes[:, 4], classes[:, 5]

    # traversals start from a heart of middle size, not rotated, in the center
    if seed_img is None:
        seed_img = np.flatnonzero((shape == 2) & (scale == 3) & (rot == 0) & (px == 16) & (py == 16))[0]
    columns = []
    with torch.no_grad():
        mu0, _ = model.encode(torch.from_numpy(imgs[seed_img:seed_img + 1]).to(device).float())
        for i in show:
            trav = []
            for v in TRAVERSAL:
                z = mu0.clone()
                z[0, i] = v
                trav.append(torch.sigmoid(model.decode(z)).view(64, 64).cpu().numpy())
            columns.append({"latent": int(i), "learnt_var": float(learnt_var[i]), "kl": float(kl[i]),
                            "pos": group_mean(mu[:, i], np.stack([py, px], 1), (32, 32)),
                            "scale": group_mean(mu[:, i], np.stack([shape, scale], 1), (3, 6)),
                            "rot": group_mean(mu[:, i], np.stack([shape, rot], 1), (3, 40)),
                            "trav": np.stack(trav)})

    table = []
    for i in order:
        if learnt_var[i] >= USED:
            continue
        m = mu[:, i]
        e_shape = explained(m, shape[:, None], (3,))
        row = {"latent": int(i), "learnt_var": float(learnt_var[i]), "kl": float(kl[i]),
               "shape": float(e_shape),
               "scale": float(explained(m, scale[:, None], (6,))),
               "rotation": float(explained(m, np.stack([shape, rot], 1), (3, 40)) - e_shape),   # within a shape
               "posX": float(explained(m, px[:, None], (32,))),
               "posY": float(explained(m, py[:, None], (32,))),
               "position": float(explained(m, np.stack([px, py], 1), (32, 32)))}
        # periods of the rotation response per full turn, per shape (rotation 40 equals rotation 0, so 39 values)
        rc = group_mean(m, np.stack([shape, rot], 1), (3, 40))[:, :39]
        spectrum = np.abs(np.fft.rfft(rc - rc.mean(1, keepdims=True), axis=1))
        row["rot_periods"] = [int(v) for v in spectrum[:, 1:].argmax(1) + 1]   # square, oval, heart
        row["rot_amplitude"] = [float(v) for v in (rc.max(1) - rc.min(1)) / 2]
        row["best_separate"] = max(row[k] for k in ("shape", "scale", "rotation", "posX", "posY"))
        row["best_joint_pos"] = max(row[k] for k in ("shape", "scale", "rotation", "position"))
        table.append(row)
    return columns, learnt_var, kl, table, {"recon_nll": recon_nll, "wrong_pixels": wrong_pixels}


def draw_panel(fig, gs_outer, columns, title):
    n = len(columns)
    gs = gs_outer.subgridspec(8, n, hspace=0.08, wspace=0.08)
    colors = ["red", "green", "blue"]   # square, oval, heart
    vmax = max(np.abs(c["pos"]).max() for c in columns)   # one color scale for all latents of a model
    for j, col in enumerate(columns):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(col["pos"], cmap="jet", origin="lower", vmin=-vmax, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"$z_{{{col['latent'] + 1}}}$" if j < n - 1 else "$z_{unused}$", fontsize=10)
        if j == 0:
            ax.set_ylabel("position", fontsize=8)
        for row, key in ((1, "scale"), (2, "rot")):
            ax = fig.add_subplot(gs[row, j])
            for s in range(3):
                ax.plot(col[key][s], color=colors[s], lw=1)
            ax.set_ylim(-3, 3)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel("scale" if key == "scale" else "rotation", fontsize=8)
        for k in range(5):
            ax = fig.add_subplot(gs[3 + k, j])
            ax.imshow(col["trav"][k], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"{TRAVERSAL[k]:g}", fontsize=8)
        ax.set_xlabel(f"{col['learnt_var']:.3f}", fontsize=9)
    box = gs_outer.get_position(fig)
    fig.text(box.x0 + box.width / 2, box.y1 + 0.025, title, ha="center", fontsize=12)


def print_model(beta, seed, table, kl, rec, entry):
    print(f"\nbeta={beta:g} seed={seed}: {len(table)} informative latents, total KL {kl.sum():.2f} nats, "
          f"reconstruction at z=mu: {rec['recon_nll']:.1f} nats and {rec['wrong_pixels']:.1f} wrong pixels per image")
    print("  latent  var    KL   | shape scale rot|shape  posX  posY  pos(x,y)")
    for r in table:
        print(f"  z{r['latent'] + 1:<2d}    {r['learnt_var']:.3f} {r['kl']:.2f}  | {r['shape']:.2f}  {r['scale']:.2f}"
              f"   {r['rotation']:.2f}     {r['posX']:.2f}  {r['posY']:.2f}   {r['position']:.2f}")
    for r in table:
        if r["rotation"] >= 0.2:
            print(f"  z{r['latent'] + 1}: periods of the rotation response per full turn (square, oval, heart) = "
                  f"{r['rot_periods']}, amplitudes = {[round(a, 2) for a in r['rot_amplitude']]}")
    print(f"  mean over latents of the largest single-factor fraction: {entry['purity_separate']:.2f} "
          f"(x and y separate), {entry['purity_joint_pos']:.2f} (position as one factor)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--betas", default="4,1")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", default="figures/fig7ab_reproduced.png", help="figure of the first seed; '' for none")
    ap.add_argument("--data", default=None)
    ap.add_argument("--summary-only", action="store_true", help="print saved measurements without weights or data")
    args = ap.parse_args()

    betas = [float(b) for b in args.betas.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.summary_only:
        with open(SUMMARY) as f:
            summary = json.load(f)
        for seed in seeds:
            for beta in betas:
                entry = summary[f"beta{beta:g}_s{seed}"]
                print_model(beta, seed, entry["table"], np.array(entry["kl_per_dim"]), entry, entry)
        return

    missing = [f"vae_beta{beta:g}_s{seed}.pt" for seed in seeds for beta in betas
               if not os.path.exists(os.path.join(args.results, f"vae_beta{beta:g}_s{seed}.pt"))]
    if missing:
        ap.error("Missing weights: " + ", ".join(missing)
                 + ". Run bash run_all.sh, or use --summary-only to read saved measurements.")
    device = get_device()
    imgs, classes = load_dsprites(args.data)
    os.makedirs(os.path.dirname(SUMMARY), exist_ok=True)
    summary = {}
    if os.path.exists(SUMMARY):
        with open(SUMMARY) as f:
            summary = json.load(f)

    for seed in seeds:
        paths = [os.path.join(args.results, f"vae_beta{b:g}_s{seed}.pt") for b in betas]
        draw = bool(args.out) and seed == seeds[0]
        if draw:
            fig = plt.figure(figsize=(4.0 * len(betas) + 1, 8.6))
            outer = fig.add_gridspec(1, len(betas), wspace=0.12, left=0.06, right=0.98, top=0.9, bottom=0.05)
        for j, (beta, path) in enumerate(zip(betas, paths)):
            model = BetaVAE().to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            columns, learnt_var, kl, table, rec = analyze(model, imgs, classes, device)

            if draw:
                prefix = ("A: " if j == 0 else "B: ") if len(betas) > 1 else ""
                name = r"VAE ($\beta$=1)" if beta == 1 else r"$\beta$-VAE ($\beta$=%g)" % beta
                draw_panel(fig, outer[0, j], columns, prefix + name)

            entry = {"learnt_var": learnt_var.tolist(), "kl_per_dim": kl.tolist(), "kl_total": float(kl.sum()),
                     "n_informative": len(table), "table": table, **rec,
                     "purity_separate": float(np.mean([r["best_separate"] for r in table])) if table else 0.0,
                     "purity_joint_pos": float(np.mean([r["best_joint_pos"] for r in table])) if table else 0.0}
            summary[f"beta{beta:g}_s{seed}"] = entry
            print_model(beta, seed, table, kl, rec, entry)
        if draw:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            plt.savefig(args.out, dpi=170)
            plt.close(fig)
            print("saved", args.out)

    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
