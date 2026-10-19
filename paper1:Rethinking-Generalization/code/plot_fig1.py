"""Draw Figure 1 from the logs in results/ and save the numbers next to it as a json summary."""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_all(results_dir):
    runs = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        with open(path) as f:
            run = json.load(f)
        runs[run["tag"]] = run
    return runs


def steps_to(run, threshold):
    """SGD steps until the training accuracy (checked after every epoch) first reaches threshold."""
    hit = np.flatnonzero(np.array(run["train_acc"]) >= threshold)
    return int((hit[0] + 1) * run["steps_per_epoch"]) if len(hit) else None


def epoch_loss(run):
    """Average training loss of each epoch, from the 25-step averages in the log."""
    steps, loss = np.array(run["step_loss"]).T
    spe = run["steps_per_epoch"]
    epoch = np.ceil(steps / spe).astype(int)
    epochs = np.unique(epoch)
    return epochs * spe, np.array([loss[epoch == e].mean() for e in epochs])


def noisy_test_err(clean_acc, p, k=10):
    """Expected error on test labels corrupted with probability p, given the accuracy on clean labels."""
    return 1 - (clean_acc * (1 - p + p / k) + (1 - clean_acc) * p / k)


def corruption_sweep(runs, arch, threshold):
    rows = []
    for run in runs.values():
        a = run["args"]
        if a["arch"] != arch or not run["tag"].endswith("_s0") or "final_test_err" not in run:
            continue
        if a["mode"] not in ("true", "corrupt"):
            continue
        p = a["p"] if a["mode"] == "corrupt" else 0.0
        acc = run["test_acc"][-1]
        rows.append({
            "p": p,
            "steps": steps_to(run, threshold) or run["epoch"][-1] * run["steps_per_epoch"],
            "train_acc": run["train_acc"][-1],
            "test_err": 1 - acc,
            "test_err_noisy_expected": noisy_test_err(acc, p),
            "test_err_noisy_measured": 1 - run["test_acc_noisy"][-1] if run.get("test_acc_noisy") else None,
            "steps_by_threshold": {str(t): steps_to(run, t) for t in (0.95, 0.99, 0.999, 1.0)},
        })
    rows.sort(key=lambda r: r["p"])
    for r in rows:
        r["time_ratio"] = r["steps"] / rows[0]["steps"]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures/fig1_reproduced.png")
    ap.add_argument("--threshold", type=float, default=0.99, help="training accuracy that counts as converged")
    args = ap.parse_args()
    runs = load_all(args.results)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
    ax_a, ax_b, ax_c, ax_c2 = axes.ravel()

    # (a) "random pixels" permutes all values of an image; the dashed curve keeps each pixel's colors together
    curves = [("inception_true_s0", "true labels", "s", "tab:blue", "-"),
              ("inception_corrupt_p1.00_s0", "random labels", "o", "tab:red", "-"),
              ("inception_shuffled_pixels_s0", "shuffled pixels", "*", "tab:green", "-"),
              ("inception_random_values_s0", "random pixels", None, "orchid", "-"),
              ("inception_random_pixels_s0", "random pixels, color kept", None, "orchid", "--"),
              ("inception_gaussian_s0", "gaussian", "D", "black", "-")]
    for tag, label, marker, color, ls in curves:
        if tag in runs:
            steps, loss = epoch_loss(runs[tag])
            ax_a.plot(steps / 1000.0, loss, label=label, color=color, marker=marker, markevery=2, ms=4, lw=1.5, ls=ls)
    ax_a.set_xlabel("thousand steps")
    ax_a.set_ylabel("average loss of the epoch")
    ax_a.set_title("(a) learning curves (Inception)")
    ax_a.set_ylim(0, 2.5)
    ax_a.set_xlim(0, 25)
    ax_a.legend(fontsize=7.5)

    # (b), (c), (c')
    summary = {}
    for arch, label, marker, color in [("inception", "Inception", "s", "black"),
                                       ("alexnet", "AlexNet", "o", "tab:green"),
                                       ("mlp1", "MLP 1x512", "*", "tab:blue")]:
        rows = corruption_sweep(runs, arch, args.threshold)
        if not rows:
            continue
        ps = [r["p"] for r in rows]
        style = dict(label=label, marker=marker, color=color, lw=2)
        ax_b.plot(ps, [r["time_ratio"] for r in rows], **style)
        ax_c.plot(ps, [r["test_err"] for r in rows], **style)
        ax_c2.plot(ps, [r["test_err_noisy_expected"] for r in rows], **style)
        summary[arch] = rows

    ax_b.set_xlabel("label corruption")
    ax_b.set_ylabel(f"time to overfit (train acc $\\geq$ {100 * args.threshold:g}%)")
    ax_b.set_title("(b) convergence slowdown")
    ax_b.legend(fontsize=8)
    ax_b.set_ylim(bottom=1.0)
    ax_c.set_title("(c) test error, clean test labels")
    ax_c2.set_title("(c') test error, test labels corrupted with the same p")
    for ax in (ax_c, ax_c2):
        ax.axhline(0.9, color="red", ls="--")
        ax.set_ylim(0.1, 1.0)
        ax.set_xlabel("label corruption")
        ax.set_ylabel("test error")
        ax.legend(fontsize=8, loc="lower right")
    for ax in (ax_b, ax_c, ax_c2):
        ax.set_xlim(0, 1.0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=200)
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=1)

    for arch, rows in summary.items():
        print(arch)
        for r in rows:
            measured = r["test_err_noisy_measured"]
            measured = "  n/a" if measured is None else f"{measured:.3f}"
            print(f"  p={r['p']:.1f}  steps={r['steps']:6d}  ratio={r['time_ratio']:.2f}  "
                  f"test err={r['test_err']:.3f}  corrupted labels: formula={r['test_err_noisy_expected']:.3f} "
                  f"measured={measured}")
    print("saved", args.out)


if __name__ == "__main__":
    main()
