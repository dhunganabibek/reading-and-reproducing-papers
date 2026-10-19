"""Reproduce Figure 1 (a,b,c) of Zhang et al. (2017) from the JSON logs written by train.py."""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_all(results_dir):
    runs = {}
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        d = json.load(open(f))
        runs[d["tag"]] = d
    return runs


def conv_steps(d, threshold=0.99):
    """Convergence time = SGD steps until the full-training-set accuracy (evaluated after every epoch)
    first reaches `threshold`.  Returns (steps, never_reached); if never reached, the last step is used."""
    acc = np.array(d["train_acc"])
    hit = np.flatnonzero(acc >= threshold)
    if len(hit):
        return int((hit[0] + 1) * d["steps_per_epoch"]), False
    return int(d["epoch"][-1] * d["steps_per_epoch"]), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures/fig1_reproduced.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.99, help="train-accuracy threshold defining convergence")
    args = ap.parse_args()
    runs = load_all(args.results)
    s = f"_s{args.seed}"

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    # (a) learning curves of Inception
    ax = axes[0]
    panels = [("inception_true" + s, "true labels", "s", "tab:blue"),
              (f"inception_corrupt_p1.00{s}", "random labels", "o", "tab:red"),
              ("inception_shuffled_pixels" + s, "shuffled pixels", "*", "tab:green"),
              ("inception_random_pixels" + s, "random pixels", None, "orchid"),
              ("inception_gaussian" + s, "gaussian", "D", "black")]
    for tag, label, marker, color in panels:
        if tag not in runs:
            continue
        sl = np.array(runs[tag]["step_loss"])
        ax.plot(sl[:, 0] / 1000.0, sl[:, 1], label=label, color=color, marker=marker, markevery=40, ms=5, lw=1.5)
    ax.set_xlabel("thousand steps"); ax.set_ylabel("average loss"); ax.set_title("(a) learning curves")
    ax.set_ylim(0, 2.5); ax.legend(fontsize=8)

    # (b) convergence slowdown and (c) test error vs label corruption
    archs = [("inception", "Inception", "s", "black"), ("alexnet", "AlexNet", "o", "tab:green"),
             ("mlp1", "MLP 1x512", "*", "tab:blue")]
    summary = {}
    for arch, label, marker, color in archs:
        ps, t, err, flagged = [], [], [], []
        for tag, d in runs.items():
            if d["args"]["arch"] != arch or not tag.endswith(s):
                continue
            if d["args"]["mode"] == "true":
                p = 0.0
            elif d["args"]["mode"] == "corrupt":
                p = d["args"]["p"]
            else:
                continue
            st, fl = conv_steps(d, args.threshold)
            ps.append(p); t.append(st); err.append(1 - d["test_acc"][-1]); flagged.append(fl)
        if not ps:
            continue
        o = np.argsort(ps); ps, t, err = np.array(ps)[o], np.array(t)[o], np.array(err)[o]
        flagged = np.array(flagged)[o]
        t0 = t[ps == 0.0][0] if (ps == 0.0).any() else t[0]
        axes[1].plot(ps, t / t0, label=label, marker=marker, color=color, lw=2)
        axes[2].plot(ps, err, label=label, marker=marker, color=color, lw=2)
        summary[arch] = dict(p=ps.tolist(), steps=t.tolist(), time_ratio=(t / t0).tolist(),
                             test_err=err.tolist(), not_converged=flagged.tolist())
    axes[1].set_xlabel("label corruption"); axes[1].set_ylabel(f"time to overfit (train acc $\\geq$ {100*args.threshold:g}%)"); axes[1].set_title("(b) convergence slowdown")
    axes[1].legend(fontsize=8)
    axes[2].axhline(0.9, color="red", ls="--"); axes[2].set_ylim(0, 1.0)
    axes[2].set_xlabel("label corruption"); axes[2].set_ylabel("test error"); axes[2].set_title("(c) generalization error growth")
    axes[2].legend(fontsize=8)
    for ax in axes[1:]:
        ax.set_xlim(0, 1.0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=200)
    json.dump(summary, open(os.path.splitext(args.out)[0] + "_summary.json", "w"), indent=1)
    for arch, sm in summary.items():
        print(arch)
        for p, st, r, e, nc in zip(sm["p"], sm["steps"], sm["time_ratio"], sm["test_err"], sm["not_converged"]):
            print(f"  p={p:.1f} steps={st:6d} ratio={r:.2f} test_err={e:.3f}{'  (threshold never reached)' if nc else ''}")
    print("saved", args.out)


if __name__ == "__main__":
    main()
