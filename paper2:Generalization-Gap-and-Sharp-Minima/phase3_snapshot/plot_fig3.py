"""Draw the F1, C1, C2 panels of Figure 3 (Keskar et al. 2017) from interpolate.py output."""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--nets", default="F1,C1,C2")
    ap.add_argument("--bn", default="interp", choices=["interp", "recompute"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    nets = args.nets.split(",")
    out = args.out or f"figures/fig3_reproduced_bn-{args.bn}.png"
    fig, axes = plt.subplots(1, len(nets), figsize=(4.6 * len(nets), 3.6))
    axes = [axes] if len(nets) == 1 else axes
    for ax, net in zip(axes, nets):
        f = os.path.join(args.results, f"{net}_interp_s{args.seed}.json")
        if not os.path.exists(f):
            ax.set_title(f"{net}: missing"); continue
        d = json.load(open(f)); r = d[args.bn]; a = d["alpha"]
        ax.plot(a, r["train_loss"], "b-", label="Train"); ax.plot(a, r["test_loss"], "b--", label="Test")
        ax.set_xlabel("alpha"); ax.set_ylabel("Cross Entropy", color="b"); ax.set_xlim(-1, 2); ax.set_ylim(bottom=0)
        ax2 = ax.twinx()
        ax2.plot(a, [100 * v for v in r["train_acc"]], "r-"); ax2.plot(a, [100 * v for v in r["test_acc"]], "r--")
        ax2.set_ylabel("Accuracy", color="r"); ax2.set_ylim(0, 100)
        ax.legend(loc="upper left", fontsize=8); ax.set_title(net); ax.grid(True, ls=":")
    plt.tight_layout(); os.makedirs(os.path.dirname(out) or ".", exist_ok=True); plt.savefig(out, dpi=200)
    print("saved", out)


if __name__ == "__main__":
    main()
