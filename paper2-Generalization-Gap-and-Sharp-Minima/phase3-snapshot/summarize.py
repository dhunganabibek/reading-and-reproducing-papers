"""Print the numbers quoted in the report from the saved runs: accuracies, distances from the starting
point (the paper says SB moves 3-10 times farther than LB), layer norms, and the loss along the line."""
import argparse
import json
import os

import numpy as np
import torch

from models import build

ALPHAS = (-1, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2)


def flat(sd, names):
    return torch.cat([sd[k].flatten().float().cpu() for k in names])


def weight_distances(paths, names):
    v0, vs, vl = (
        flat(torch.load(path, map_location="cpu", weights_only=True), names)
        for path in paths
    )
    return {
        "SB_from_init": (vs - v0).norm().item(),
        "LB_from_init": (vl - v0).norm().item(),
        "SB_from_LB": (vs - vl).norm().item(),
        "init_norm": v0.norm().item(),
        "SB_norm": vs.norm().item(),
        "LB_norm": vl.norm().item(),
    }


def print_distances(d):
    ds, dl = d["SB_from_init"], d["LB_from_init"]
    ratio = ds / dl if dl else float("nan")
    print(f"  ||x_s - x0|| = {ds:.1f}   ||x_l - x0|| = {dl:.1f}   "
          f"ratio = {ratio:.2f}   ||x_s - x_l|| = {d['SB_from_LB']:.1f}")
    print(f"  ||x0|| = {d['init_norm']:.1f}   ||x_s|| = {d['SB_norm']:.1f}   "
          f"||x_l|| = {d['LB_norm']:.1f}")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def print_run(name, d):
    print(f"  {name}: batch {d['batch_size']:5d}, {d['epoch'][-1]} epochs, "
          f"train acc {100 * d['train_acc'][-1]:6.2f}%  test acc {100 * d['test_acc'][-1]:5.2f}%  "
          f"train loss {d['train_loss'][-1]:.5f}  test loss {d['test_loss'][-1]:.4f}  "
          f"best test acc {100 * max(d['test_acc']):5.2f}%")


def print_line(d):
    a = np.array(d["alpha"])
    if "layer_norms" in d:
        print("  norm of each layer that is followed by batch norm (SB / LB): "
              + ", ".join(f"{v['SB']:.1f}/{v['LB']:.1f}" for v in d["layer_norms"].values()))
    for mode in ("recompute", "interp", "recompute_norm"):
        if mode not in d:
            continue
        r = d[mode]
        idx = [int(np.argmin(abs(a - v))) for v in ALPHAS]
        print(f"  loss along the line, variant '{mode}':")
        print("    alpha     " + "  ".join(f"{v:+.2f}" for v in ALPHAS))
        for key, label, scale in [("train_loss", "train loss", 1), ("test_loss", "test loss ", 1),
                                  ("train_acc", "train acc ", 100), ("test_acc", "test acc  ", 100)]:
            print(f"    {label}" + "  ".join(f"{scale * r[key][i]:5.2f}" for i in idx))
        i0, i1 = int(np.argmin(abs(a))), int(np.argmin(abs(a - 1)))
        between = slice(min(i0, i1), max(i0, i1) + 1)
        peak = int(np.argmax(r["train_loss"][between]))
        print(f"    highest train loss between the two solutions: {max(r['train_loss'][between]):.3f} "
              f"at alpha = {a[between][peak]:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--nets", default="F1,C1,C2,C1s2")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()

    for net in args.nets.split(","):
        names = [k for k, _ in build(net).named_parameters()]
        for seed in [int(s) for s in args.seeds.split(",")]:
            def path(kind, ext):
                return os.path.join(args.results, f"{net}_{kind}_s{seed}.{ext}")

            if not (os.path.exists(path("SB", "json")) and os.path.exists(path("LB", "json"))):
                continue
            sb, lb = load_json(path("SB", "json")), load_json(path("LB", "json"))
            print(f"\n=== {net} seed {seed}")
            print_run("SB", sb)
            print_run("LB", lb)
            print(f"  test accuracy SB - LB = {100 * (sb['test_acc'][-1] - lb['test_acc'][-1]):+.2f} points")

            interp_path = path("interp", "json")
            interpolation = load_json(interp_path) if os.path.exists(interp_path) else {}
            weights = [path(kind, "pt") for kind in ("init", "SB", "LB")]
            if all(os.path.exists(p) for p in weights):
                print_distances(weight_distances(weights, names))
            elif "weight_distances" in interpolation:
                print_distances(interpolation["weight_distances"])
            else:
                print("  Weight distances unavailable; run bash run_all.sh to regenerate them.")

            if os.path.exists(interp_path):
                print_line(interpolation)


if __name__ == "__main__":
    main()
