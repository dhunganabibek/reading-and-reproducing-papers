"""Evaluate the line between SB and LB solutions with three batch-norm choices."""
import argparse
import json
import os

import numpy as np
import torch

from data import load
from models import NETS, build
from summarize import weight_distances
from train import evaluate, get_device

BN = torch.nn.modules.batchnorm._BatchNorm
MODES = ["recompute", "interp", "recompute_norm"]


@torch.no_grad()
def recompute_bn(model, x, batch_size=1000):
    """Re-measure the batch-norm statistics on x with only the BN layers in training mode (dropout stays off)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, BN):
            m.reset_running_stats()
            m.momentum = None   # plain average over all batches
            m.train()
    for i in range(0, len(x), batch_size):
        model(x[i:i + batch_size])
    model.eval()


def scale_invariant_layers(model):
    """Find layers whose scale is approximately canceled by recalibrated batch norm."""
    names = []
    for prefix, seq in model.named_modules():
        if isinstance(seq, torch.nn.Sequential):
            children = list(seq.named_children())
            for (name, m1), (_, m2) in zip(children[:-1], children[1:]):
                if isinstance(m1, (torch.nn.Conv2d, torch.nn.Linear)) and isinstance(m2, BN):
                    names.append(f"{prefix}.{name}" if prefix else name)
    return names


def layer_norm(sd, name):
    return torch.sqrt(sd[name + ".weight"].pow(2).sum() + sd[name + ".bias"].pow(2).sum())


def normalize_layers(sd, layers):
    sd = dict(sd)
    for name in layers:
        c = 1.0 / layer_norm(sd, name)
        sd[name + ".weight"], sd[name + ".bias"] = sd[name + ".weight"] * c, sd[name + ".bias"] * c
    return sd


def interpolate(sb, lb, alpha):
    sd = {}
    for k in sb:
        if sb[k].dtype.is_floating_point:
            sd[k] = alpha * lb[k] + (1 - alpha) * sb[k]
            if k.endswith("running_var"):   # can go negative when extrapolating
                sd[k] = sd[k].clamp_min(0.0)
        else:
            sd[k] = sb[k]   # num_batches_tracked
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="F1", choices=list(NETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    device = get_device()
    x_train, y_train, x_test, y_test = load(args.net)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)
    sb = torch.load(os.path.join(args.results, f"{args.net}_SB_s{args.seed}.pt"), map_location=device)
    lb = torch.load(os.path.join(args.results, f"{args.net}_LB_s{args.seed}.pt"), map_location=device)
    model = build(args.net).to(device)
    layers = scale_invariant_layers(model)

    alphas = np.linspace(-1, 2, 61)
    out = {"alpha": alphas.tolist(),
           "layer_norms": {n: {"SB": float(layer_norm(sb, n)), "LB": float(layer_norm(lb, n))} for n in layers}}
    out["weight_distances"] = weight_distances(
        [os.path.join(args.results, f"{args.net}_{kind}_s{args.seed}.pt")
         for kind in ("init", "SB", "LB")],
        [name for name, _ in model.named_parameters()],
    )
    for mode in MODES:
        a_sd, b_sd = (normalize_layers(sb, layers), normalize_layers(lb, layers)) if mode == "recompute_norm" else (sb, lb)
        res = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        for a in alphas:
            model.load_state_dict(interpolate(a_sd, b_sd, a))
            if mode != "interp":
                recompute_bn(model, x_train)
            train_acc, train_loss = evaluate(model, x_train, y_train)
            test_acc, test_loss = evaluate(model, x_test, y_test)
            res["train_loss"].append(train_loss)
            res["train_acc"].append(train_acc)
            res["test_loss"].append(test_loss)
            res["test_acc"].append(test_acc)
            print(f"[{args.net} {mode}] alpha={a:+.2f}  train {train_loss:.4f}/{train_acc:.4f}  "
                  f"test {test_loss:.4f}/{test_acc:.4f}", flush=True)
        out[mode] = res

    with open(os.path.join(args.results, f"{args.net}_interp_s{args.seed}.json"), "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
