"""Parametric 1-D plot of Keskar et al. (2017), Figure 3: evaluate train/test cross-entropy and
accuracy at theta(alpha) = alpha * theta_LB + (1 - alpha) * theta_SB for alpha in [-1, 2].

Batch-norm statistics at an interpolated point (our choice, both variants are computed):
  'interp'   : the BN running mean/var buffers are interpolated like every other tensor
  'recompute': the BN statistics are re-estimated on the training set at each alpha
Dropout is off (eval mode) in both cases.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F

from data import load
from models import build
from train import evaluate, get_device


@torch.no_grad()
def recompute_bn(model, x, bs=1000):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats(); m.momentum = None      # cumulative average
    model.train()
    for i in range(0, x.shape[0], bs):
        model(x[i:i + bs])
    model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="F1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", default="results")
    ap.add_argument("--n_alpha", type=int, default=61)
    ap.add_argument("--bn", default="both", choices=["interp", "recompute", "both"])
    ap.add_argument("--data_dir", default="data")
    args = ap.parse_args()
    dev = get_device()
    xtr, ytr, xte, yte = load(args.net, args.data_dir)
    xtr, ytr, xte, yte = xtr.to(dev), ytr.to(dev), xte.to(dev), yte.to(dev)
    sb = torch.load(os.path.join(args.results, f"{args.net}_SB_s{args.seed}.pt"), map_location=dev)
    lb = torch.load(os.path.join(args.results, f"{args.net}_LB_s{args.seed}.pt"), map_location=dev)
    model = build(args.net).to(dev)
    alphas = np.linspace(-1, 2, args.n_alpha)
    out = dict(alpha=alphas.tolist())
    modes = ["interp", "recompute"] if args.bn == "both" else [args.bn]
    for mode in modes:
        res = dict(train_loss=[], train_acc=[], test_loss=[], test_acc=[])
        for a in alphas:
            sd = {}
            for k in sb:
                if sb[k].dtype.is_floating_point:
                    sd[k] = a * lb[k] + (1 - a) * sb[k]
                else:
                    sd[k] = sb[k]                         # num_batches_tracked
            model.load_state_dict(sd)
            if mode == "recompute":
                recompute_bn(model, xtr)
            model.eval()
            tr_acc, tr_loss = evaluate(model, xtr, ytr)
            te_acc, te_loss = evaluate(model, xte, yte)
            for k, v in zip(["train_loss", "train_acc", "test_loss", "test_acc"], [tr_loss, tr_acc, te_loss, te_acc]):
                res[k].append(v)
            print(f"[{args.net} bn={mode}] alpha={a:+.2f} train {tr_loss:.4f}/{tr_acc:.4f} test {te_loss:.4f}/{te_acc:.4f}", flush=True)
        out[mode] = res
    json.dump(out, open(os.path.join(args.results, f"{args.net}_interp_s{args.seed}.json"), "w"))


if __name__ == "__main__":
    main()
