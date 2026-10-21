"""Train one model per (beta, seed) side by side on dSprites and save them to results/."""
import argparse
import json
import os
import time

import numpy as np
import torch

from data import load_dsprites
from model import BetaVAE, BatchedBetaVAE, loss_terms

# Adagrad with learning rate 1e-2 is from the paper; the batch size is not given there.
LR = 1e-2
BATCH_SIZE = 256
# A nonzero accumulator prevents the unstable first update observed at beta=16.
INIT_ACCUMULATOR = 0.1
CKPT_EVERY = 3   # epochs


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def rng_state(device):
    state = {"rng": torch.get_rng_state()}
    if device.type in ("cuda", "mps"):
        state["device_rng"] = getattr(torch, device.type).get_rng_state(device)
    return state


def restore_rng(checkpoint, device):
    if "rng" in checkpoint:
        torch.set_rng_state(checkpoint["rng"].cpu())
    if device.type in ("cuda", "mps") and "device_rng" in checkpoint:
        getattr(torch, device.type).set_rng_state(checkpoint["device_rng"].cpu(), device)
    elif "rng" not in checkpoint or device.type in ("cuda", "mps"):
        print("Older checkpoint has incomplete RNG state; resume is not an exact continuation.", flush=True)


def save(obj, path):
    # write to a temp file first, so a run killed mid-write cannot leave a broken file
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def export(model, members, logs, out):
    for i, (beta, seed) in enumerate(members):
        with open(os.path.join(out, f"log_beta{beta:g}_s{seed}.json"), "w") as f:
            json.dump(logs[i], f)
        save(model.get_member(i).state_dict(), os.path.join(out, f"vae_beta{beta:g}_s{seed}.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", default="1,4")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="results")
    ap.add_argument("--data", default=None, help="path to the dSprites npz")
    ap.add_argument("--subset", type=int, default=None, help="train on N random images only (quick tests)")
    ap.add_argument("--skip-complete", action="store_true", help="skip a group only if every model finished")
    args = ap.parse_args()
    if args.epochs < 1:
        ap.error("--epochs must be positive")

    device = get_device()
    betas = [float(b) for b in args.betas.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    members = [(b, s) for s in seeds for b in betas]
    if args.skip_complete:
        complete = True
        for beta_value, seed in members:
            log_path = os.path.join(args.out, f"log_beta{beta_value:g}_s{seed}.json")
            weights = os.path.join(args.out, f"vae_beta{beta_value:g}_s{seed}.pt")
            if not (os.path.exists(log_path) and os.path.exists(weights)):
                complete = False
                break
            with open(log_path) as f:
                epochs = json.load(f)["epoch"]
            if not epochs or epochs[-1] < args.epochs:
                complete = False
                break
        if complete:
            print(f"skip completed models: {members}")
            return

    imgs, _ = load_dsprites(args.data)
    if args.subset:
        imgs = imgs[np.random.RandomState(0).permutation(len(imgs))[:args.subset]]
    x_all = torch.from_numpy(imgs).to(device)   # uint8, (N, 64, 64)
    n = len(x_all)
    steps = n // BATCH_SIZE
    if steps == 0:
        ap.error(f"Training needs at least {BATCH_SIZE} images; increase --subset")

    model = BatchedBetaVAE(len(members))
    for i, (_, seed) in enumerate(members):
        torch.manual_seed(seed)   # for a given seed, every beta starts from the same weights
        model.set_member(i, BetaVAE())
    model.to(device)
    beta = torch.tensor([b for b, _ in members], device=device)
    # Adagrad works element-wise, so the models do not affect each other
    opt = torch.optim.Adagrad(model.parameters(), lr=LR, initial_accumulator_value=INIT_ACCUMULATOR)
    logs = [{"beta": b, "seed": s, "epoch": [], "recon": [], "kl": [], "kl_per_dim": []} for b, s in members]

    os.makedirs(args.out, exist_ok=True)
    name = "b" + "-".join(f"{b:g}" for b in betas) + "_s" + "-".join(str(s) for s in seeds)
    ckpt_path = os.path.join(args.out, f"train_{name}.ckpt")
    torch.manual_seed(seeds[0])
    start_epoch = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        logs, start_epoch = ck["logs"], ck["epoch"]
        restore_rng(ck, device)
        print(f"resumed from epoch {start_epoch}", flush=True)
    print(f"device={device} N={n} steps/epoch={steps} models={members}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        perm = torch.randperm(n, device=device)
        totals = torch.zeros(len(members), 2 + 10, device=device)   # recon, KL, KL per latent
        for i in range(steps):
            x = x_all[perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]].float().flatten(1)
            logits, mu, logvar = model(x)
            recon, kl = loss_terms(logits, x, mu, logvar)   # (S, B), (S, B, 10)
            # summing the per-model losses keeps the gradients of different models apart
            loss = (recon + beta[:, None] * kl.sum(-1)).mean(1).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                totals[:, 0] += recon.mean(1)
                totals[:, 1] += kl.sum(-1).mean(1)
                totals[:, 2:] += kl.mean(1)

        avg = (totals / steps).cpu().numpy()
        for log, row in zip(logs, avg):
            log["epoch"].append(epoch + 1)
            log["recon"].append(float(row[0]))
            log["kl"].append(float(row[1]))
            log["kl_per_dim"].append(row[2:].tolist())
            print(f"[beta={log['beta']:g} seed={log['seed']}] epoch {epoch + 1:3d}  recon {row[0]:.2f}  "
                  f"KL {row[1]:.2f}  KL per latent {[round(float(v), 2) for v in row[2:]]}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

        if (epoch + 1) % CKPT_EVERY == 0 and epoch + 1 < args.epochs:
            export(model, members, logs, args.out)
            save({"model": model.state_dict(), "opt": opt.state_dict(), "logs": logs, "epoch": epoch + 1,
                  **rng_state(device)}, ckpt_path)

    export(model, members, logs, args.out)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


if __name__ == "__main__":
    main()
