"""Train a VAE (beta=1) and a beta-VAE (beta=4) on dSprites with the paper's optimiser
(Adagrad, lr 1e-2).  Both models see the same minibatches (one process, memory friendly).
Batch size and number of epochs are not given in the paper; defaults here: 256 and 30.
"""
import argparse, json, os, time
import numpy as np
import torch

from data import load_dsprites
from model import BetaVAE, loss_terms


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", default="1,4")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--optimizer", default="adagrad", choices=["adagrad", "adam"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    dev = get_device(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    imgs, _ = load_dsprites(args.data)
    x_all = torch.from_numpy(imgs).to(dev)              # uint8 on device, (N,64,64)
    n = x_all.shape[0]; bs = args.batch_size; steps = n // bs
    betas = [float(b) for b in args.betas.split(",")]
    models, opts, logs = {}, {}, {}
    for b in betas:
        torch.manual_seed(args.seed)                        # same initialisation for every beta
        m = BetaVAE().to(dev)
        models[b] = m
        opts[b] = (torch.optim.Adagrad if args.optimizer == "adagrad" else torch.optim.Adam)(m.parameters(), lr=args.lr)
        logs[b] = dict(beta=b, args=vars(args), epoch=[], recon=[], kl=[], kl_per_dim=[], loss=[], time=[])
    os.makedirs(args.out, exist_ok=True)
    print(f"device={dev} N={n} steps/epoch={steps} betas={betas}", flush=True)
    start_ep = 0
    ckpt_path = os.path.join(args.out, f"train_s{args.seed}.ckpt")
    if os.path.exists(ckpt_path):                    # resume after an interruption
        ck = torch.load(ckpt_path, map_location=dev)
        for b in betas:
            models[b].load_state_dict(ck["models"][b]); opts[b].load_state_dict(ck["opts"][b]); logs[b] = ck["logs"][b]
        start_ep = ck["epoch"]; torch.set_rng_state(ck["rng"])
        print(f"resumed from epoch {start_ep}", flush=True)
    for ep in range(start_ep, args.epochs):
        t0 = time.time(); perm = torch.randperm(n, device=dev)
        acc = {b: torch.zeros(3 + 10, device=dev) for b in betas}      # recon, kl, loss, kl per dim
        for i in range(steps):
            x = x_all[perm[i * bs:(i + 1) * bs]].float()
            for b in betas:
                logits, mu, logvar = models[b](x)
                recon, kl = loss_terms(logits, x, mu, logvar)
                loss = (recon + b * kl.sum(1)).mean()
                opts[b].zero_grad(set_to_none=True); loss.backward(); opts[b].step()
                with torch.no_grad():
                    acc[b][0] += recon.mean(); acc[b][1] += kl.sum(1).mean(); acc[b][2] += loss
                    acc[b][3:] += kl.mean(0)
        dt = time.time() - t0
        for b in betas:
            a = (acc[b] / steps).cpu().numpy(); L = logs[b]
            L["epoch"].append(ep + 1); L["recon"].append(float(a[0])); L["kl"].append(float(a[1]))
            L["loss"].append(float(a[2])); L["kl_per_dim"].append(a[3:].tolist()); L["time"].append(dt)
            print(f"[beta={b:g}] ep {ep+1:3d} recon {a[0]:.2f} KL {a[1]:.2f} loss {a[2]:.2f} "
                  f"KL/dim {np.round(a[3:], 2).tolist()} ({dt:.0f}s)", flush=True)
            json.dump(L, open(os.path.join(args.out, f"log_beta{b:g}_s{args.seed}.json"), "w"))
            torch.save(models[b].state_dict(), os.path.join(args.out, f"vae_beta{b:g}_s{args.seed}.pt"))
        torch.save(dict(models={b: models[b].state_dict() for b in betas}, opts={b: opts[b].state_dict() for b in betas},
                        logs=logs, epoch=ep + 1, rng=torch.get_rng_state()), ckpt_path)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


if __name__ == "__main__":
    main()
