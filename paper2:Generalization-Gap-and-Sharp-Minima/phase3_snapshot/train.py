"""Train one network with the small-batch (SB, 256) or large-batch (LB, 10% of the training set)
regime of Keskar et al. (2017) using ADAM, 'until the loss function ceased to improve'.

Stopping rule (our choice): stop when the full-training-set loss (eval mode) has not improved by
more than 1% relative over the last `patience_updates` parameter updates (2000 by default, i.e.
about 10 SB epochs or 200 LB epochs; a LB epoch is only 10 updates), after at least `min_epochs`;
hard cap `max_epochs` (F1 300, C1 200/400, C2 100/200 for SB/LB).  Both regimes start from the same random initialisation when the same --seed is
given (choice; the paper does not say whether SB and LB share an initialisation).
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from data import load
from models import build


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    model.eval()
    correct, loss_sum = 0, 0.0
    for i in range(0, x.shape[0], bs):
        out = model(x[i:i + bs])
        loss_sum += F.cross_entropy(out, y[i:i + bs], reduction="sum").item()
        correct += (out.argmax(1) == y[i:i + bs]).sum().item()
    return correct / x.shape[0], loss_sum / x.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="F1", choices=["F1", "C1", "C2"])
    ap.add_argument("--regime", default="SB", choices=["SB", "LB"])
    ap.add_argument("--lr", type=float, default=1e-3, help="ADAM learning rate (default values)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--min_epochs", type=int, default=20)
    ap.add_argument("--patience_updates", type=int, default=2000)
    ap.add_argument("--out", default="results")
    ap.add_argument("--data_dir", default="data")
    args = ap.parse_args()

    dev = get_device()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    xtr, ytr, xte, yte = load(args.net, args.data_dir)
    n = xtr.shape[0]
    bs = 256 if args.regime == "SB" else n // 10
    max_epochs = args.max_epochs or {("F1", "SB"): 300, ("F1", "LB"): 300, ("C1", "SB"): 200, ("C1", "LB"): 400,
                                      ("C2", "SB"): 100, ("C2", "LB"): 200}[(args.net, args.regime)]
    patience = max(10, -(-args.patience_updates // (n // bs)))      # in epochs
    xtr, ytr, xte, yte = xtr.to(dev), ytr.to(dev), xte.to(dev), yte.to(dev)

    model = build(args.net).to(dev)       # seeded -> SB and LB share this initialisation
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tag = f"{args.net}_{args.regime}_s{args.seed}"
    os.makedirs(args.out, exist_ok=True)
    log = dict(args=vars(args), tag=tag, batch_size=bs, epoch=[], train_loss=[], train_acc=[],
               test_loss=[], test_acc=[], epoch_time=[])
    print(f"[{tag}] device={dev} batch={bs} steps/epoch={n // bs} patience={patience} epochs max={max_epochs}", flush=True)

    best, best_ep, start_ep = float("inf"), 0, 0
    ckpt_path = os.path.join(args.out, tag + ".ckpt")
    if os.path.exists(ckpt_path):                    # resume after an interruption
        ck = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); log = ck["log"]
        best, best_ep, start_ep = ck["best"], ck["best_ep"], ck["epoch"]
        torch.set_rng_state(ck["rng"])
        print(f"[{tag}] resumed from epoch {start_ep}", flush=True)
    for ep in range(start_ep, max_epochs):
        t0 = time.time(); model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        tr_acc, tr_loss = evaluate(model, xtr, ytr)
        te_acc, te_loss = evaluate(model, xte, yte)
        dt = time.time() - t0
        for k, v in zip(["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "epoch_time"],
                        [ep + 1, tr_loss, tr_acc, te_loss, te_acc, dt]):
            log[k].append(v)
        print(f"[{tag}] ep {ep+1:3d} train_loss {tr_loss:.5f} train_acc {tr_acc:.4f} "
              f"test_loss {te_loss:.4f} test_acc {te_acc:.4f} ({dt:.1f}s)", flush=True)
        if tr_loss < best * 0.99:
            best, best_ep = tr_loss, ep + 1
        if ep + 1 >= args.min_epochs and ep + 1 - best_ep >= patience:
            print(f"[{tag}] loss stopped improving (best {best:.5f} at ep {best_ep}); stopping", flush=True)
            break
        json.dump(log, open(os.path.join(args.out, tag + ".json"), "w"))
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), log=log, best=best, best_ep=best_ep,
                        epoch=ep + 1, rng=torch.get_rng_state()), ckpt_path)
    log["final"] = dict(train_acc=tr_acc, test_acc=te_acc, train_loss=tr_loss, test_loss=te_loss, epochs=ep + 1)
    json.dump(log, open(os.path.join(args.out, tag + ".json"), "w"))
    torch.save(model.state_dict(), os.path.join(args.out, tag + ".pt"))
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"[{tag}] done: train_acc {tr_acc:.4f} test_acc {te_acc:.4f} after {ep+1} epochs", flush=True)


if __name__ == "__main__":
    main()
