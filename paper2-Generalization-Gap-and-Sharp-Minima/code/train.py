"""Train an SB or LB network with Adam, using the same initialization for each seed."""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from data import load
from models import NETS, build

# a 5,000-image batch does not fit in GPU memory for C1 and C2, so it is split into chunks
CHUNK = {"C1": 2500, "C2": 1250}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, x, y, batch_size=1000):
    model.eval()
    correct, loss = 0, 0.0
    for i in range(0, len(x), batch_size):
        out = model(x[i:i + batch_size])
        loss += F.cross_entropy(out, y[i:i + batch_size], reduction="sum").item()
        correct += (out.argmax(1) == y[i:i + batch_size]).sum().item()
    return correct / len(x), loss / len(x)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="F1", choices=list(NETS))
    ap.add_argument("--regime", default="SB", choices=["SB", "LB"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    if args.epochs < 1:
        ap.error("--epochs must be positive")

    device = get_device()
    torch.manual_seed(args.seed)
    x_train, y_train, x_test, y_test = load(args.net)
    n = len(x_train)
    batch_size = 256 if args.regime == "SB" else n // 10
    chunk = CHUNK.get(args.net, batch_size)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    model = build(args.net).to(device)   # same seed -> same starting point for SB and LB
    opt = torch.optim.Adam(model.parameters())   # default settings, lr 1e-3
    tag = f"{args.net}_{args.regime}_s{args.seed}"
    os.makedirs(args.out, exist_ok=True)
    log = {"args": vars(args), "tag": tag, "batch_size": batch_size,
           "epoch": [], "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    print(f"[{tag}] device={device} batch={batch_size} chunk={min(chunk, batch_size)} "
          f"updates/epoch={n // batch_size}", flush=True)

    init_path = os.path.join(args.out, f"{args.net}_init_s{args.seed}.pt")
    if not os.path.exists(init_path):
        save(model.state_dict(), init_path)

    start_epoch = 0
    ckpt_path = os.path.join(args.out, tag + ".ckpt")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        log, start_epoch = ck["log"], ck["epoch"]
        restore_rng(ck, device)
        print(f"[{tag}] resumed from epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad(set_to_none=True)
            # One Adam step per batch; batch norm uses each chunk's statistics.
            for j in range(0, batch_size, chunk):
                sub = idx[j:j + chunk]
                loss = F.cross_entropy(model(x_train[sub]), y_train[sub]) * (len(sub) / batch_size)
                loss.backward()
            opt.step()

        train_acc, train_loss = evaluate(model, x_train, y_train)
        test_acc, test_loss = evaluate(model, x_test, y_test)
        log["epoch"].append(epoch + 1)
        log["train_loss"].append(train_loss)
        log["train_acc"].append(train_acc)
        log["test_loss"].append(test_loss)
        log["test_acc"].append(test_acc)
        print(f"[{tag}] epoch {epoch + 1:3d}  train loss {train_loss:.5f}  train acc {train_acc:.4f}  "
              f"test loss {test_loss:.4f}  test acc {test_acc:.4f}  ({time.time() - t0:.1f}s)", flush=True)

        with open(os.path.join(args.out, tag + ".json"), "w") as f:
            json.dump(log, f)
        save({"model": model.state_dict(), "opt": opt.state_dict(), "log": log, "epoch": epoch + 1,
              **rng_state(device)}, ckpt_path)

    save(model.state_dict(), os.path.join(args.out, tag + ".pt"))
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"[{tag}] done: train acc {log['train_acc'][-1]:.4f}  test acc {log['test_acc'][-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
