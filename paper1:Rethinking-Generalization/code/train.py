"""Train one network on one randomization test and write its curves to <out>/<tag>.json."""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from data import load_cifar10, preprocess, randomize
from models import ARCHS, build, count_params, glorot_init

# Appendix A of the paper. The batch size is not given there; 128 matches the spacing in Fig. 1a.
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.95   # per epoch
MODES = ["true", "corrupt", "shuffled_pixels", "random_pixels", "random_values", "gaussian"]


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
    model.train()
    return correct / len(x), loss / len(x)


def corrupt_test_labels(y, p, seed):
    # Added in Phase 4: the authors' demo corrupts the test labels too. Only used for evaluation.
    g = torch.Generator().manual_seed(seed + 1000)
    replace = torch.rand(len(y), generator=g) < p
    return torch.where(replace, torch.randint(0, 10, (len(y),), generator=g), y)


def save(obj, path):
    # write to a temp file first, so a run killed mid-write cannot leave a broken file
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="inception", choices=list(ARCHS))
    ap.add_argument("--mode", default="true", choices=MODES)
    ap.add_argument("--p", type=float, default=0.0, help="label corruption probability (mode=corrupt)")
    ap.add_argument("--init", default="pytorch", choices=["pytorch", "glorot"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=os.environ.get("CIFAR10_NPZ"), help="path to a cifar10.npz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag
    if tag is None:
        p_part = f"_p{args.p:.2f}" if args.mode == "corrupt" else ""
        tag = f"{args.arch}_{args.mode}{p_part}_s{args.seed}"
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, tag + ".json")
    ckpt_path = os.path.join(args.out, tag + ".ckpt")

    torch.manual_seed(args.seed)
    device = get_device()
    lr0 = 0.1 if args.arch == "inception" else 0.01

    x_train, y_train, x_test, y_test = load_cifar10(args.data)
    y_clean = torch.from_numpy(y_train).long()
    x_train, y_train, x_test, y_test = randomize(preprocess(x_train), y_clean, preprocess(x_test),
                                                 torch.from_numpy(y_test).long(), args.mode, args.p, args.seed)
    frac_changed = None
    y_test_noisy = y_test
    if args.mode == "corrupt":
        frac_changed = (y_train != y_clean).float().mean().item()
        y_test_noisy = corrupt_test_labels(y_test, args.p, args.seed)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test, y_test_noisy = x_test.to(device), y_test.to(device), y_test_noisy.to(device)

    model = build(args.arch)
    if args.init == "glorot":
        glorot_init(model)
    model.to(device)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr0, momentum=MOMENTUM)

    n = len(x_train)
    steps_per_epoch = n // BATCH_SIZE
    log = {"args": vars(args), "tag": tag, "params": count_params(model), "lr0": lr0,
           "frac_labels_changed": frac_changed, "steps_per_epoch": steps_per_epoch,
           "step_loss": [], "epoch": [], "train_acc": [], "train_loss": [],
           "test_acc": [], "test_loss": [], "test_acc_noisy": []}
    print(f"[{tag}] device={device} params={log['params']:,} lr0={lr0} labels changed={frac_changed}", flush=True)

    step, start_epoch = 0, 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        log, step, start_epoch = ck["log"], ck["step"], ck["epoch"]
        torch.set_rng_state(ck["rng"].cpu())
        print(f"[{tag}] resumed from epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        lr = lr0 * LR_DECAY ** epoch
        for group in opt.param_groups:
            group["lr"] = lr

        perm = torch.randperm(n, device=device)
        losses = []
        for i in range(steps_per_epoch):
            idx = perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.detach())
            step += 1

        # mean loss of every 25 steps, for the learning curves
        losses = torch.stack(losses).cpu().numpy()
        window = 25
        for j in range(0, len(losses) - window + 1, window):
            log["step_loss"].append([step - len(losses) + j + window, float(losses[j:j + window].mean())])

        train_acc, train_loss = evaluate(model, x_train, y_train)
        test_acc, test_loss = evaluate(model, x_test, y_test)
        test_acc_noisy = evaluate(model, x_test, y_test_noisy)[0] if args.mode == "corrupt" else test_acc
        log["epoch"].append(epoch + 1)
        log["train_acc"].append(train_acc)
        log["train_loss"].append(train_loss)
        log["test_acc"].append(test_acc)
        log["test_loss"].append(test_loss)
        log["test_acc_noisy"].append(test_acc_noisy)
        print(f"[{tag}] epoch {epoch + 1:3d}  lr {lr:.4f}  train loss {train_loss:.4f}  train acc {train_acc:.4f}  "
              f"test acc {test_acc:.4f}  ({time.time() - t0:.1f}s)", flush=True)

        with open(log_path, "w") as f:
            json.dump(log, f)
        save({"model": model.state_dict(), "opt": opt.state_dict(), "log": log, "step": step,
              "epoch": epoch + 1, "rng": torch.get_rng_state()}, ckpt_path)
        if train_acc >= 1.0:
            break

    log["final_test_err"] = 1 - log["test_acc"][-1]
    with open(log_path, "w") as f:
        json.dump(log, f)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"[{tag}] done, test error {log['final_test_err']:.4f}", flush=True)


if __name__ == "__main__":
    main()
