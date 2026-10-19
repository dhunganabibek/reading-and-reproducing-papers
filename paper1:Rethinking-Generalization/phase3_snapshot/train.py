"""Train one (architecture, randomization) setting of Zhang et al. (2017) Figure 1 and log curves.

Setup from Appendix A: SGD, momentum 0.9, lr 0.1 (Inception) or 0.01 (AlexNet, MLP), lr decayed
by 0.95 every epoch, no weight decay / dropout / augmentation.  Batch size is not stated in the
paper; we use 128.  Labels/pixels are randomized once before training (fixed across epochs).

Convergence ("time to overfit", Fig. 1b) = number of SGD steps until the training accuracy,
measured on the full training set in eval mode at the end of an epoch, first reaches 100%.
We also record the first epoch at which it reaches 99.9% as a robustness check.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from data import load_cifar10, preprocess, randomize
from models import build, count_params


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
    model.train()
    return correct / x.shape[0], loss_sum / x.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="inception", choices=["inception", "alexnet", "mlp1", "mlp3"])
    ap.add_argument("--mode", default="true",
                    choices=["true", "corrupt", "shuffled_pixels", "random_pixels", "gaussian"])
    ap.add_argument("--p", type=float, default=0.0, help="label corruption probability (mode=corrupt)")
    ap.add_argument("--epochs", type=int, default=120, help="max epochs")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None, help="default: 0.1 inception, 0.01 otherwise")
    ap.add_argument("--decay", type=float, default=0.95)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=os.environ.get("CIFAR10_NPZ", "data/cifar10.npz"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--stop_at_100", type=int, default=1, help="stop once train acc hits 100%")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = get_device()
    lr0 = args.lr if args.lr is not None else (0.1 if args.arch == "inception" else 0.01)
    tag = args.tag or f"{args.arch}_{args.mode}" + (f"_p{args.p:.2f}" if args.mode == "corrupt" else "") + f"_s{args.seed}"
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, tag + ".json")

    xtr, ytr, xte, yte = load_cifar10(args.data)
    xtr, xte = preprocess(xtr), preprocess(xte)
    ytr, yte = torch.from_numpy(ytr).long(), torch.from_numpy(yte).long()
    xtr, ytr, xte, yte = randomize(xtr, ytr, xte, yte, args.mode, args.p, seed=args.seed)
    frac_changed = None
    if args.mode == "corrupt":
        frac_changed = (ytr != torch.from_numpy(load_cifar10(args.data)[1]).long()).float().mean().item()
    xtr, ytr, xte, yte = xtr.to(dev), ytr.to(dev), xte.to(dev), yte.to(dev)

    model = build(args.arch).to(dev)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr0, momentum=args.momentum)
    n = xtr.shape[0]
    steps_per_epoch = n // args.batch_size          # drop last partial batch
    log = dict(args=vars(args), tag=tag, lr0=lr0, params=count_params(model), frac_labels_changed=frac_changed,
               steps_per_epoch=steps_per_epoch, step_loss=[], epoch=[], train_acc=[], train_loss=[],
               test_acc=[], test_loss=[], epoch_time=[], steps_to_100=None, steps_to_999=None)
    print(f"[{tag}] device={dev} params={log['params']:,} lr0={lr0} frac_changed={frac_changed}", flush=True)

    step, start_ep = 0, 0
    ckpt_path = os.path.join(args.out, tag + ".ckpt")
    if os.path.exists(ckpt_path):                    # resume after an interruption
        ck = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); log = ck["log"]
        step, start_ep = ck["step"], ck["epoch"]
        torch.set_rng_state(ck["rng"])
        print(f"[{tag}] resumed from epoch {start_ep}", flush=True)
    for ep in range(start_ep, args.epochs):
        t0 = time.time()
        lr = lr0 * args.decay ** ep
        for g in opt.param_groups:
            g["lr"] = lr
        perm = torch.randperm(n, device=dev)
        model.train()
        run = []
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            out = model(xtr[idx])
            loss = F.cross_entropy(out, ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run.append(loss.detach())
            step += 1
        run = torch.stack(run).cpu().numpy()
        # average loss over windows of 25 steps (cheap, dense enough for Fig. 1a)
        w = 25
        for j in range(0, len(run) - w + 1, w):
            log["step_loss"].append([step - len(run) + j + w, float(run[j:j + w].mean())])
        tr_acc, tr_loss = evaluate(model, xtr, ytr)
        te_acc, te_loss = evaluate(model, xte, yte)
        dt = time.time() - t0
        log["epoch"].append(ep + 1); log["train_acc"].append(tr_acc); log["train_loss"].append(tr_loss)
        log["test_acc"].append(te_acc); log["test_loss"].append(te_loss); log["epoch_time"].append(dt)
        if log["steps_to_999"] is None and tr_acc >= 0.999:
            log["steps_to_999"] = step
        if log["steps_to_100"] is None and tr_acc >= 1.0:
            log["steps_to_100"] = step
        print(f"[{tag}] ep {ep+1:3d} lr {lr:.4f} train_loss {tr_loss:.4f} train_acc {tr_acc:.4f} "
              f"test_acc {te_acc:.4f} ({dt:.1f}s)", flush=True)
        json.dump(log, open(out_path, "w"))
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), log=log, step=step, epoch=ep + 1,
                        rng=torch.get_rng_state()), ckpt_path)
        if args.stop_at_100 and tr_acc >= 1.0:
            break
    log["final_train_acc"] = log["train_acc"][-1]; log["final_test_err"] = 1 - log["test_acc"][-1]
    json.dump(log, open(out_path, "w"))
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"[{tag}] done: steps_to_100={log['steps_to_100']} test_err={log['final_test_err']:.4f}", flush=True)


if __name__ == "__main__":
    main()
