"""Print the values plotted in the paper's Figure 1 next to my runs (the tables of the report)."""
import json
import os

import numpy as np

from plot_fig1 import epoch_loss, load_all, steps_to

with open("paper_fig1_values.json") as f:
    paper = json.load(f)
with open("figures/fig1_reproduced_summary.json") as f:
    ours = json.load(f)
runs = load_all("results")


def fmt(x, spec):
    return "n/a" if x is None else format(x, spec)


def first_below(steps, loss, threshold=0.1):
    i = np.flatnonzero(np.array(loss) < threshold)
    return int(np.array(steps)[i[0]]) if len(i) else None


print("Panels (b) and (c): time to overfit relative to p=0, test error")
print(f"{'network':10s} {'p':>4s} | {'paper':>6s} {'mine':>6s} | {'paper':>6s} {'clean':>6s} "
      f"{'corrupted (formula / measured)':>31s}")
for arch, rows in ours.items():
    for r in rows:
        i = round(r["p"] * 10)
        print(f"{arch:10s} {r['p']:4.1f} | {paper['b'][arch][i]:6.2f} {r['time_ratio']:6.2f} | "
              f"{paper['c'][arch][i]:6.3f} {r['test_err']:6.3f} "
              f"{r['test_err_noisy_expected']:20.3f} / {fmt(r['test_err_noisy_measured'], '.3f')}")

print("\nPanel (a), Inception: first-epoch loss, steps until the epoch loss is below 0.1, "
      "and that time relative to true labels")
print(f"{'setting':27s} | {'paper':>6s} {'mine':>6s} | {'paper':>6s} {'mine':>6s} | {'paper':>6s} {'mine':>6s}")
curves = {"true labels": "inception_true_s0",
          "random labels": "inception_corrupt_p1.00_s0",
          "shuffled pixels": "inception_shuffled_pixels_s0",
          "random pixels": "inception_random_values_s0",
          "gaussian": "inception_gaussian_s0",
          "random pixels, color kept": "inception_random_pixels_s0"}
ref_paper = first_below(paper["a"]["true labels"]["steps"], paper["a"]["true labels"]["loss"])
ref_mine = first_below(*epoch_loss(runs["inception_true_s0"]))
for name, tag in curves.items():
    steps, loss = epoch_loss(runs[tag])
    mine = first_below(steps, loss)
    p_first = p_steps = p_ratio = None
    if name in paper["a"]:
        pc = paper["a"][name]
        p_first = pc["loss"][0]
        p_steps = first_below(pc["steps"], pc["loss"])
        p_ratio = p_steps / ref_paper
    print(f"{name:27s} | {fmt(p_first, '6.2f'):>6s} {loss[0]:6.2f} | {fmt(p_steps, '6d'):>6s} {mine:6d} | "
          f"{fmt(p_ratio, '6.2f'):>6s} {mine / ref_mine:6.2f}")

if os.path.exists("results_ablation/inception_true_glorot.json"):
    abl = load_all("results_ablation")
    print("\nInitialization test: first-epoch loss, steps to 99% training accuracy, test error")
    for label, run in [("Inception, PyTorch default", runs["inception_true_s0"]),
                       ("Inception, Glorot uniform", abl["inception_true_glorot"]),
                       ("MLP, PyTorch default", runs["mlp1_true_s0"]),
                       ("MLP, Glorot uniform", abl["mlp1_true_glorot"]),
                       ("MLP random labels, PyTorch default", runs["mlp1_corrupt_p1.00_s0"]),
                       ("MLP random labels, Glorot uniform", abl["mlp1_corrupt_p1.00_glorot"])]:
        print(f"  {label:36s} {epoch_loss(run)[1][0]:5.2f}  {fmt(steps_to(run, 0.99), '6d'):>6s}  "
              f"{1 - run['test_acc'][-1]:.3f}")
    print(f"  {'paper, Inception true labels':36s} {paper['a']['true labels']['loss'][0]:5.2f}")
