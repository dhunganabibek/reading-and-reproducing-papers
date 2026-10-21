"""Compare reconstructions of the same eight images at beta = 1, 4 and 16."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import load_dsprites
from model import BetaVAE
from train import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("figures/reconstructions.png"))
    args = parser.parse_args()
    betas = (1, 4, 16)
    paths = [args.results / f"vae_beta{beta}_s0.pt" for beta in betas]
    if any(not path.exists() for path in paths):
        parser.error("Missing seed-0 weights. Run bash run_all.sh before rebuilding reconstructions.")

    device = get_device()
    images, _ = load_dsprites()
    indices = np.random.RandomState(0).choice(len(images), 8, replace=False)
    inputs = torch.from_numpy(images[indices]).to(device).float()
    rows = [images[indices]]
    for path in paths:
        model = BetaVAE().to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        with torch.no_grad():
            means, _ = model.encode(inputs)
            rows.append(torch.sigmoid(model.decode(means)).view(-1, 64, 64).cpu().numpy())

    fig, axes = plt.subplots(4, 8, figsize=(12, 6.2))
    titles = ["data"] + [f"reconstruction, beta={beta}" for beta in betas]
    for row, (images, title) in enumerate(zip(rows, titles)):
        for column, image in enumerate(images):
            axes[row, column].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[row, column].axis("off")
        axes[row, 0].set_title(title, loc="left", fontsize=9)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print("saved", args.out)


if __name__ == "__main__":
    main()
