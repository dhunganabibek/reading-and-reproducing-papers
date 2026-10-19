"""Read the values plotted in Figure 1 of the paper from its PDF and save them to paper_fig1_values.json.

The figure is vector graphics, so the points can be recovered exactly from an SVG of page 4.
These values are only used by compare_with_paper.py, never for drawing my figure.

    python3 extract_paper_fig1.py 1611.03530.pdf    # the arXiv PDF of the paper
"""
import collections
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

# line colors in the PDF
ARCH_COLORS = {"rgb(0%, 0%, 0%)": "inception", "rgb(0%, 100%, 0%)": "alexnet", "rgb(0%, 0%, 100%)": "mlp1"}
CURVE_COLORS = {"rgb(0%, 0%, 100%)": "true labels", "rgb(100%, 0%, 0%)": "random labels",
                "rgb(0%, 100%, 0%)": "shuffled pixels", "rgb(84.312439%, 25.881958%, 95.684814%)": "random pixels",
                "rgb(0%, 0%, 0%)": "gaussian"}


def read_plots(pdf):
    """Paths of page 4, grouped by the transform of the plot they belong to."""
    svg = os.path.join(tempfile.mkdtemp(), "page4.svg")
    subprocess.run(["pdftocairo", "-svg", "-f", "4", "-l", "4", pdf, svg], check=True)
    with open(svg) as f:
        text = f.read()
    groups = collections.defaultdict(list)
    for m in re.finditer(r"<path([^>]*?)/>", text):
        attrs = m.group(1)
        d = re.search(r'\sd="([^"]+)"', attrs)
        transform = re.search(r'transform="matrix\(([^)]+)\)"', attrs)
        if not d or not transform:
            continue
        points = [(float(x), float(y)) for x, y in re.findall(r"[ML]\s*(-?[\d.]+)\s+(-?[\d.]+)", d.group(1))]
        stroke = re.search(r'stroke="([^"]+)"', attrs)
        key = tuple(round(float(v), 4) for v in transform.group(1).split(","))
        groups[key].append((stroke.group(1) if stroke else None, points))
    return [g for g in groups.values() if len(g) >= 20]


def axes_frame(paths):
    """Drawing coordinates of the left/right and bottom/top edges, from the long axis lines."""
    lines = [p for _, p in paths if len(p) == 2]
    xs = sorted({round(p[0][0], 2) for p in lines if p[0][0] == p[1][0] and abs(p[0][1] - p[1][1]) > 50})
    ys = sorted({round(p[0][1], 2) for p in lines if p[0][1] == p[1][1] and abs(p[0][0] - p[1][0]) > 50})
    return xs[0], xs[-1], ys[0], ys[-1]


def main(pdf, out="paper_fig1_values.json"):
    plots = read_plots(pdf)
    plots.sort(key=lambda g: -max(len(p) for _, p in g))   # panel (a) has the longest lines
    panel_a, panels_bc = plots[0], plots[1:]
    panels_bc.sort(key=lambda g: max(p[0][1] for _, p in g if len(p) >= 8))

    result = {}
    for name, paths, (lo, hi) in zip("bc", panels_bc, [(1.0, 4.0), (0.1, 1.0)]):
        x0, x1, y0, y1 = axes_frame(paths)
        result[name] = {}
        for stroke, p in paths:
            if len(p) >= 8 and stroke in ARCH_COLORS:
                p = np.array(p)
                x = (p[:, 0] - x0) / (x1 - x0)
                y = lo + (p[:, 1] - y0) / (y1 - y0) * (hi - lo)
                # matplotlib drops collinear points, so interpolate back onto p = 0, 0.1, ..., 1
                result[name][ARCH_COLORS[stroke]] = [round(float(v), 3) for v in np.interp(np.linspace(0, 1, 11), x, y)]

    # panel (a): x from 0 to 25 thousand steps, y from 0 to 2.5
    x0, x1, y0, y1 = axes_frame(panel_a)
    result["a"] = {}
    for stroke, p in panel_a:
        if len(p) >= 15 and stroke in CURVE_COLORS:
            p = np.array(p)
            steps = (p[:, 0] - x0) / (x1 - x0) * 25000.0
            loss = (p[:, 1] - y0) / (y1 - y0) * 2.5
            result["a"][CURVE_COLORS[stroke]] = {"steps": [round(float(v)) for v in steps],
                                                 "loss": [round(float(v), 4) for v in loss]}

    result["p"] = [round(0.1 * i, 1) for i in range(11)]
    result["note"] = ("a: training loss against SGD steps (one point per 391 steps = one epoch at batch size 128); "
                      "b: time to overfit relative to p=0; c: test error.  Extracted from the PDF of arXiv:1611.03530.")
    with open(out, "w") as f:
        json.dump(result, f, indent=1)

    for k in "bc":
        for arch, values in result[k].items():
            print(k, f"{arch:10s}", values)
    for name, c in result["a"].items():
        print("a", f"{name:16s} points every {c['steps'][1] - c['steps'][0]} steps, first loss {c['loss'][0]:.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
