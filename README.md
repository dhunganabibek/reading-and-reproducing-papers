# CS 6362 Assignment 2

Three paper reports and their PyTorch reproductions, following [assignment2.pdf](assignment2.pdf).

| Paper                                                                            | Required figure      | Report                                                               | Reproduction instructions                                                       |
| -------------------------------------------------------------------------------- | -------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Zhang et al., *Understanding deep learning requires rethinking generalization* | Figure 1a–c         | [PDF](./paper1:Rethinking-Generalization/report/report.pdf)           | [Code and notebook](./paper1:Rethinking-Generalization/code/README.md)           |
| Keskar et al., *On Large-Batch Training for Deep Learning*                     | Figure 3: F1, C1, C2 | [PDF](./paper2:Generalization-Gap-and-Sharp-Minima/report/report.pdf) | [Code and notebook](./paper2:Generalization-Gap-and-Sharp-Minima/code/README.md) |
| Higgins et al., *beta-VAE*                                                     | Figure 7A–B         | [PDF](<./paper3: beta-VAE/report/report.pdf>)                         | [Code and notebook](<./paper3: beta-VAE/code/README.md>)                         |

Each report contains the problem statement, methodology, answers to the paper-specific questions,
original and reproduced figures, failures and corrections, Phase 4 comparison, and a critical review
followed by comparison with the OpenReview reviews. Supporting diagnostics are kept where they explain
a mismatch or answer an assignment question.

## Shared LLM conversations

- [Paper 1](https://chatgpt.com/share/e/6ad70878-aa80-83e9-a81a-71775854baf0)
- [Paper 2](https://chatgpt.com/share/e/6ad7087b-6704-83e8-9168-a7a5151d61f2)
- [Paper 3](https://chatgpt.com/share/e/6ad7087d-51d0-83e9-8f59-be137ca3d921)

## Run a reproduction

Create a Python environment, then enter the paper's `code/` directory. For example:

```bash
python3 -m venv .venv
source .venv/bin/activate
cd 'paper1:Rethinking-Generalization/code'
pip install -r requirements.txt
jupyter notebook reproduce_fig1.ipynb
```

All three notebooks work from the committed results by default. Papers 1 and 2 redraw their figures
from JSON measurements. Paper 3 displays its saved figures and latent measurements; regenerating them
requires trained weights. Set `RUN_TRAINING = True` in a notebook, or run `bash run_all.sh` in its
`code/` directory, to generate the required runs. Completed runs are skipped. To repeat all training, use a working copy without saved result directories
or checkpoints. Run one training job at a time.

Datasets, model weights, checkpoints, environments and build caches are ignored. Training downloads
the data when needed and recreates the weights. JSON results, figures and final report PDFs are kept.
Each paper's README describes its dataset paths and reproduction choices.

To rebuild a report, install a LaTeX distribution with `latexmk`, then run `make` in its `report/`
directory. The `paper/` directories contain the source papers. The `phase3_snapshot/` directories
preserve the saved earlier implementations; use `code/` for the final reproduction.
