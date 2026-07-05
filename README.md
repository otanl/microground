# MicroGround

Code and data for the paper **"Input Pathways Shape How, Not Whether, Tiny Transformers Learn to
Bind: A Fully-Enumerable Study."**

MicroGround is a finite, fully-enumerable factored world for studying how the *input pathway*
(symbolic tokens vs. a clean per-factor "oracle" code vs. entangled or one-hot perceptual codes)
affects compositional **binding** in ~6–10K-parameter transformers. Because the worlds are small
(128–729 states), every model is evaluated on the *entire* input space (zero sampling variance),
every informative route is information-matched (exact Bayes ceiling = 1.0), and the full battery of
probes and interventions is tractable.

## What's here

| Path | Contents |
|---|---|
| `src/mg/` | Core: enumerable worlds (`world.py`, `multiworld.py`), route encoders + metrics (`data.py`), model (`model.py`), training (`train.py`), statistics (`stats.py`) |
| `scripts/` | Config-driven runner (`mg_run.py`) and analysis (`mg_analyze.py`, `mg_doseresp.py`, `mg_dynamics.py`, `mg_probe.py`, `mg_causal.py`, `mg_nonlinear.py`, `mg_ceilings.py`) |
| `results/mg/` | JSONL run manifests (config, seeds, full trajectories) — every table/figure re-aggregates from these without retraining |
| `paper/` | LaTeX source, bibliography, figures, and compiled PDF |
| `REPRODUCE.md` | Exact commands for every experiment (E1–E14), mapping each to its manifest |

## Quick start

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # torch, numpy, scikit-learn, matplotlib
export PYTHONUTF8=1

# validate the enumeration and controls (no training)
.venv/bin/python scripts/mg_smoke_test.py

# a small end-to-end sweep -> results/mg/demo.jsonl
.venv/bin/python scripts/mg_run.py --name demo --task bind \
    --conditions text_only state_factored state_perceptual --split bind:3 \
    --seeds 4 --epochs 200 --workers 4

# re-aggregate any manifest (converged metric, CI, paired tests) without retraining
.venv/bin/python scripts/mg_analyze.py --name bind_holdout_route --reference text_minimal
```

CPU is sufficient. See `REPRODUCE.md` for the full experiment→command→manifest map.

## Reproducing the paper

All headline numbers re-aggregate from the committed manifests in `results/mg/`. `REPRODUCE.md`
lists the exact command that produced each manifest (E1–E14). Trained model checkpoints are not
committed (regenerable via the runner with `--save_dir`); the manifests contain the full per-epoch
trajectories needed for every table and figure.

## License

Code released under the MIT License (see `LICENSE`).
