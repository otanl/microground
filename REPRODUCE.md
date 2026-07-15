# Reproduction guide (anonymous supplement)

Companion to the paper *"Input Pathways Shape Few-Shot, Not Zero-Shot, Binding in Tiny Transformers: A Fully-Enumerable Study"* (arXiv:2607.04926).
Everything re-aggregates from committed JSONL manifests in `results/mg/` **without retraining**;
the training commands below regenerate those manifests from scratch.

## Environment
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt   # torch, numpy, scikit-learn, matplotlib
$env:PYTHONUTF8=1
```
CPU is sufficient (models are ~6K params). Core: `src/mg/` (`world.py`, `multiworld.py`, `data.py`,
`model.py`, `train.py`, `stats.py`). Runner: `scripts/mg_run.py` (parallel, resume-aware).

## Experiments → commands → manifest
`P=.\.venv\Scripts\python.exe`; all runs append to `results/mg/<name>.jsonl` (config, seeds, full
trajectories per run).

| ID | What | Command (abbrev.; `--workers 14` for parallelism) |
|----|------|---------|
| E1 | control validation / ceiling | `$P scripts/mg_run.py --name cf_first --task counterfactual --conditions text_minimal uninformative_state scrambled_state text_only state_factored state_perceptual --seeds 8 --epochs 200` |
| E2 | transitions: all memorize | `... --name cf_route_trans --task counterfactual --split transition_frac:0.3 --conditions text_minimal uninformative_state text_only state_factored state_perceptual --seeds 10 --epochs 1000` |
| E3/E4 | zero-shot binding + dynamics | `... --name bind_holdout_route --task bind --split bind:3 --conditions text_minimal uninformative_state text_only state_factored state_perceptual state_perceptual_hard state_onehot_shared --seeds 20 --epochs 500` |
| E5 | exact Bayes ceilings | `$P scripts/mg_ceilings.py` → `results/mg/ceilings.json` |
| E6 | failure anatomy (probe) | `... --name bind_mech --task bind --split bind:3 --conditions text_only state_factored state_perceptual state_perceptual_hard state_onehot_shared --seeds 10 --epochs 500 --checkpoints 5,10,15,20,50 --save_dir models/mg/bind_mech` then `$P scripts/mg_probe.py --models_dir models/mg/bind_mech` |
| E7 | weight-decay null | `... --name cf_wd{W} --task counterfactual --split transition_frac:0.3 --conditions state_factored --wd {W} --epochs 4000 --seeds 5` for W in 0.01,0.1,0.3,1.0 |
| E8 | capacity sweep | `... --name bind_cap_h{H}_L{L} --task bind --split bind:3 --conditions text_only state_factored state_perceptual --hidden_size {H} --num_layers {L} --seeds 8 --epochs 500` for (48,1),(48,2),(96,2) |
| E9/E12 | k-shot dose-response + 2x2 | `... --name bind_kshot_f{F} --task bind --split bind_kshot:3:{F} --conditions text_only state_factored state_perceptual state_perceptual_hard state_onehot_shared --seeds 20 --epochs 500` for F in 0.02,0.05,0.1,0.2 |
| E11 | confirmatory (fresh holdout) | as E9/E12 with `--name bindC1_kshot_f{F} --split bind_kshot:1:{F}` |
| E13a | causal input-intervention | `$P scripts/mg_causal.py --models_dir models/mg/bind_mech` → `results/mg/bind_causal.json` |
| E13b | nonlinear-decoder control | `$P scripts/mg_nonlinear.py` |
| E13c | larger world (3 objects) | `... --name bind3_kshot_f{F} --task bind3 --split bind_kshot:5:{F} --conditions text_only state_factored state_perceptual state_perceptual_hard state_onehot_shared --seeds 10 --epochs 500` for F in 0.05,0.1 |
| E13d | lr sensitivity | `... --name bind_lr{LR} --task bind --split bind:3 --conditions text_only state_factored state_perceptual --lr {LR} --seeds 10 --epochs 500` for LR in 3e-4,3e-3 |
| E15 | dimension-matched readability control | `... --name bind_dimctrl_f{F} --task bind --split bind_kshot:3:{F} --conditions state_factored state_perceptual state_perceptual_hard state_perceptual_hard16 --seeds 10 --epochs 500` for F in 0.05,0.1,0.2 |
| E16 | graded readability sweep (fixed dim16) | `... --name bind_readsweep_f{F} --task bind --split bind_kshot:3:{F} --conditions state_factored state_perceptual state_perceptual_r085 state_perceptual_r073 state_perceptual_hard16 --seeds 10 --epochs 500` for F in 0.1,0.2 |

## Analysis / figures (re-aggregate, no retraining)
- `scripts/mg_analyze.py --name <run> --reference <cond>` — converged metric, CI, Wilcoxon+Holm, peak.
- `scripts/mg_doseresp.py` — dose-response table + `figs/bind_doseresp.png` (Fig. 4).
- `scripts/mg_dynamics.py --name bind_holdout_route` — transient collapse + `figs/bind_holdout_route_dynamics.png` (Fig. 1).
- `scripts/mg_probe_fig.py` — `figs/bind_probe_anatomy.png` (Fig. 3).
- `scripts/mg_smoke_test.py` — sanity/validation of the enumeration and controls.

## Notes
- Primary metric is **converged** balanced accuracy (mean over last 10% of evals), not peak (see paper §4).
- The paper's Results sections (experiments E1–E14) describe each result; the table above maps them to commands and manifests.
- Legacy `src/micro_ground/` is the pre-redesign code, referenced only by `mg_smoke_test.py` to demonstrate the non-exhaustive-eval bug it fixes; it is not used by the paper's pipeline.
