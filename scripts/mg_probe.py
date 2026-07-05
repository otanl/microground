r"""
Mechanistic follow-up on the binding failure (lever 4): representation-vs-behavior
dissociation across training, per route.

For each saved checkpoint of the bind_mech retrain we measure, on the HELD-OUT query type
("shape of right", factor 3):
  (a) behavior: balanced accuracy + error breakdown -- among states where left-shape !=
      right-shape, is the error the LEFT shape (systematic mis-binding) or something else?
  (b) representation: can a linear probe decode the right shape (state[3]) from the final-token
      residual? 5-fold CV logistic regression, with a Hewitt-Liang control task (fixed random
      state->label map) subtracted: selectivity = probe - control.

The key question: does decodability track the behavioral transient (E4), or does the
representation retain the answer even after behavior collapses to memorization?

Usage (after the bind_mech run finishes):
    $env:PYTHONUTF8=1
    .\.venv\Scripts\python.exe scripts\mg_probe.py --models_dir models/mg/bind_mech
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from mg import CONDITIONS, STATE_MODE, Vocab, build_examples_from_queries
from mg.multiworld import TwoObjectWorld, SHAPES
from mg.model import load_mg
from mg.train import parse_holdout, _collate

HELD_FACTOR = 3  # "shape of right"


def hidden_and_preds(model, examples, vocab, state_mode, max_len=32):
    model.eval()
    H, preds = [], []
    with torch.no_grad():
        for i in range(0, len(examples), 256):
            batch = examples[i:i + 256]
            ids, _, state = _collate(batch, vocab.pad_id, max_len, state_mode, "cpu")
            logits, h = model(ids, state, return_hidden=True)
            H.append(h.numpy())
            preds.extend(logits.argmax(dim=-1).tolist())
    return np.concatenate(H, axis=0), preds


def probe_cv(H, y, seed=0, folds=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(H, y):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(H[tr], y[tr])
        accs.append(float(clf.score(H[te], y[te])))
    return float(np.mean(accs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", default="models/mg/bind_mech")
    p.add_argument("--out", default="results/mg/bind_probe.json")
    args = p.parse_args()

    world = TwoObjectWorld()
    vocab = Vocab(world)
    sp = world.split_queries("bind", split_seed=0, holdout=parse_holdout("bind:3"))
    held_q = sp["test"]  # 256 queries, one per state
    states = [q.state for q in held_q]
    y_right = np.array([s[HELD_FACTOR] for s in states])            # probe target
    y_left = np.array([s[1] for s in states])                       # mis-binding candidate
    differing = y_right != y_left                                   # states where they differ
    # Hewitt-Liang control: fixed random label per state, matched class count
    rng = np.random.default_rng(0)
    y_control = rng.integers(0, 4, size=len(states))

    results = defaultdict(dict)
    pat = re.compile(r"bind_(?P<cond>.+)_bind-3_s(?P<seed>\d+)(?:_(?P<tag>ep\d+))?\.pt$")
    for path in sorted(glob.glob(os.path.join(args.models_dir, "*.pt"))):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        cond_key, seed, tag = m.group("cond"), int(m.group("seed")), m.group("tag") or "converged"
        cond = CONDITIONS[cond_key]
        state_mode = STATE_MODE[cond.encoder_name]
        model = load_mg(path)
        ex = build_examples_from_queries(world, cond, held_q, vocab)
        H, preds = hidden_and_preds(model, ex, vocab, state_mode)

        # behavior on held-out
        correct = np.array([p == e.target_id for p, e in zip(preds, ex)])
        acc = float(correct.mean())
        # error breakdown on states where left != right shape
        shape_tok = {s: vocab.tok2id[s] for s in SHAPES}
        pred_tok = np.array(preds)
        left_tok = np.array([shape_tok[SHAPES[v]] for v in y_left])
        misbind = float((pred_tok[differing] == left_tok[differing]).mean())
        # representation
        probe = probe_cv(H, y_right)
        control = probe_cv(H, y_control)
        results[(cond_key, tag)][seed] = {
            "acc": acc, "misbind_rate": misbind,
            "probe": probe, "control": control, "selectivity": probe - control,
        }

    # aggregate + print
    tags_order = ["ep5", "ep10", "ep15", "ep20", "ep50", "converged"]
    conds_order = ["text_only", "state_factored", "state_perceptual",
                   "state_perceptual_hard", "state_onehot_shared"]
    print(f"{'condition':18s} {'ckpt':>10s} {'behav':>7s} {'misbind':>8s} {'probe':>7s} {'ctrl':>6s} {'select':>7s}")
    print("-" * 70)
    agg = {}
    for c in conds_order:
        for t in tags_order:
            if (c, t) not in results:
                continue
            rs = list(results[(c, t)].values())
            mean = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
            agg[f"{c}/{t}"] = {"mean": mean, "n": len(rs),
                               "per_seed": {str(s): r for s, r in results[(c, t)].items()}}
            print(f"{c:18s} {t:>10s} {mean['acc']:7.3f} {mean['misbind_rate']:8.3f} "
                  f"{mean['probe']:7.3f} {mean['control']:6.3f} {mean['selectivity']:7.3f}")
        print("-" * 70)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    print(f"saved {args.out}")
    print("\nbehav = balanced acc on held-out type (chance 0.25); misbind = P(pred == LEFT shape | left!=right)")
    print("probe = 5-fold CV decoding of right-shape from final-token residual; select = probe - control")


if __name__ == "__main__":
    main()
