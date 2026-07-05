r"""
Exact per-route solvability ceilings (analysis A).

For a given (task, split, condition) we compute the Bayes-optimal balanced accuracy on the
TEST queries: group test queries by the exact model-visible input (token ids + state vector),
and within each group the best any predictor can do is the majority target frequency. This is
an *a priori, model-agnostic upper bound* on what any learner could emit for those inputs --
a channel property, not a property of a trained model (contrast: Gross et al. 2024 prove
posterior lower bounds for one trained network).

The point of the analysis: it separates two failure classes.
  * ceiling == chance  -> the information is absent; failure is FORCED (e.g. text_minimal).
  * ceiling == 1.0 but model at/below chance -> the input determines the answer, yet training
    provides no constraint on the held-out mapping; failure is an INDUCTIVE-BIAS failure,
    not an information failure (e.g. all informative routes on held-out transitions/bindings).

Usage:
    $env:PYTHONUTF8=1
    .\.venv\Scripts\python.exe scripts\mg_ceilings.py
"""
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mg import FactoredWorld, CONDITIONS, Vocab, build_examples_from_queries
from mg.multiworld import TwoObjectWorld
from mg.train import parse_holdout

CONDS = ["text_minimal", "uninformative_state", "scrambled_state",
         "text_only", "state_factored", "state_perceptual",
         "state_perceptual_hard", "state_onehot_shared"]


def ceiling(world, cond_key, queries, vocab):
    """Bayes-optimal balanced accuracy over the given queries under this condition."""
    cond = CONDITIONS[cond_key]
    ex = build_examples_from_queries(world, cond, queries, vocab)
    # group by exact visible input; keep per-factor bookkeeping for balanced accuracy
    groups = defaultdict(list)  # input-key -> list of (factor, target)
    for e in ex:
        state_key = tuple(round(v, 6) for v in e.state) if e.state is not None else None
        groups[(tuple(e.input_ids), state_key)].append((e.factor, e.target_name))
    hit = Counter()
    tot = Counter()
    for _, items in groups.items():
        # Bayes: within an input group, predict the majority target
        best = Counter(t for _, t in items).most_common(1)[0][0]
        for f, t in items:
            tot[f] += 1
            hit[f] += int(t == best)
    per_factor = {f: hit[f] / tot[f] for f in tot}
    return sum(per_factor.values()) / len(per_factor)


def main():
    out = {}
    settings = [
        ("counterfactual", FactoredWorld(), "transition_frac:0.3"),
        ("counterfactual", FactoredWorld(), "random"),
        ("bind", TwoObjectWorld(), "bind:3"),
        ("bind", TwoObjectWorld(), "random"),
    ]
    for task, world, split in settings:
        vocab = Vocab(world)
        holdout = parse_holdout(split)
        sp = world.split_queries(task, split_seed=0, holdout=holdout)
        row = {}
        for ck in CONDS:
            row[ck] = ceiling(world, ck, sp["test"], vocab)
        out[f"{task}/{split}"] = row
        print(f"\n=== {task}  split={split}  (test queries: {len(sp['test'])}) ===")
        for ck, v in row.items():
            print(f"  {ck:22s} ceiling = {v:.3f}")

    os.makedirs("results/mg", exist_ok=True)
    with open("results/mg/ceilings.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nsaved results/mg/ceilings.json")


if __name__ == "__main__":
    main()
