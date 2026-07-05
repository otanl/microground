r"""
Causal (interventional) test of mis-binding: does the output functionally depend on the WRONG
object slot? Upgrades the correlational probe (decodable) to a causal claim (used).

For the held-out binding query type ``shape of right`` (factor 3 -> correct slot = state[3] = s2)
we intervene directly on the world state and read the model's output:
  - vary the RIGHT shape (slot 3) over its values, holding the rest fixed -> does the output track
    the value we set? (a correctly-bound model must; this is the correct slot)
  - vary the LEFT shape (slot 1), holding the rest fixed -> does the output track it? (a
    mis-binding model does; this is the wrong slot)
Averaged over many backgrounds, this gives the output's causal dependence on each slot. As a
positive control we run the same on a TRAINED query type (``shape of left``, factor 1 -> slot 1),
where every route should causally track the correct (left) slot.

No probe assumptions: it is a direct manipulation of the input with the output measured.

Usage (after bind_mech models are saved):
    $env:PYTHONUTF8=1
    .\.venv\Scripts\python.exe scripts\mg_causal.py --models_dir models/mg/bind_mech
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch

from mg import CONDITIONS, STATE_MODE, Vocab, build_examples_from_queries
from mg.multiworld import TwoObjectWorld, SHAPES
from mg.world import Query
from mg.model import load_mg
from mg.train import _collate


def output_shapes(model, world, cond, vocab, state_mode, queries):
    """Return the model's argmax output token id for each query."""
    ex = build_examples_from_queries(world, cond, queries, vocab)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(ex), 256):
            ids, _, st = _collate(ex[i:i + 256], vocab.pad_id, 32, state_mode, "cpu")
            preds.extend(model(ids, st).argmax(dim=-1).tolist())
    return preds


def causal_dependence(model, world, cond, vocab, state_mode, query_factor, vary_slot, rng, n_bg=120):
    """Fraction of (background, value) cases where the output equals the shape set in vary_slot."""
    shp_tok = [vocab.tok2id[s] for s in SHAPES]
    bg_states = [world.states[i] for i in rng.choice(world.num_states, size=n_bg, replace=False)]
    queries, want = [], []
    for s in bg_states:
        for v in range(4):                     # 4 shape values
            st = list(s); st[vary_slot] = v
            queries.append(Query(tuple(st), "bind", query_factor, st[query_factor],
                                 world.value_name(tuple(st), query_factor)))
            want.append(shp_tok[v])
    preds = output_shapes(model, world, cond, vocab, state_mode, queries)
    return float(np.mean([p == w for p, w in zip(preds, want)]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", default="models/mg/bind_mech")
    p.add_argument("--out", default="results/mg/bind_causal.json")
    args = p.parse_args()

    world = TwoObjectWorld(); vocab = Vocab(world)
    rng = np.random.default_rng(0)
    # only converged models (no _epN tag)
    paths = [x for x in sorted(glob.glob(os.path.join(args.models_dir, "*.pt")))
             if not re.search(r"_ep\d+\.pt$", x)]
    pat = re.compile(r"bind_(?P<cond>.+)_bind-3_s(?P<seed>\d+)\.pt$")

    acc = defaultdict(lambda: defaultdict(list))
    for path in paths:
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        ck = m.group("cond")
        if ck not in CONDITIONS:
            continue
        cond = CONDITIONS[ck]; sm = STATE_MODE[cond.encoder_name]
        model = load_mg(path)
        # held-out query "shape of right" (factor 3): correct slot 3, wrong slot 1
        acc[ck]["held_track_right"].append(causal_dependence(model, world, cond, vocab, sm, 3, 3, rng))
        acc[ck]["held_track_left"].append(causal_dependence(model, world, cond, vocab, sm, 3, 1, rng))
        # trained query "shape of left" (factor 1): correct slot 1 (positive control)
        acc[ck]["train_track_left"].append(causal_dependence(model, world, cond, vocab, sm, 1, 1, rng))
        acc[ck]["train_track_right"].append(causal_dependence(model, world, cond, vocab, sm, 1, 3, rng))

    order = ["text_only", "state_factored", "state_onehot_shared", "state_perceptual", "state_perceptual_hard"]
    print("Causal dependence of output on each object slot (mean over seeds; chance 0.25).")
    print("Held-out query = 'shape of RIGHT' (correct=right slot); trained query = 'shape of LEFT'.")
    print(f"{'route':22s} {'HELD:track-right':>16s} {'HELD:track-left':>16s} | {'TRAIN:track-left':>16s} {'TRAIN:track-right':>17s}")
    out = {}
    for ck in order:
        if ck not in acc:
            continue
        r = {k: float(np.mean(v)) for k, v in acc[ck].items()}
        out[ck] = r
        print(f"{ck:22s} {r['held_track_right']:16.2f} {r['held_track_left']:16.2f} | "
              f"{r['train_track_left']:16.2f} {r['train_track_right']:17.2f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}")
    print("Mis-binding = the output causally tracks the LEFT (wrong) slot on a 'right' query, "
          "while correctly tracking the correct slot on the trained query.")


if __name__ == "__main__":
    main()
