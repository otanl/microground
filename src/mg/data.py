"""Conditions, encoders, dataset, and metrics for the redesigned MicroGround core.

A **Condition** is factored into two independent choices (see ``docs/RESEARCH_PLAN.md`` §5.1):

* the *text form* shown to the model (full description vs. bare question), and
* the *state encoder* that produces an optional side-channel state vector.

Keeping these orthogonal lets us build the control conditions (uninformative / scrambled
state) without touching task or text logic, and guarantees every condition is evaluated on
the identical exhaustive query set produced by :mod:`mg.world`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from .world import FactoredWorld, Query

# Fixed (run-independent) seeds for the control / perceptual codes. These are properties of
# the *condition*, not of a training run, so they must not vary with the model seed.
SCRAMBLE_SEED = 1234
PERCEPT_SEED = 5678


# ---------------------------------------------------------------------------
# State encoders  (state -> optional integer index vector, model-compatible)
# ---------------------------------------------------------------------------
class StateEncoder:
    name: str = "none"
    kind: Optional[str] = None   # None | "index" | "real"
    def encode(self, q: Query, world: FactoredWorld) -> Optional[List]:
        return None


class NoState(StateEncoder):
    name = "none"
    kind = None


class FactoredState(StateEncoder):
    """The ground-truth factor indices [c, s, p, z] -- the 'oracle' grounding channel."""
    name = "factored"
    kind = "index"
    def encode(self, q, world):
        return list(q.state)


class UninformativeState(StateEncoder):
    """Constant vector: carries no information about the state (should match text_minimal)."""
    name = "uninformative"
    kind = "index"
    def encode(self, q, world):
        return [0] * world.num_factors


class ScrambledState(StateEncoder):
    """A fixed bijective re-labelling of each factor's indices (arbitrary but learnable code)."""
    name = "scrambled"
    kind = "index"
    def __init__(self, world: FactoredWorld):
        rng = random.Random(SCRAMBLE_SEED)
        self.perms: List[List[int]] = []
        for size in world.sizes:
            p = list(range(size))
            rng.shuffle(p)
            self.perms.append(p)
    def encode(self, q, world):
        return [self.perms[f][q.state[f]] for f in range(world.num_factors)]


class PerceptualState(StateEncoder):
    """A fixed nonlinear projection of the one-hot factored state into a real vector.

    Factors are *entangled* by two fixed random tanh layers, so the model cannot read a
    factor off a single coordinate -- it must learn to disentangle. This is the condition
    that makes grounding a genuine computation rather than a lookup (RESEARCH_PLAN §4).
    The projection is fixed across all runs (seeded once), i.e. a shared 'perception'.
    """
    name = "perceptual"
    kind = "real"
    def __init__(self, world: FactoredWorld, dim: int = 16, seed: int = PERCEPT_SEED,
                 gain: float = 1.0, depth: int = 2):
        # Entanglement strength is an OPERATIONAL VARIABLE (not a bug to hide): the audit in
        # E6 showed the weak default stays linearly decodable. Low dim + high gain (saturating
        # tanh) + more depth reduces linear decodability, letting us report a weak/strong pair.
        import numpy as np
        self.dim = dim
        self.world = world
        self.gain = gain
        onehot_dim = sum(world.sizes)
        rng = np.random.default_rng(seed)
        dims = [onehot_dim] + [dim] * depth
        self._W = [rng.standard_normal((dims[i + 1], dims[i])) / np.sqrt(dims[i]) for i in range(depth)]
        self._b = [rng.standard_normal(dims[i + 1]) * 0.1 for i in range(depth)]
        self._np = np

    def _onehot(self, state):
        vec = []
        for f, size in enumerate(self.world.sizes):
            oh = [0.0] * size
            oh[state[f]] = 1.0
            vec.extend(oh)
        return self._np.asarray(vec, dtype=float)

    def encode(self, q, world):
        np = self._np
        h = self._onehot(q.state)
        for W, b in zip(self._W, self._b):
            h = np.tanh(self.gain * (W @ h + b))
        return h.astype(float).tolist()


class OneHotState(StateEncoder):
    """Concatenated one-hot factors as a REAL vector through the model's shared projection.

    The 2x2 cell that dissociates code readability from input-pathway sharing: maximally
    readable (like factored indices) but routed through the SAME shared Linear as the
    perceptual codes (unlike factored's per-factor embedding tables). If few-shot binding
    efficiency tracks pathway sharing, this should match weak-perceptual and beat factored.
    """
    name = "onehot"
    kind = "real"
    def encode(self, q, world):
        vec = []
        for f, size in enumerate(world.sizes):
            oh = [0.0] * size
            oh[q.state[f]] = 1.0
            vec.extend(oh)
        return vec


# Which model state_mode each encoder maps to.
STATE_MODE = {
    "none": "none",
    "free_order": "none",
    "factored": "index",
    "uninformative": "index",
    "scrambled": "index",
    "perceptual": "perceptual",
    "perceptual_hard": "perceptual",
    "perceptual_hard16": "perceptual",
    "perceptual_r085": "perceptual",
    "perceptual_r073": "perceptual",
    "perceptual_b": "perceptual",
    "perceptual_c": "perceptual",
    "perceptual_hard_b": "perceptual",
    "perceptual_hard_c": "perceptual",
    "onehot": "perceptual",
}


# ---------------------------------------------------------------------------
# Condition = text form + state encoder
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Condition:
    key: str
    show_description: bool          # include full surface description in the text?
    encoder_name: str               # "none" | "factored" | "uninformative" | "scrambled"


CONDITIONS: Dict[str, Condition] = {
    "text_only":         Condition("text_only",         show_description=True,  encoder_name="none"),
    "text_free_order":   Condition("text_free_order",   show_description=True,  encoder_name="free_order"),
    "text_minimal":      Condition("text_minimal",      show_description=False, encoder_name="none"),
    "state_factored":    Condition("state_factored",    show_description=False, encoder_name="factored"),
    "state_perceptual":  Condition("state_perceptual",  show_description=False, encoder_name="perceptual"),
    "state_perceptual_hard": Condition("state_perceptual_hard", show_description=False, encoder_name="perceptual_hard"),
    "state_perceptual_hard16": Condition("state_perceptual_hard16", show_description=False, encoder_name="perceptual_hard16"),
    "state_perceptual_r085": Condition("state_perceptual_r085", show_description=False, encoder_name="perceptual_r085"),
    "state_perceptual_r073": Condition("state_perceptual_r073", show_description=False, encoder_name="perceptual_r073"),
    "state_onehot_shared": Condition("state_onehot_shared", show_description=False, encoder_name="onehot"),
    # alternative mix instantiations (robustness to the fixed synthetic map, not a single draw)
    "state_perceptual_b": Condition("state_perceptual_b", show_description=False, encoder_name="perceptual_b"),
    "state_perceptual_c": Condition("state_perceptual_c", show_description=False, encoder_name="perceptual_c"),
    "state_perceptual_hard_b": Condition("state_perceptual_hard_b", show_description=False, encoder_name="perceptual_hard_b"),
    "state_perceptual_hard_c": Condition("state_perceptual_hard_c", show_description=False, encoder_name="perceptual_hard_c"),
    # controls
    "uninformative_state": Condition("uninformative_state", show_description=False, encoder_name="uninformative"),
    "scrambled_state":     Condition("scrambled_state",     show_description=False, encoder_name="scrambled"),
}


def make_encoder(name: str, world: FactoredWorld) -> StateEncoder:
    if name in ("none", "free_order"):
        return NoState()
    if name == "factored":
        return FactoredState()
    if name == "uninformative":
        return UninformativeState()
    if name == "scrambled":
        return ScrambledState(world)
    if name == "perceptual":
        return PerceptualState(world)                                   # weak (audited)
    if name == "perceptual_hard":
        return PerceptualState(world, dim=8, gain=2.5, depth=3, seed=PERCEPT_SEED + 1)  # strong
    if name == "perceptual_hard16":
        # dimension-matched to weak (d=16) but poorly readable -> isolates readability from dim
        return PerceptualState(world, dim=16, gain=4.0, depth=4, seed=PERCEPT_SEED + 2)
    if name == "perceptual_r085":   # d=16 readability sweep: intermediate (~0.85)
        return PerceptualState(world, dim=16, gain=1.8, depth=3, seed=PERCEPT_SEED + 3)
    if name == "perceptual_r073":   # d=16 readability sweep: intermediate (~0.73)
        return PerceptualState(world, dim=16, gain=2.5, depth=3, seed=PERCEPT_SEED + 4)
    if name == "perceptual_b":
        return PerceptualState(world, seed=PERCEPT_SEED + 10)
    if name == "perceptual_c":
        return PerceptualState(world, seed=PERCEPT_SEED + 20)
    if name == "perceptual_hard_b":
        return PerceptualState(world, dim=8, gain=2.5, depth=3, seed=PERCEPT_SEED + 11)
    if name == "perceptual_hard_c":
        return PerceptualState(world, dim=8, gain=2.5, depth=3, seed=PERCEPT_SEED + 21)
    if name == "onehot":
        return OneHotState()
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Vocabulary (word-level, built from the world)
# ---------------------------------------------------------------------------
class Vocab:
    def __init__(self, world):
        specials = ["<pad>", "<s>", "</s>", "<unk>"]
        self.tokens = specials + list(world.vocab_words())
        # de-dup while preserving order
        seen = set()
        self.tokens = [t for t in self.tokens if not (t in seen or seen.add(t))]
        self.tok2id = {t: i for i, t in enumerate(self.tokens)}
        self.pad_id = self.tok2id["<pad>"]
        self.bos_id = self.tok2id["<s>"]
        self.eos_id = self.tok2id["</s>"]
        self.unk_id = self.tok2id["<unk>"]

    def encode(self, text: str) -> List[int]:
        return [self.bos_id] + [self.tok2id.get(t, self.unk_id) for t in text.split()] + [self.eos_id]

    def __len__(self):
        return len(self.tokens)


# ---------------------------------------------------------------------------
# Text realisation of a query under a condition
# ---------------------------------------------------------------------------
def query_text(q: Query, cond: Condition, world) -> str:
    """Delegate to the world so single- and two-object worlds render their own text."""
    if getattr(cond, "encoder_name", "") == "free_order":
        return world.render_text(q, show_description=True, free_order=True)
    return world.render_text(q, cond.show_description)


# ---------------------------------------------------------------------------
# Example / dataset construction
# ---------------------------------------------------------------------------
@dataclass
class Example:
    input_ids: List[int]
    state: Optional[List[int]]
    target_id: int
    factor: int
    target_name: str


def build_examples(world: FactoredWorld, task: str, cond: Condition,
                   state_ids, vocab: Vocab) -> List[Example]:
    return build_examples_from_queries(world, cond, world.queries(task, state_ids), vocab)


def build_examples_from_queries(world: FactoredWorld, cond: Condition,
                                queries, vocab: Vocab) -> List[Example]:
    """Build examples from an explicit query list (supports query-level holdouts)."""
    encoder = make_encoder(cond.encoder_name, world)
    out: List[Example] = []
    for q in queries:
        text = query_text(q, cond, world)
        out.append(Example(
            input_ids=vocab.encode(text),
            state=encoder.encode(q, world),
            target_id=vocab.tok2id[q.target_name],
            factor=q.factor,
            target_name=q.target_name,
        ))
    return out


# ---------------------------------------------------------------------------
# Metrics  (balanced accuracy + chance + majority, per RESEARCH_PLAN §5.4)
# ---------------------------------------------------------------------------
def baseline_metrics(world: FactoredWorld, examples: List[Example]) -> Dict[str, float]:
    """Chance and majority baselines for a set of examples (no model needed)."""
    from collections import Counter, defaultdict
    per_factor_targets: Dict[int, List[str]] = defaultdict(list)
    for e in examples:
        per_factor_targets[e.factor].append(e.target_name)
    # balanced chance = mean over factors of 1/k
    chance = sum(1.0 / world.sizes[f] for f in per_factor_targets) / len(per_factor_targets)
    majorities = []
    for f, targets in per_factor_targets.items():
        top = Counter(targets).most_common(1)[0][1]
        majorities.append(top / len(targets))
    majority = sum(majorities) / len(majorities)
    return {"balanced_chance": chance, "balanced_majority": majority}


def accuracy_metrics(examples: List[Example], correct_flags: List[bool]) -> Dict[str, float]:
    """Overall + balanced (per-factor mean) accuracy from per-example correctness."""
    from collections import defaultdict
    hit = defaultdict(int); tot = defaultdict(int)
    for e, ok in zip(examples, correct_flags):
        tot[e.factor] += 1
        hit[e.factor] += int(ok)
    overall = sum(hit.values()) / max(sum(tot.values()), 1)
    per_factor = {f: hit[f] / tot[f] for f in tot}
    balanced = sum(per_factor.values()) / max(len(per_factor), 1)
    return {"overall": overall, "balanced": balanced,
            "per_factor": {str(f): v for f, v in per_factor.items()}}
