"""Corrected, deterministic, exhaustively-enumerable finite world (TMLR-grade core).

This module replaces the flawed enumeration in ``src/micro_ground`` for the redesigned
study (see ``docs/RESEARCH_PLAN.md``). Key differences from the legacy code:

* **Exhaustive.** Query generation enumerates *all* (state x factor) pairs. The legacy
  code sampled one random attribute per state (``random.Random(seed+idx).choice(...)``),
  so the evaluated query set depended on the seed. Here the full query space is fixed
  and complete (e.g. 128 states x 4 factors = 512 attribute queries).
* **Seed separation.** ``split_seed`` partitions the *state* space into train/test only.
  Query generation is fully deterministic given the states, so evaluation carries **zero
  sampling variance** -- the only variance source is model initialisation.
* **Holdouts as first-class splits.** random / attribute-value / compositional holdouts
  all return disjoint (train, test) state-index sets from one interface.

A ``FactoredWorld`` is defined purely by its factors, so single- and multi-object worlds
share one implementation.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Factor definitions
# ---------------------------------------------------------------------------
# A factor is (name, [value strings]). A state is a tuple of value indices.
FACTORS_1OBJ: List[Tuple[str, List[str]]] = [
    ("color", ["red", "blue", "green", "yellow"]),
    ("shape", ["circle", "square", "triangle", "star"]),
    ("position", ["left", "center", "right", "back"]),
    ("size", ["small", "big"]),
]

State = Tuple[int, ...]


@dataclass(frozen=True)
class Query:
    """A single fully-specified example, independent of condition/encoding.

    ``condition`` (text form, state vector) is applied later by the encoder layer, so the
    same Query is reused across all conditions -- this guarantees conditions are compared
    on an identical query set.
    """

    state: State
    task: str          # "attr" | "counterfactual"
    factor: int        # which factor is queried / transformed
    target_value: int  # the ground-truth value index of the answer
    target_name: str   # the ground-truth answer as a word (the classification label)


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------
class FactoredWorld:
    """A finite world defined by independent categorical factors."""

    def __init__(self, factors: Sequence[Tuple[str, List[str]]] = FACTORS_1OBJ):
        self.factor_names: List[str] = [n for n, _ in factors]
        self.factor_values: List[List[str]] = [list(v) for _, v in factors]
        self.sizes: List[int] = [len(v) for _, v in factors]
        self.num_factors: int = len(factors)
        self.states: List[State] = list(itertools.product(*[range(s) for s in self.sizes]))
        self.num_states: int = len(self.states)

    # -- helpers -------------------------------------------------------------
    def value_name(self, state: State, factor: int) -> str:
        return self.factor_values[factor][state[factor]]

    def state_text(self, state: State) -> str:
        """Full surface description, e.g. 'red circle left small'."""
        return " ".join(self.factor_values[f][state[f]] for f in range(self.num_factors))

    def render_text(self, q: "Query", show_description: bool) -> str:
        """Surface text for a query under a condition (route delegates here)."""
        verb = "what is the" if q.task == "attr" else "change"
        question = f"{verb} {self.factor_names[q.factor]}"
        return f"{self.state_text(q.state)} {question}" if show_description else question

    def vocab_words(self) -> List[str]:
        """Every word this world can emit in text (for the shared Vocab)."""
        words = ["what", "is", "the", "change"] + list(self.factor_names)
        for vals in self.factor_values:
            words += vals
        return words

    # -- splits (partition the STATE space; queries stay exhaustive) ---------
    def split_states(
        self,
        split_seed: int = 0,
        ratio: float = 0.8,
        holdout: Optional[Tuple] = None,
    ) -> Dict[str, List[int]]:
        """Return {'train': [state_idx...], 'test': [...], 'all': [...]}.

        ``holdout`` selects the generalisation regime:
          * ``None``                       -> random ``ratio`` split (seeded).
          * ``("attr", factor, value)``    -> hold out every state whose ``factor==value``.
          * ``("comp", {factor: value})``  -> hold out states matching the whole combination.
        """
        all_ids = list(range(self.num_states))
        if holdout is None:
            rng = random.Random(split_seed)
            shuffled = all_ids[:]
            rng.shuffle(shuffled)
            n_train = int(self.num_states * ratio)
            train, test = shuffled[:n_train], shuffled[n_train:]
        elif holdout[0] == "attr":
            _, factor, value = holdout
            test = [i for i, s in enumerate(self.states) if s[factor] == value]
            train = [i for i in all_ids if i not in set(test)]
        elif holdout[0] == "comp":
            _, combo = holdout
            def matches(s: State) -> bool:
                return all(s[f] == v for f, v in combo.items())
            test = [i for i, s in enumerate(self.states) if matches(self.states[i])]
            train = [i for i in all_ids if i not in set(test)]
        else:
            raise ValueError(f"unknown holdout spec: {holdout!r}")
        return {"train": sorted(train), "test": sorted(test), "all": all_ids}

    # -- exhaustive query generation ----------------------------------------
    def queries(self, task: str, state_ids: Sequence[int]) -> List[Query]:
        """Every (state x factor) query for the given states. Deterministic, complete."""
        out: List[Query] = []
        for si in state_ids:
            state = self.states[si]
            for f in range(self.num_factors):
                if task == "attr":
                    tv = state[f]
                elif task == "counterfactual":
                    # systematic successor transformation: v -> (v+1) mod k
                    tv = (state[f] + 1) % self.sizes[f]
                else:
                    raise ValueError(f"unknown task: {task}")
                out.append(
                    Query(
                        state=state,
                        task=task,
                        factor=f,
                        target_value=tv,
                        target_name=self.factor_values[f][tv],
                    )
                )
        return out

    def full_query_space(self, task: str) -> List[Query]:
        """All queries over all states -- the correct 'exhaustive' evaluation set."""
        return self.queries(task, range(self.num_states))

    def split_queries(self, task: str, split_seed: int = 0, ratio: float = 0.8,
                      holdout: Optional[Tuple] = None) -> Dict[str, List[Query]]:
        """Split the exhaustive query space into train/test.

        State-level holdouts (None/attr/comp) partition states then take their queries.
        Query-level holdouts operate directly on queries:
          * ``("transition", factor, value)`` -- hold out counterfactual queries that
            transform ``factor`` starting from source ``value``. This is the rule-vs-
            memorisation test: the model still sees ``factor==value`` elsewhere, but never
            sees the ``value -> value+1`` transition, so only rule-learning generalises.
        """
        all_q = self.full_query_space(task)
        if holdout is None or holdout[0] in ("attr", "comp"):
            sp = self.split_states(split_seed, ratio, holdout)
            train_states = {self.states[i] for i in sp["train"]}
            train = [q for q in all_q if q.state in train_states]
            test = [q for q in all_q if q.state not in train_states]
        elif holdout[0] == "transition":
            _, factor, value = holdout
            def held(q: Query) -> bool:
                return q.factor == factor and q.state[factor] == value
            train = [q for q in all_q if not held(q)]
            test = [q for q in all_q if held(q)]
        elif holdout[0] == "transition_frac":
            # Hold out a random subset of (factor, source-value) transition-types. The test
            # set spans several factors/targets (non-degenerate baselines); only a model that
            # learned the +1 rule generalises to unseen transitions.
            _, frac = holdout
            types = [(f, v) for f in range(self.num_factors) for v in range(self.sizes[f])]
            rng = random.Random(split_seed)
            rng.shuffle(types)
            n_hold = max(1, int(round(len(types) * frac)))
            held_types = set(types[:n_hold])
            def held(q: Query) -> bool:
                return (q.factor, q.state[q.factor]) in held_types
            train = [q for q in all_q if not held(q)]
            test = [q for q in all_q if held(q)]
        else:
            raise ValueError(f"unknown holdout spec: {holdout!r}")
        return {"train": train, "test": test, "all": all_q}
