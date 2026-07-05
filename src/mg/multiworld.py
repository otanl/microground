"""Two-object world for the binding task (analysis B / C1).

Two objects with fixed positional roles: object 1 = "left", object 2 = "right", each with
a colour and a shape -> 4x4x4x4 = 256 fully-enumerable states. A binding query asks for one
attribute of the object at one position ("shape of right"); answering requires binding the
queried (attribute, position) to the correct object slot.

The four query-types map one-to-one onto the four state slots, so ``Query.factor`` doubles as
both the state index and the (position, attribute) key:

    factor 0 = (left,  colour) -> state[0] = c1
    factor 1 = (left,  shape)  -> state[1] = s1
    factor 2 = (right, colour) -> state[2] = c2
    factor 3 = (right, shape)  -> state[3] = s2

The key generalisation test is ``bind_holdout``: hold out one query-type entirely (e.g. never
ask "shape of right" in training) and test whether the model composes the seen "shape" concept
with the seen "right" object -- a genuine compositional-binding generalisation, at intermediate
difficulty (unlike the 1-object holdouts which are all ceiling or floor).
"""
from __future__ import annotations

import random
from itertools import product as _product
from typing import Dict, List, Optional, Sequence, Tuple

from .world import Query

COLORS = ["red", "blue", "green", "yellow"]
SHAPES = ["circle", "square", "triangle", "star"]
POSITIONS = ["left", "right"]
ATTRS = ["color", "shape"]


class TwoObjectWorld:
    def __init__(self):
        self.sizes: List[int] = [4, 4, 4, 4]           # c1, s1, c2, s2 (for state encoders)
        self.num_factors: int = 4
        self.factor_names = ["color1", "shape1", "color2", "shape2"]
        self.factor_values = [COLORS, SHAPES, COLORS, SHAPES]
        self.states: List[Tuple[int, ...]] = [
            (c1, s1, c2, s2)
            for c1 in range(4) for s1 in range(4)
            for c2 in range(4) for s2 in range(4)
        ]
        self.num_states = len(self.states)             # 256

    # factor <-> (position, attribute)
    @staticmethod
    def _pos_attr(factor: int) -> Tuple[int, int]:
        return factor // 2, factor % 2                 # position (0=left,1=right), attr (0=color,1=shape)

    def value_name(self, state, factor: int) -> str:
        _, attr = self._pos_attr(factor)
        return (COLORS if attr == 0 else SHAPES)[state[factor]]

    def state_text(self, state) -> str:
        return f"{COLORS[state[0]]} {SHAPES[state[1]]} {COLORS[state[2]]} {SHAPES[state[3]]}"

    def vocab_words(self) -> List[str]:
        return ["color", "shape", "of", "left", "right"] + COLORS + SHAPES

    def render_text(self, q: Query, show_description: bool, free_order: bool = False,
                    order_seed: int = 0) -> str:
        position, attr = self._pos_attr(q.factor)
        question = f"{ATTRS[attr]} of {POSITIONS[position]}"
        if not show_description:
            return question
        if not free_order:
            return f"{self.state_text(q.state)} {question}"
        # Free-order control: shuffle the objects and TAG each with its position word, so binding
        # cannot exploit absolute token position -- the model must use the position tag.
        objs = [(POSITIONS[o], COLORS[q.state[2 * o]], SHAPES[q.state[2 * o + 1]]) for o in (0, 1)]
        rng = random.Random(order_seed + hash(q.state) % 100000)
        rng.shuffle(objs)
        desc = " ".join(f"{p} {c} {s}" for p, c, s in objs)
        return f"{desc} {question}"

    # -- query generation ----------------------------------------------------
    def queries(self, task: str, state_ids: Sequence[int]) -> List[Query]:
        assert task == "bind", f"TwoObjectWorld only supports task='bind', got {task!r}"
        out: List[Query] = []
        for si in state_ids:
            state = self.states[si]
            for f in range(self.num_factors):
                out.append(Query(state=state, task="bind", factor=f,
                                 target_value=state[f], target_name=self.value_name(state, f)))
        return out

    def full_query_space(self, task: str) -> List[Query]:
        return self.queries(task, range(self.num_states))

    # -- splits --------------------------------------------------------------
    def split_states(self, split_seed: int = 0, ratio: float = 0.8) -> Dict[str, List[int]]:
        ids = list(range(self.num_states))
        rng = random.Random(split_seed)
        sh = ids[:]
        rng.shuffle(sh)
        n = int(self.num_states * ratio)
        return {"train": sorted(sh[:n]), "test": sorted(sh[n:]), "all": ids}

    def split_queries(self, task: str, split_seed: int = 0, ratio: float = 0.8,
                      holdout: Optional[Tuple] = None) -> Dict[str, List[Query]]:
        all_q = self.full_query_space(task)
        if holdout is None:
            sp = self.split_states(split_seed, ratio)
            train_states = {self.states[i] for i in sp["train"]}
            train = [q for q in all_q if q.state in train_states]
            test = [q for q in all_q if q.state not in train_states]
        elif holdout[0] == "bind":
            # hold out one query-type (position, attribute) entirely -> compositional binding test
            held_factor = holdout[1]
            train = [q for q in all_q if q.factor != held_factor]
            test = [q for q in all_q if q.factor == held_factor]
        elif holdout[0] == "bind_kshot":
            # dose-response variant: leak a fraction of the held-out query-type back into
            # training. Converts the 0/1 zero-shot cliff into a graded sample-efficiency
            # measure ("how many examples of the unseen combination does each route need?").
            _, held_factor, frac = holdout
            held = [q for q in all_q if q.factor == held_factor]
            rest = [q for q in all_q if q.factor != held_factor]
            rng = random.Random(split_seed)
            idx = list(range(len(held)))
            rng.shuffle(idx)
            k = int(round(len(held) * frac))
            leak = set(idx[:k])
            train = rest + [q for i, q in enumerate(held) if i in leak]
            test = [q for i, q in enumerate(held) if i not in leak]
        else:
            raise ValueError(f"unknown holdout for TwoObjectWorld: {holdout!r}")
        return {"train": train, "test": test, "all": all_q}


class NObjectWorld:
    """Structurally larger / compositionally deeper binding world (larger-world replication).

    ``num_objects`` objects in fixed positional roles, each with a colour and a shape drawn from
    the first ``ncol``/``nshape`` values. Query types = num_objects x 2 attributes; factor ``f``
    maps to state slot ``f`` with object ``f//2`` and attribute ``f%2`` (0=colour,1=shape), exactly
    as in the two-object world, so all encoders, splits, and analyses transfer unchanged.
    """
    POS_NAMES = ["left", "middle", "right", "fourth", "fifth"]

    def __init__(self, num_objects: int = 3, ncol: int = 3, nshape: int = 3):
        assert num_objects <= len(self.POS_NAMES)
        self.num_objects = num_objects
        self.cols = COLORS[:ncol]
        self.shps = SHAPES[:nshape]
        self.positions = self.POS_NAMES[:num_objects]
        self.num_factors = 2 * num_objects
        self.sizes = [ncol if f % 2 == 0 else nshape for f in range(self.num_factors)]
        self.factor_names = [f"{'color' if f % 2 == 0 else 'shape'}{f // 2}" for f in range(self.num_factors)]
        self.factor_values = [self.cols if f % 2 == 0 else self.shps for f in range(self.num_factors)]
        ranges = [range(s) for s in self.sizes]
        self.states = list(_product(*ranges))
        self.num_states = len(self.states)

    @staticmethod
    def _pos_attr(factor):
        return factor // 2, factor % 2

    def value_name(self, state, factor):
        _, attr = self._pos_attr(factor)
        return (self.cols if attr == 0 else self.shps)[state[factor]]

    def state_text(self, state):
        return " ".join((self.cols if f % 2 == 0 else self.shps)[state[f]] for f in range(self.num_factors))

    def vocab_words(self):
        return ["color", "shape", "of"] + self.positions + self.cols + self.shps

    def render_text(self, q, show_description):
        position, attr = self._pos_attr(q.factor)
        question = f"{ATTRS[attr]} of {self.positions[position]}"
        return f"{self.state_text(q.state)} {question}" if show_description else question

    def queries(self, task, state_ids):
        assert str(task).startswith("bind")
        out = []
        for si in state_ids:
            state = self.states[si]
            for f in range(self.num_factors):
                out.append(Query(state=state, task="bind", factor=f,
                                 target_value=state[f], target_name=self.value_name(state, f)))
        return out

    def full_query_space(self, task):
        return self.queries(task, range(self.num_states))

    def split_states(self, split_seed=0, ratio=0.8):
        ids = list(range(self.num_states))
        rng = random.Random(split_seed)
        sh = ids[:]; rng.shuffle(sh)
        n = int(self.num_states * ratio)
        return {"train": sorted(sh[:n]), "test": sorted(sh[n:]), "all": ids}

    def split_queries(self, task, split_seed=0, ratio=0.8, holdout=None):
        all_q = self.full_query_space(task)
        if holdout is None:
            sp = self.split_states(split_seed, ratio)
            train_states = {self.states[i] for i in sp["train"]}
            train = [q for q in all_q if q.state in train_states]
            test = [q for q in all_q if q.state not in train_states]
        elif holdout[0] == "bind":
            hf = holdout[1]
            train = [q for q in all_q if q.factor != hf]
            test = [q for q in all_q if q.factor == hf]
        elif holdout[0] == "bind_kshot":
            _, hf, frac = holdout
            held = [q for q in all_q if q.factor == hf]
            rest = [q for q in all_q if q.factor != hf]
            rng = random.Random(split_seed)
            idx = list(range(len(held))); rng.shuffle(idx)
            leak = set(idx[:int(round(len(held) * frac))])
            train = rest + [q for i, q in enumerate(held) if i in leak]
            test = [q for i, q in enumerate(held) if i not in leak]
        else:
            raise ValueError(f"unknown holdout: {holdout!r}")
        return {"train": train, "test": test, "all": all_q}
