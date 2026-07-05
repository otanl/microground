"""Two-object finite world for MicroGround.

World: 2 objects, each with 4 colors x 4 shapes x 2 positions = 256 joint states.
Supports text-only, text-minimal, state-grounded, and counterfactual conditions.
"""
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

from .world import Condition, ATTRIBUTE_NAMES, ATTRIBUTE_LISTS

COLORS = ATTRIBUTE_LISTS[0]
SHAPES = ATTRIBUTE_LISTS[1]
POSITIONS = ["left", "right"]


@dataclass(frozen=True)
class MultiObjectState:
    color1: int
    shape1: int
    pos1: int
    color2: int
    shape2: int
    pos2: int

    def to_vector(self) -> List[int]:
        return [self.color1, self.shape1, self.pos1, self.color2, self.shape2, self.pos2]


class MultiObjectWorld:
    def __init__(self):
        self.states = [
            MultiObjectState(c1, s1, p1, c2, s2, p2)
            for c1 in range(len(COLORS))
            for s1 in range(len(SHAPES))
            for p1 in range(len(POSITIONS))
            for c2 in range(len(COLORS))
            for s2 in range(len(SHAPES))
            for p2 in range(len(POSITIONS))
        ]
        self.num_states = len(self.states)

    def text_for_state(self, state: MultiObjectState) -> str:
        return (
            f"{COLORS[state.color1]} {SHAPES[state.shape1]} {POSITIONS[state.pos1]} "
            f"{COLORS[state.color2]} {SHAPES[state.shape2]} {POSITIONS[state.pos2]}"
        )

    def random_split(self, ratio: float = 0.8, seed: int = 0):
        rng = random.Random(seed)
        ids = list(range(self.num_states))
        rng.shuffle(ids)
        n = int(self.num_states * ratio)
        return ids[:n], ids[n:]

    def generate_examples(
        self,
        condition: Condition,
        split: str = "train",
        seed: int = 0,
    ):
        train_ids, test_ids = self.random_split(0.8, seed)
        ids = train_ids if split == "train" else test_ids

        examples = []
        for idx in ids:
            state = self.states[idx]
            rng = random.Random(seed + idx)
            task_type = rng.choice(["loc_color", "loc_shape", "relation", "counterfactual"])

            if task_type == "loc_color":
                pos = rng.choice([0, 1])
                # Ask color of object at position pos
                target = COLORS[state.color1 if pos == 0 else state.color2]
                text = self._input_text(condition, state, task_type, pos=pos)
            elif task_type == "loc_shape":
                pos = rng.choice([0, 1])
                target = SHAPES[state.shape1 if pos == 0 else state.shape2]
                text = self._input_text(condition, state, task_type, pos=pos)
            elif task_type == "relation":
                # Ask: is color1 left of color2? (binary yes/no)
                target = "yes" if state.pos1 < state.pos2 else "no"
                text = self._input_text(condition, state, task_type)
            elif task_type == "counterfactual":
                # Move the first object to the other position and ask its new position
                obj = 0
                new_pos = 1 - state.pos1
                target = POSITIONS[new_pos]
                text = self._input_text(condition, state, task_type, obj=obj, new_pos=new_pos)
            else:
                raise ValueError(task_type)

            examples.append((text, target, {"task": task_type, "state": state.to_vector()}))
        return examples

    def _input_text(self, condition: Condition, state: MultiObjectState, task_type: str, pos: int = None, obj: int = None, new_pos: int = None) -> str:
        if condition == Condition.TEXT_ONLY:
            base = self.text_for_state(state)
            if task_type == "loc_color":
                return f"{base} color of {POSITIONS[pos]}"
            if task_type == "loc_shape":
                return f"{base} shape of {POSITIONS[pos]}"
            if task_type == "relation":
                return f"{base} is {COLORS[state.color1]} left of {COLORS[state.color2]}"
            if task_type == "counterfactual":
                return f"{base} move first to {POSITIONS[new_pos]} where is it"
        elif condition == Condition.TEXT_MINIMAL:
            if task_type == "loc_color":
                return f"color of {POSITIONS[pos]}"
            if task_type == "loc_shape":
                return f"shape of {POSITIONS[pos]}"
            if task_type == "relation":
                return f"is {COLORS[state.color1]} left of {COLORS[state.color2]}"
            if task_type == "counterfactual":
                return f"move first where is it"
        elif condition == Condition.STATE_GROUNDED:
            if task_type == "loc_color":
                return f"color of {POSITIONS[pos]}"
            if task_type == "loc_shape":
                return f"shape of {POSITIONS[pos]}"
            if task_type == "relation":
                return "is first left of second"
            if task_type == "counterfactual":
                return f"move first where is it"
        elif condition == Condition.COUNTERFACTUAL:
            if task_type != "counterfactual":
                raise ValueError("Counterfactual condition only for counterfactual task")
            return f"move first where is it"
        raise ValueError(condition)
