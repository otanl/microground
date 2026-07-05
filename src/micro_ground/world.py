"""Single-object finite world for MicroGround.

World: 4 colors x 4 shapes x 4 positions x 2 sizes = 128 states.
Supports text-only, state-grounded, and counterfactual conditions.
"""
import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple, Optional


class Condition(Enum):
    TEXT_ONLY = "text_only"      # full description, no state vector
    TEXT_MINIMAL = "text_minimal"  # minimal query, no state vector (should fail on many tasks)
    STATE_GROUNDED = "state_grounded"  # minimal query + state vector
    COUNTERFACTUAL = "counterfactual"  # minimal query + state vector (counterfactual task)


class Task(Enum):
    ATTRIBUTE_CLASSIFICATION = "attr"        # input: text, output: attribute value
    OBJECT_IDENTIFICATION = "id"             # input: text, output: object id
    COUNTERFACTUAL = "counterfactual"        # input: text + state + counterfactual change, output: new attribute/id


COLORS = ["red", "blue", "green", "yellow"]
SHAPES = ["circle", "square", "triangle", "star"]
POSITIONS = ["left", "center", "right", "back"]
SIZES = ["small", "big"]

ATTRIBUTE_NAMES = ["color", "shape", "position", "size"]
ATTRIBUTE_LISTS = [COLORS, SHAPES, POSITIONS, SIZES]


@dataclass(frozen=True)
class ObjectState:
    color: int  # index in COLORS
    shape: int
    position: int
    size: int

    def __str__(self):
        return f"{COLORS[self.color]} {SHAPES[self.shape]} {POSITIONS[self.position]} {SIZES[self.size]}"

    def to_vector(self) -> List[int]:
        return [self.color, self.shape, self.position, self.size]


class ObjectWorld:
    def __init__(self):
        self.states = [
            ObjectState(c, s, p, z)
            for c in range(len(COLORS))
            for s in range(len(SHAPES))
            for p in range(len(POSITIONS))
            for z in range(len(SIZES))
        ]
        self.state_to_id = {s: i for i, s in enumerate(self.states)}
        self.num_states = len(self.states)

    def id_to_state(self, idx: int) -> ObjectState:
        return self.states[idx]

    def text_for_state(self, state: ObjectState) -> str:
        return str(state)

    def attribute_value(self, state: ObjectState, attr: str) -> int:
        return getattr(state, attr)

    def attribute_name(self, state: ObjectState, attr: str) -> str:
        lst = ATTRIBUTE_LISTS[ATTRIBUTE_NAMES.index(attr)]
        return lst[self.attribute_value(state, attr)]

    def all_states(self) -> List[ObjectState]:
        return list(self.states)

    def apply_counterfactual(self, state: ObjectState, attr: str, new_value: int) -> ObjectState:
        """Return a new state with one attribute changed."""
        if attr == "color":
            return ObjectState(new_value, state.shape, state.position, state.size)
        if attr == "shape":
            return ObjectState(state.color, new_value, state.position, state.size)
        if attr == "position":
            return ObjectState(state.color, state.shape, new_value, state.size)
        if attr == "size":
            return ObjectState(state.color, state.shape, state.position, new_value)
        raise ValueError(attr)

    def random_split(self, ratio: float = 0.8, seed: int = 0) -> Tuple[List[int], List[int]]:
        rng = random.Random(seed)
        ids = list(range(self.num_states))
        rng.shuffle(ids)
        n = int(self.num_states * ratio)
        return ids[:n], ids[n:]

    def generate_examples(
        self,
        task: Task,
        condition: Condition,
        split: str = "train",
        seed: int = 0,
        holdout_attr: Optional[str] = None,
        holdout_composition: Optional[dict] = None,
    ):
        """Generate a list of (input_text, target) examples.

        Args:
            task: which task to generate.
            condition: text_only, state_grounded, or counterfactual.
            split: 'train' or 'test'.
            seed: random seed for splitting.
            holdout_attr: if set, hold out all states with this attribute value from training.
            holdout_composition: dict like {"color": 0, "shape": 1} to hold out all states matching the combination.
        """
        if holdout_composition is not None:
            train_ids = []
            test_ids = []
            for i, s in enumerate(self.states):
                if all(self.attribute_value(s, attr) == val for attr, val in holdout_composition.items()):
                    test_ids.append(i)
                else:
                    train_ids.append(i)
            rng = random.Random(seed)
            rng.shuffle(train_ids)
            rng.shuffle(test_ids)
        elif holdout_attr is not None:
            assert holdout_attr in ATTRIBUTE_NAMES
            train_ids = [i for i, s in enumerate(self.states) if self.attribute_value(s, holdout_attr) == 0]
            test_ids = [i for i, s in enumerate(self.states) if self.attribute_value(s, holdout_attr) != 0]
            rng = random.Random(seed)
            rng.shuffle(train_ids)
            rng.shuffle(test_ids)
        else:
            train_ids, test_ids = self.random_split(0.8, seed)

        ids = train_ids if split == "train" else test_ids

        examples = []
        for idx in ids:
            state = self.states[idx]
            if task == Task.ATTRIBUTE_CLASSIFICATION:
                # randomly pick an attribute to predict
                attr = random.Random(seed + idx).choice(ATTRIBUTE_NAMES)
                text = self._input_text(condition, state, attr=attr)
                target = self.attribute_name(state, attr)
                examples.append((text, target, {"task": task.value, "attr": attr, "state": state.to_vector()}))
            elif task == Task.OBJECT_IDENTIFICATION:
                text = self._input_text(condition, state)
                target = str(idx)
                examples.append((text, target, {"task": task.value, "state": state.to_vector()}))
            elif task == Task.COUNTERFACTUAL:
                attr = random.Random(seed + idx).choice(ATTRIBUTE_NAMES)
                attr_idx = ATTRIBUTE_NAMES.index(attr)
                attr_size = len(ATTRIBUTE_LISTS[attr_idx])
                # Counterfactual: shift to the next value (systematic transformation)
                new_value = (self.attribute_value(state, attr) + 1) % attr_size
                new_state = self.apply_counterfactual(state, attr, new_value)
                text = self._input_text(condition, state, attr=attr, new_value=new_value)
                target = self.attribute_name(new_state, attr)
                examples.append((text, target, {
                    "task": task.value,
                    "attr": attr,
                    "old_state": state.to_vector(),
                    "new_state": new_state.to_vector(),
                }))
            else:
                raise ValueError(task)
        return examples

    def _input_text(self, condition: Condition, state: ObjectState, attr: Optional[str] = None, new_value: Optional[int] = None) -> str:
        if condition == Condition.TEXT_ONLY:
            base = self.text_for_state(state)
            if attr is not None and new_value is not None:
                # Do not reveal the new value; model must infer from text alone (impossible without state)
                return f"{base} change {attr}"
            if attr is not None:
                return f"{base} what is the {attr}"
            return base
        elif condition == Condition.TEXT_MINIMAL:
            # Only the question, no state description. Model must rely on state vector.
            if attr is not None and new_value is not None:
                return f"change {attr}"
            if attr is not None:
                return f"what is the {attr}"
            return "what is this"
        elif condition == Condition.STATE_GROUNDED:
            # state vector is appended separately, not in text
            if attr is not None and new_value is not None:
                return f"change {attr}"
            if attr is not None:
                return f"what is the {attr}"
            return self.text_for_state(state)
        elif condition == Condition.COUNTERFACTUAL:
            # state vector + instruction to apply counterfactual
            if attr is None:
                raise ValueError("Counterfactual condition requires attr")
            return f"change {attr}"
        else:
            raise ValueError(condition)
