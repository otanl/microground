r"""Generate a synthetic grounded-language corpus for pre-training MicroGround.

Sentences use the MicroGround vocabulary but are not task instances. They are meant to
mimic a grounded language corpus that a tiny LM could be pre-trained on.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from micro_ground import ObjectWorld, ATTRIBUTE_LISTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_sentences", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="data/synthetic_grounded_corpus.txt")
    args = p.parse_args()

    world = ObjectWorld()
    rng = random.Random(args.seed)
    colors = ATTRIBUTE_LISTS[0]
    shapes = ATTRIBUTE_LISTS[1]
    positions = ATTRIBUTE_LISTS[2]
    sizes = ATTRIBUTE_LISTS[3]

    templates = [
        "The {color} {shape} is {size}.",
        "The {color} {shape} is at the {position}.",
        "There is a {size} {color} {shape} on the {position}.",
        "Look at the {color} {shape} on the {position}.",
        "The {shape} is {color} and {size}.",
        "The {color} object is {size} and at the {position}.",
        "What color is the {shape} at the {position}? {color}.",
        "What shape is the {color} object at the {position}? {shape}.",
        "Where is the {color} {shape}? {position}.",
        "Is the {color} {shape} {size}? yes.",
    ]

    sentences = []
    for _ in range(args.num_sentences):
        state = rng.choice(world.states)
        color = colors[world.attribute_value(state, "color")]
        shape = shapes[world.attribute_value(state, "shape")]
        position = positions[world.attribute_value(state, "position")]
        size = sizes[world.attribute_value(state, "size")]
        template = rng.choice(templates)
        sentences.append(template.format(color=color, shape=shape, position=position, size=size))

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences))
    print(f"Wrote {len(sentences)} sentences to {args.output}")


if __name__ == "__main__":
    main()
