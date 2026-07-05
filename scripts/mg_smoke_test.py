r"""
Validate the redesigned MicroGround core (src/mg): exhaustive enumeration, seed-separated
splits, deterministic queries, and baseline metrics -- and demonstrate the legacy eval bug.

No torch required. Run:
    $env:PYTHONUTF8=1
    .\.venv\Scripts\python.exe scripts\mg_smoke_test.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mg import FactoredWorld, Vocab, CONDITIONS, build_examples, baseline_metrics, make_encoder

PASS, FAIL = "[PASS]", "[FAIL]"
def check(name, cond):
    print(f"{PASS if cond else FAIL} {name}")
    assert cond, name


def main():
    world = FactoredWorld()
    vocab = Vocab(world)
    print(f"world: {world.num_states} states, factors={world.factor_names} sizes={world.sizes}")
    print(f"vocab: {len(vocab)} tokens\n")

    # 1. Exhaustive query space: 128 states x 4 factors = 512, for both tasks.
    for task, expect in [("attr", 512), ("counterfactual", 512)]:
        qs = world.full_query_space(task)
        check(f"{task}: full query space == {expect} (got {len(qs)})", len(qs) == expect)

    # 2. Determinism: identical queries on rebuild (legacy varied with seed).
    a = world.full_query_space("attr")
    b = world.full_query_space("attr")
    check("attr queries are deterministic across calls", a == b)

    # 3. Split: disjoint train/test, union == all, exhaustive queries per split sum to 512.
    sp = world.split_states(split_seed=0)
    disjoint = set(sp["train"]).isdisjoint(sp["test"])
    covers = sorted(sp["train"] + sp["test"]) == list(range(world.num_states))
    n_train_q = len(world.queries("attr", sp["train"]))
    n_test_q = len(world.queries("attr", sp["test"]))
    check("split train/test disjoint", disjoint)
    check("split covers all states", covers)
    check(f"train_q + test_q == 512 (got {n_train_q}+{n_test_q})", n_train_q + n_test_q == 512)

    # 4. Baselines on the full attr space: uniform -> majority == chance == mean(1/k).
    ex_full = build_examples(world, "attr", CONDITIONS["text_minimal"], sp["all"], vocab)
    base = baseline_metrics(world, ex_full)
    expected_chance = sum(1.0 / k for k in world.sizes) / world.num_factors  # 0.3125
    check(f"balanced chance == mean(1/k) == {expected_chance:.4f} (got {base['balanced_chance']:.4f})",
          abs(base["balanced_chance"] - expected_chance) < 1e-9)
    check(f"full-space attr is balanced: majority == chance (maj={base['balanced_majority']:.4f})",
          abs(base["balanced_majority"] - base["balanced_chance"]) < 1e-9)

    # 5. TAUTOLOGY DEMO: for attr + factored state, the answer index == state[factor].
    #    This concretely shows why 'state_factored attr' is a lookup, not grounding.
    enc = make_encoder("factored", world)
    tautological = all(
        enc.encode(q, world)[q.factor] == q.target_value
        for q in world.full_query_space("attr")
    )
    check("DEMO attr+factored is a pure lookup (state[factor] == answer)", tautological)

    # 6. CONTROL DEMO: uninformative state is constant -> carries no state info.
    enc_u = make_encoder("uninformative", world)
    vecs = {tuple(enc_u.encode(q, world)) for q in world.full_query_space("attr")}
    check("DEMO uninformative_state is constant (== text_minimal in info)", len(vecs) == 1)

    # 7. LEGACY BUG DEMO: micro_ground 'all' split is actually the test split, seed-dependent.
    print("\n--- legacy src/micro_ground bug demonstration ---")
    try:
        from micro_ground import ObjectWorld
        ow = ObjectWorld()
        n_all_s0 = len(ow.generate_examples_all() if hasattr(ow, "generate_examples_all") else [])
    except Exception:
        n_all_s0 = None
    try:
        from micro_ground import ObjectWorld, Task, Condition as LCond
        ow = ObjectWorld()
        a0 = len(ow.generate_examples(Task("attr"), LCond("text_minimal"), "all", seed=0))
        a1 = len(ow.generate_examples(Task("attr"), LCond("text_minimal"), "all", seed=1))
        print(f"legacy generate_examples(..., 'all', seed=0) -> {a0} examples (NOT 512)")
        print(f"legacy generate_examples(..., 'all', seed=1) -> {a1} examples")
        check("legacy 'all' != 512 (proves it is the test split, not exhaustive)", a0 != 512)
        print("  => confirms RESEARCH_PLAN §1.1: 'all_states' is really the 20% test split,")
        print("     one random attribute per state, and seed-dependent.")
    except Exception as e:
        print(f"(legacy comparison skipped: {e})")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
