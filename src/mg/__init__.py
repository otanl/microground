"""TMLR-grade redesigned MicroGround core (see docs/RESEARCH_PLAN.md).

Deterministic, exhaustively-enumerable finite worlds with seed-separated splits and
first-class control conditions. Replaces the flawed legacy ``src/micro_ground`` for the
redesigned study; the legacy package is kept only as reference.
"""
from .world import FactoredWorld, Query, FACTORS_1OBJ
from .data import (
    Condition, CONDITIONS, Vocab, Example,
    build_examples, build_examples_from_queries, baseline_metrics, accuracy_metrics,
    make_encoder, STATE_MODE, query_text,
)
from .model import MGConfig, MGTransformer, save_mg, load_mg

__all__ = [
    "FactoredWorld", "Query", "FACTORS_1OBJ",
    "Condition", "CONDITIONS", "Vocab", "Example",
    "build_examples", "baseline_metrics", "accuracy_metrics",
    "make_encoder", "STATE_MODE", "query_text",
    "MGConfig", "MGTransformer", "save_mg", "load_mg",
]
