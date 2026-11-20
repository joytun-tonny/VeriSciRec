"""
Feature extraction for the learned multimodal verifier (novelty item 1).

h_c (claim):  relation type one-hot (5 dims) + is_dependency flag.
h_g (graph):  the structural signals the rule already computes (c_AB, c_BA, s_AB,
              o_AB) PLUS degree-normalized bibliographic coupling and node degrees,
              which the rule does NOT use -- giving the learned model strictly more
              graph information than the fixed rule has access to.
h_t (text):   TF-IDF cosine similarity between the two papers' abstracts (reusing the
              same index-building code as hard_negatives.py's topically_similar_swap,
              so both stay consistent).

SCOPED, NOT THE FULL "citation contexts + full text" design in the original design proposal:
citation-context sentences ARE fetchable (confirmed: 70-84% edge coverage via the
Semantic Scholar API, see this session's coverage probe) but not cached at the scale
needed for 11,517+ claims yet -- that's a rate-limited, multi-hour fetch better run
as its own background job on Kaggle with an API key, not inline here. Abstract-level
TF-IDF similarity is v1's text modality; swap in real citation-context embeddings
once contexts_api.py's bulk fetch has run, without changing this module's interface.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from graph_utils import CitationGraph, DEPENDENCY_RELATIONS, NONDEPENDENCY_RELATIONS, TAU_DEFAULT

ALL_RELATIONS = sorted(DEPENDENCY_RELATIONS | NONDEPENDENCY_RELATIONS)
REL_INDEX = {r: i for i, r in enumerate(ALL_RELATIONS)}

FEATURE_NAMES = [
    # h_c -- claim / relation type (one-hot + derived)
    *[f"rel_{r}" for r in ALL_RELATIONS],
    "is_dependency",
    # h_g -- graph, beyond what the rule uses
    "c_ab", "c_ba", "shared_refs_log1p", "coupling_strength", "order_ok",
    "years_resolved", "year_diff_signed", "deg_a_log1p", "deg_b_log1p",
    # h_t -- text
    "abstract_cosine_sim",
]


def claim_features(a: str, rho: str, b: str, graph: CitationGraph,
                    tfidf_vectors: Dict[str, dict], tau: int = TAU_DEFAULT) -> np.ndarray:
    x = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}

    if rho in REL_INDEX:
        x[idx[f"rel_{rho}"]] = 1.0
    x[idx["is_dependency"]] = float(rho in DEPENDENCY_RELATIONS)

    c_ab = float(graph.cites(a, b))
    c_ba = float(graph.cites(b, a))
    s_ab = graph.shared_refs(a, b)
    deg_a, deg_b = len(graph.refs(a)), len(graph.refs(b))
    coupling = s_ab / math.sqrt((deg_a + 1) * (deg_b + 1))  # normalized, unlike the raw rule's s_AB >= tau

    ya, yb = graph.year(a), graph.year(b)
    years_resolved = ya is not None and yb is not None
    order_ok = 1.0 if (not years_resolved or ya >= yb) else 0.0
    year_diff = float(ya - yb) if years_resolved else 0.0

    x[idx["c_ab"]] = c_ab
    x[idx["c_ba"]] = c_ba
    x[idx["shared_refs_log1p"]] = math.log1p(s_ab)
    x[idx["coupling_strength"]] = coupling
    x[idx["order_ok"]] = order_ok
    x[idx["years_resolved"]] = float(years_resolved)
    x[idx["year_diff_signed"]] = year_diff
    x[idx["deg_a_log1p"]] = math.log1p(deg_a)
    x[idx["deg_b_log1p"]] = math.log1p(deg_b)

    va, vb = tfidf_vectors.get(a), tfidf_vectors.get(b)
    if va and vb:
        if len(va) > len(vb):
            va, vb = vb, va
        sim = sum(w * vb.get(t, 0.0) for t, w in va.items())
    else:
        sim = 0.0
    x[idx["abstract_cosine_sim"]] = sim

    return x


def build_dataset(claims: List[dict], graph: CitationGraph, tfidf_vectors: Dict[str, dict],
                   labels: Dict[Tuple[str, str], int]) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    labels: maps (instance_id, claim_id) -> 0/1 (0 = unsupported/false, 1 = supported/true)
    Returns (X, y, meta) for claims that have a label.
    """
    X, y, meta = [], [], []
    for c in claims:
        key = (c["instance_id"], c["claim_id"])
        if key not in labels:
            continue
        a, rho, b = c["paper_a"], c["relation"], c["paper_b"]
        if not (a and rho and b) or a not in graph or b not in graph:
            continue
        X.append(claim_features(a, rho, b, graph, tfidf_vectors))
        y.append(labels[key])
        meta.append(c)
    return np.stack(X) if X else np.zeros((0, len(FEATURE_NAMES))), np.array(y), meta
