"""
Counterfactual consistency testing (novelty item 6).

For each verified claim, apply a controlled perturbation to its evidence and check
whether verify()'s verdict changes the way it should:
  - "relation-breaking" perturbations remove or contradict the evidence a supported
    verdict rests on -- the verdict SHOULD change (high consistency = good).
  - "irrelevant" perturbations touch nothing the rule actually depends on -- the
    verdict should NOT change (low consistency = good; here "consistency" means
    "the verdict flipped," so for irrelevant perturbations we want this near 0).

This directly implements CC(T) = (1/|C|) * sum_{c in C} 1[V(c) != V(T(c))] from the
original design proposal, reported separately per perturbation type, since a single pooled
number would hide the fact that different perturbations are *supposed* to behave
differently.

Applies to the existing rule-based verify() right now -- nothing here waits on the
learned verifier (item 1). Once that exists, run this same harness against it too.
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Optional

from graph_utils import CitationGraph, DEPENDENCY_RELATIONS, verify

VerifierFn = Callable[[str, str, str, CitationGraph], str]


@dataclass
class PerturbationResult:
    instance_id: str
    claim_id: str
    paper_a: str
    relation: str
    paper_b: str
    original_verdict: str
    perturbation: str
    perturbed_verdict: str
    changed: bool


# Which perturbations can only possibly break something if the ORIGINAL verdict was
# "supported" -- there is no support to remove from an already-unsupported claim, so
# scoring those trials would just dilute the signal with guaranteed non-flips.
# Fixed after a first run surfaced this: an earlier version of this harness scored
# "should flip" against the FULL claim population, most of which is already
# unsupported, and produced misleadingly low "consistency" numbers that reflected a
# flaw in the test's population filter, not in the verifier. See README note below.
BREAKING_PERTURBATIONS = {"remove_direct_edge", "reverse_direction", "publication_order", "relation_label_swap"}
STABILITY_PERTURBATIONS = {"unrelated_noise_control"}  # should stay ~0% regardless of original verdict
DESCRIPTIVE_PERTURBATIONS = {"corrupt_edges_30pct"}  # reported as-is, no should/shouldn't claim


# -- The seven perturbations from the original design proposal -----------------------------

def perturb_remove_edge(a, rho, b, graph: CitationGraph, **_) -> tuple[CitationGraph, str]:
    """(1) Remove the direct citation edge, whichever direction it runs."""
    return graph.without_edge(a, b).without_edge(b, a), rho


def perturb_reverse_direction(a, rho, b, graph: CitationGraph, **_) -> tuple[CitationGraph, str]:
    """(2) Reverse citation direction: A cites B becomes B cites A, or vice versa."""
    if graph.cites(a, b):
        return graph.with_reversed_edge(a, b), rho
    if graph.cites(b, a):
        return graph.with_reversed_edge(b, a), rho
    return graph, rho  # no direct edge to reverse -- no-op, correctly produces no flip


def perturb_publication_order(a, rho, b, graph: CitationGraph, **_) -> tuple[CitationGraph, str]:
    """(5) Change publication order by swapping A and B's years."""
    return graph.with_swapped_years(a, b), rho


def perturb_relation_label(a, rho, b, graph: CitationGraph, rng: random.Random, **_) -> tuple[CitationGraph, str]:
    """(6) Preserve the graph but change the relation label to a different taxonomy member."""
    from graph_utils import DEPENDENCY_RELATIONS, NONDEPENDENCY_RELATIONS
    all_relations = DEPENDENCY_RELATIONS | NONDEPENDENCY_RELATIONS
    alt = rng.choice(sorted(all_relations - {rho}))
    return graph, alt


def perturb_corrupt_edges(a, rho, b, graph: CitationGraph, rng: random.Random,
                           corruption_rate: float = 0.3, **_) -> tuple[CitationGraph, str]:
    """(7) Corrupt a percentage of graph edges for A and B specifically (drop refs at random)."""
    return graph.with_corrupted_refs([a, b], corruption_rate, rng), rho


def perturb_unrelated_noise(a, rho, b, graph: CitationGraph, rng: random.Random,
                             all_paper_ids: list, **_) -> tuple[CitationGraph, str]:
    """
    Irrelevant-perturbation control: remove a random edge between two OTHER papers,
    unconnected to A or B. The verdict for (A, rho, B) should never change.
    """
    return graph.without_random_edge_elsewhere({a, b}, rng), rho


PERTURBATIONS = {
    "remove_direct_edge": perturb_remove_edge,
    "reverse_direction": perturb_reverse_direction,
    "publication_order": perturb_publication_order,
    "relation_label_swap": perturb_relation_label,
    "corrupt_edges_30pct": perturb_corrupt_edges,
    "unrelated_noise_control": perturb_unrelated_noise,
    # "replace_citation_context" intentionally omitted: the rule-based verifier uses
    # no text evidence, so this perturbation is inapplicable until the learned
    # multimodal verifier (item 1) exists. Re-run this harness against it then.
}


def run_counterfactual_suite(
    claims: List[dict],
    graph: CitationGraph,
    verifier: VerifierFn = verify,
    all_paper_ids: Optional[list] = None,
    seed: int = 0,
    limit: Optional[int] = None,
) -> List[PerturbationResult]:
    rng = random.Random(seed)
    all_paper_ids = all_paper_ids or graph.all_ids()
    results = []

    pool = claims if limit is None else claims[:limit]
    for c in pool:
        a, rho, b = c["paper_a"], c["relation"], c["paper_b"]
        if not (a and rho and b) or a not in graph or b not in graph:
            continue
        original = verifier(a, rho, b, graph)
        if original == "unverifiable":
            continue

        for name, fn in PERTURBATIONS.items():
            g2, rho2 = fn(a, rho, b, graph, rng=rng, all_paper_ids=all_paper_ids)
            perturbed = verifier(a, rho2, b, g2)
            results.append(PerturbationResult(
                instance_id=c["instance_id"], claim_id=c["claim_id"],
                paper_a=a, relation=rho, paper_b=b,
                original_verdict=original, perturbation=name,
                perturbed_verdict=perturbed, changed=(perturbed != original),
            ))
    return results


def summarize(results: List[PerturbationResult]) -> dict:
    """
    Reports flip rate conditioned on the population that's actually meaningful for
    each perturbation category:
      - breaking:     only claims originally "supported" (nothing to break otherwise).
                       High flip rate = good (the rule's support is genuinely load-bearing).
      - stability:    all claims. Near-0% flip rate = good (irrelevant noise is ignored).
      - descriptive:  all claims, reported with no should/shouldn't framing.
    """
    from collections import defaultdict
    by_pert = defaultdict(list)
    for r in results:
        by_pert[r.perturbation].append(r)

    summary = {}
    for pert, rs in by_pert.items():
        if pert in BREAKING_PERTURBATIONS:
            pool = [r for r in rs if r.original_verdict == "supported"]
            category = "breaking"
        elif pert in STABILITY_PERTURBATIONS:
            pool = rs
            category = "stability"
        else:
            pool = rs
            category = "descriptive"
        n = len(pool)
        flips = sum(r.changed for r in pool)
        summary[pert] = {
            "category": category,
            "n_scored": n,
            "n_total_trials": len(rs),
            "flip_rate": flips / n if n else None,
        }
    return summary


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from graph_utils import load_relational_claims

    root = Path(__file__).resolve().parents[2]
    refs_cache = root / "benchmarks/final/relational/citecheck_refs_cache.json"
    cite_results = root / "benchmarks/final/relational/citecheck_results.json"
    bench = root / "benchmarks/final/nb2-output/scireqa_full_relational_v1.1_humanmerged.jsonl"

    graph = CitationGraph.load(refs_cache)
    claims = load_relational_claims(bench, cite_results)
    claims = [c for c in claims if c["recorded_verdict"] is not None]
    print(f"Loaded {len(claims)} relational claims with recorded verdicts, graph has {len(graph)} nodes.")

    results = run_counterfactual_suite(claims, graph, seed=0)
    print(f"Ran {len(results)} (claim, perturbation) trials.\n")

    summary = summarize(results)
    print(f"{'perturbation':26s} {'category':12s} {'n scored':>9s} {'/total':>7s} {'flip rate':>10s}")
    for pert, s in summary.items():
        print(f"{pert:26s} {s['category']:12s} {s['n_scored']:9d} {s['n_total_trials']:7d} {s['flip_rate']:10.1%}")

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "counterfactual_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "n_trials": len(results),
                    "n_claims": len(claims)}, f, indent=2)
    print(f"\nSaved summary to {out_dir / 'counterfactual_results.json'}")
